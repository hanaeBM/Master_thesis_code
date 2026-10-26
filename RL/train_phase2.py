import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI, BadRequestError, APIConnectionError, APITimeoutError, RateLimitError

from train_phase1 import (Curriculum, CURRICULUMS, TASK_NAMES, TASK_SPECS, N_TASKS, SEEDS, TOTAL_TIMESTEPS,
                          EVAL_FREQ, N_EVAL_EPISODES_PER_ZONE, SNAPSHOT_EVERY, train_one_run,
                          _overall_success, _slug)


def start_vllm(model: str, port: int, log_file: str = "vllm.log", timeout: int = 1200,
               gpu_memory_utilization: float = 0.8, max_model_len: int = 8192, max_num_seqs: int = 8) -> subprocess.Popen:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.path.join(sys.prefix, "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    if os.path.exists(os.path.join(sys.prefix, "bin", "nvcc")):
        env["CUDA_HOME"] = sys.prefix

    log = open(log_file, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", model,
         "--port", str(port),
         "--gpu-memory-utilization", str(gpu_memory_utilization),
         "--max-model-len", str(max_model_len),
         "--max-num-seqs", str(max_num_seqs)],
        stdout=log, stderr=subprocess.STDOUT, env=env,
    )

    url = f"http://127.0.0.1:{port}/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM stopped during start-up, see {log_file}")
        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"vLLM server ready on port {port} ({time.time() - t0:.0f}s)", flush=True)
            return proc
        except Exception:
            time.sleep(5)

    proc.terminate()
    raise TimeoutError(f"vLLM not ready after {timeout}s, see {log_file}")


def stop_vllm(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()


class LLMClient:
    def __init__(self, base_url: str, model: str = "Qwen/Qwen3.8-27B-FP8", api_key: str = "not-needed",
                 temperature: float = 0.2, max_tokens: int = 700, max_retries: int = 3, timeout: float = 120.0):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def generate_json(self, prompt: str) -> str:
        kwargs = dict(model=self.model, messages=[{"role": "user", "content": prompt}],
                      temperature=self.temperature, max_tokens=self.max_tokens,extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                try:
                    response = self.client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
                except BadRequestError:
                    response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                last_exc = exc
                time.sleep(2.0 * (attempt + 1))
        raise last_exc


LADDER_DESCRIPTION = (
    "A robot arm is trained on 5 manipulation tasks: reach, push, slide, pickandplace, stack. "
    "reach/push/slide use a fixed, non-grasping gripper. pickandplace and stack require grasping; "
    "stack needs essentially the same grasp-and-place competence as pickandplace, and both are only "
    "learnable once basic reaching/pushing competence exists. Each task is split into 3 distance "
    "levels (near, medium, far) = how far the target is from the centre of that task's target region. "
    "Every (task, distance_level) pair is a 'bin'. Success = the object/gripper ends within a small "
    "threshold of the target at the end of an episode."
)


INTERFACE_FACTS = (
    "How your plan is used: the bins you list are consumed one per episode, in order, and cyclically. The list "
    "is therefore the allocation of the next episodes: a bin listed several times gets several episodes, a bin "
    "that is not listed gets none.\n"
    "How you are evaluated: after training, the policy is tested on held-out target zones spread over the whole "
    "target range of every task (near, medium and far), and the score is the average success over the 5 tasks. "
    "One single neural policy is shared by all tasks and bins; your plans are the only thing you control."
)

EXPERT_RULES = (
    "Expert rules to follow:\n"
    "1. Train reach, push and slide first. Give NO episode to pickandplace and stack until reach, push and slide each "
    "reach about 0.70 success. Give NO episode to stack until pickandplace reaches about 0.50.\n"
    "2. Within a task, start at near, move to medium when near is about 0.70, then far when medium is about 0.70.\n"
    "3. Drop bins above 0.90 success (keep one episode of rehearsal per mastered task) and bins that stayed near "
    "0 over several windows (go back to an easier level).\n"
    "4. Spend most of the episodes on the tasks that are currently improving or closest to 0.70."
)

def sample_on_shell(spec, frac: float, rng: np.random.Generator) -> np.ndarray:
    low, high = spec.goal_low, spec.goal_high
    free = (high - low) > 1e-9
    center = (low + high) / 2.0
    half_diag = np.linalg.norm(((high - low) / 2.0)[free])
    g = rng.normal(size=int(free.sum()))
    direction = np.zeros(len(low))
    direction[free] = g / max(np.linalg.norm(g), 1e-9)
    return np.clip(center + direction * frac * half_diag, low, high).astype(np.float32)


class SingleAgentLLMCurriculum(Curriculum):
    DIFFICULTY_LEVELS = ["near", "medium", "far"]
    DIFFICULTY_TO_FRACTION = [0.20, 0.55, 0.80]
    WINDOW = 20
    MAX_LLM_ERRORS = 20
    MEMORY_SIZE = 4

    def __init__(self, llm_client: LLMClient, revision_freq: int = 10, band: float = 0.15,
                 nb_obj_prompt: int = 10, log_path: Optional[str] = None, use_rules: bool = False):
        self.llm_client = llm_client
        self.revision_freq = revision_freq
        self.band = band
        self.nb_obj_prompt = nb_obj_prompt
        self.use_rules = use_rules
        self.log_path = log_path
        self.bins = [(t, l) for t in TASK_NAMES for l in self.DIFFICULTY_LEVELS]
        self.bin_index = {b: i for i, b in enumerate(self.bins)}
        self.history: Dict[int, List[bool]] = {i: [] for i in range(len(self.bins))}
        self.task_history: Dict[str, List[bool]] = {t: [] for t in TASK_NAMES}
        self.recent_tasks: List[str] = []
        self.cur_alloc: Counter = Counter()
        self.start_success: Dict[str, Optional[float]] = {}
        self.memory: List[Dict] = []
        self.plan: List[int] = []
        self.pos = 0
        self.last_bin: Optional[int] = None
        self.episode_count = 0
        self.n_llm_calls = 0
        self.n_revisions = 0
        self.n_fallbacks = 0
        self.n_invalid_items = 0
        self.n_llm_errors = 0

    def _windows(self, h: List[bool], n: int = 3) -> List[Optional[float]]:
        w = self.WINDOW
        out: List[Optional[float]] = []
        for k in range(n - 1, -1, -1):
            end = len(h) - k * w
            start = max(end - w, 0)
            chunk = h[start:end] if end > 0 else []
            out.append(round(float(np.mean(chunk)), 2) if len(chunk) >= 5 else None)
        return out

    def _bin_stats(self) -> Dict[str, Dict]:
        stats = {}
        for i, (task, level) in enumerate(self.bins):
            h = self.history[i]
            stats[f"{task}/{level}"] = {"episodes": len(h), "success_last_3_windows": self._windows(h)}
        return stats

    def _task_summary(self) -> Dict[str, Dict]:
        last = self.recent_tasks[-100:]
        summary = {}
        for t in TASK_NAMES:
            h = self.task_history[t]
            summary[t] = {"episodes": len(h), "success_last_3_windows": self._windows(h),
                          "share_of_last_100_episodes": round(last.count(t) / len(last), 2) if last else 0.0}
        return summary

    def _task_success_now(self) -> Dict[str, Optional[float]]:
        return {t: self._windows(self.task_history[t], 1)[0] for t in TASK_NAMES}

    def _close_memory(self) -> None:
        if not self.start_success:
            return
        now = self._task_success_now()
        change = {}
        for t in TASK_NAMES:
            a, b = self.start_success.get(t), now[t]
            change[t] = round(b - a, 2) if a is not None and b is not None else None
        self.memory.append({"episodes": sum(self.cur_alloc.values()), "allocation": dict(self.cur_alloc),
                            "success_change": change})
        self.memory = self.memory[-self.MEMORY_SIZE:]

    def _memory_text(self) -> str:
        if not self.memory:
            return "none yet (this is your first plan)"
        lines = []
        for k, m in enumerate(self.memory, 1):
            alloc = {t: m["allocation"].get(t, 0) for t in TASK_NAMES}
            lines.append(f"plan -{len(self.memory) - k + 1}: episodes per task {json.dumps(alloc)}; "
                         f"change of task success over those episodes {json.dumps(m['success_change'])}")
        return "\n".join(lines)

    @staticmethod
    def _lines(d: Dict[str, Dict]) -> str:
        return "\n".join(f"{k}: {json.dumps(v)}" for k, v in d.items())

    def build_prompt(self) -> str:
        previous = [f"{self.bins[i][0]}/{self.bins[i][1]}" for i in self.plan]
        return (
            f"{LADDER_DESCRIPTION}\n\n"
            f"{INTERFACE_FACTS}\n\n"
            f"{EXPERT_RULES + chr(10) * 2 if self.use_rules else ''}"
            f"Training episodes so far: {self.episode_count}.\n\n"
            f"Task summary (success_last_3_windows = success rate over three consecutive blocks of "
            f"{self.WINDOW} episodes of that task, oldest first, so you can see the trend; null = fewer than 5 "
            "episodes in that block; share_of_last_100_episodes = how the last 100 episodes were allocated):\n"
            f"{self._lines(self._task_summary())}\n\n"
            "Per-bin statistics, same format:\n"
            f"{self._lines(self._bin_stats())}\n\n"
            f"Your previous plan, already executed: {json.dumps(previous)}\n\n"
            "Outcome of your most recent plans, oldest first (change = success now minus success when the plan "
            "started, null if not measurable):\n"
            f"{self._memory_text()}\n\n"
            f"Write the plan for the next {self.nb_obj_prompt} episodes to maximise the final score. First reason "
            "briefly about what the statistics show and what changed since your previous plan. "
            'Respond with ONLY a JSON object: {"analysis": "<at most two sentences>", '
            f'"bins": [<exactly {self.nb_obj_prompt} strings of the form "task/level">]}}, for example '
            '["reach/near", "reach/medium", "push/near"]. Valid tasks: reach, push, slide, pickandplace, stack. '
            "Valid levels: near, medium, far."
        )

    @staticmethod
    def _norm(x) -> str:
        return re.sub(r"[^a-z]", "", str(x).lower())

    def _pair_from_item(self, item) -> Optional[Tuple[str, str]]:
        if isinstance(item, dict):
            task, level = item.get("task", ""), item.get("distance_level", item.get("level", ""))
        elif isinstance(item, str) and "/" in item:
            task, level = item.split("/", 1)
        else:
            return None
        return self._norm(task), self._norm(level)

    def _parse_llm_choice(self, raw: str) -> List[int]:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.M).strip()
        obj = None
        for candidate in (text, *re.findall(r"\{.*\}|\[.*\]", text, flags=re.S)):
            try:
                obj = json.loads(candidate)
                break
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(obj, dict):
            obj = obj.get("bins")
        pairs: List[Optional[Tuple[str, str]]]
        if isinstance(obj, list):
            pairs = [self._pair_from_item(item) for item in obj]
        else:
            cut = text.find('"bins"')
            tail = text[cut:] if cut >= 0 else text
            found = re.findall(r'"task"\s*:\s*"([^"]+)"\s*,\s*"distance_level"\s*:\s*"([^"]+)"', tail)
            found += re.findall(r'"([A-Za-z_ ]+)/(near|medium|far)"', tail)
            pairs = [(self._norm(t), self._norm(l)) for t, l in found]
        idxs: List[int] = []
        for pair in pairs:
            if pair in self.bin_index:
                idxs.append(self.bin_index[pair])
            else:
                self.n_invalid_items += 1
        return idxs

    def _call_llm(self, prompt: str) -> str:
        self.n_llm_calls += 1
        try:
            return self.llm_client.generate_json(prompt)
        except (APIConnectionError, APITimeoutError, RateLimitError):
            self.n_llm_errors += 1
            if self.n_llm_errors >= self.MAX_LLM_ERRORS:
                raise RuntimeError(f"{self.n_llm_errors} LLM calls failed: is the vLLM server still up?")
            return ""

    def _append_log(self, prompt: str, raw: str, idxs: List[int], fallback: bool) -> None:
        if not self.log_path:
            return
        record = {"episode": self.episode_count, "revision": self.n_revisions, "prompt": prompt, "raw": raw,
                  "plan": [f"{self.bins[i][0]}/{self.bins[i][1]}" for i in self.plan],
                  "parsed": [f"{self.bins[i][0]}/{self.bins[i][1]}" for i in idxs], "fallback": fallback,
                  "n_llm_calls": self.n_llm_calls, "n_fallbacks": self.n_fallbacks,
                  "n_invalid_items": self.n_invalid_items}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _revise(self) -> None:
        self._close_memory()
        prompt = self.build_prompt()
        raw = self._call_llm(prompt)
        idxs = self._parse_llm_choice(raw)
        fallback = not idxs
        if idxs:
            self.plan = idxs[: self.nb_obj_prompt]
        else:
            self.n_fallbacks += 1
            if not self.plan:
                self.plan = list(range(min(self.nb_obj_prompt, len(self.bins))))
        self.n_revisions += 1
        self.pos = 0
        self.cur_alloc = Counter()
        self.start_success = self._task_success_now()
        self._append_log(prompt, raw, idxs, fallback)
        if self.n_revisions <= 3 or self.n_revisions % 50 == 0:
            names = [f"{self.bins[i][0]}/{self.bins[i][1]}" for i in self.plan]
            print(f"[llm] revision {self.n_revisions} (episode {self.episode_count}, fallback={fallback}): {names}",
                  flush=True)

    def propose(self, rng: np.random.Generator) -> Tuple[int, np.ndarray]:
        if not self.plan or self.pos >= self.revision_freq:
            self._revise()
        bin_idx = self.plan[self.pos % len(self.plan)]
        self.pos += 1
        self.last_bin = bin_idx
        name, level = self.bins[bin_idx]
        spec = TASK_SPECS[name]
        frac_c = self.DIFFICULTY_TO_FRACTION[self.DIFFICULTY_LEVELS.index(level)]
        frac = rng.uniform(max(frac_c - self.band, 0.0), min(frac_c + self.band, 1.0))
        return spec.task_id, sample_on_shell(spec, frac, rng)

    def notify_result(self, task_id: int, success: bool) -> None:
        if self.last_bin is not None:
            self.history[self.last_bin].append(bool(success))
            task = self.bins[self.last_bin][0]
            self.task_history[task].append(bool(success))
            self.recent_tasks.append(task)
            self.cur_alloc[task] += 1
            if len(self.recent_tasks) > 500:
                self.recent_tasks = self.recent_tasks[-500:]
        self.episode_count += 1

    def snapshot(self) -> Dict:
        return {"plan": [f"{self.bins[i][0]}/{self.bins[i][1]}" for i in self.plan],
                "n_llm_calls": self.n_llm_calls, "n_revisions": self.n_revisions,
                "n_fallbacks": self.n_fallbacks}

    def llm_stats(self) -> Dict:
        return {"n_llm_calls": self.n_llm_calls, "n_revisions": self.n_revisions,
                "n_fallbacks": self.n_fallbacks, "n_invalid_items": self.n_invalid_items,
                "n_llm_errors": self.n_llm_errors, "episodes": self.episode_count,
                "revision_freq": self.revision_freq}


CURRICULUMS["single-agent LLM"] = SingleAgentLLMCurriculum
STRATEGIES = list(CURRICULUMS.keys())

NAME = "single-agent LLM"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--llm-url", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--plan-len", type=int, default=10)
    parser.add_argument("--revision-freq", type=int, default=None)
    parser.add_argument("--rules", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    parser.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY)
    args = parser.parse_args()
    seed = args.seed
    name = NAME + (" (+rules)" if args.rules else "")
    suffix = "_rules" if args.rules else ""

    results_path = os.path.join(args.out_dir, "results", f"results_llm_single{suffix}_seed{seed}.json")
    stats_path = os.path.join(args.out_dir, "results", f"llm_stats_single{suffix}_seed{seed}.json")
    goal_prefix = os.path.join(args.out_dir, "results", "goals", f"{_slug(name)}_seed{seed}")
    log_path = os.path.join(args.out_dir, "results", "goals", f"{_slug(name)}_seed{seed}_llm_log.jsonl")
    os.makedirs(os.path.dirname(goal_prefix), exist_ok=True)

    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {}

    if name in results:
        print(f"skip {name!r} seed={seed} (already done)")
        print(f"success rate (mean over tasks): {_overall_success(results[name][-1]):.2f}")
        return

    server = None
    if args.llm_url:
        llm_url = args.llm_url
    else:
        port = args.port if args.port is not None else 8000 + seed
        llm_url = f"http://127.0.0.1:{port}/v1"
        vllm_log = os.path.join(args.out_dir, "results", f"vllm{suffix}_seed{seed}.log")
        print(f"starting vLLM ({args.model}) on port {port}, log: {vllm_log}", flush=True)
        server = start_vllm(args.model, port, log_file=vllm_log,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                            max_model_len=args.max_model_len)

    try:
        llm = LLMClient(llm_url, model=args.model)
        print(f"checking LLM server at {llm_url} ...", flush=True)
        try:
            test = llm.generate_json('Reply with the JSON object {"ok": true}')
        except Exception as exc:
            raise SystemExit(f"LLM server unreachable ({type(exc).__name__}: {exc}). Check --llm-url and --model.")
        print(f"LLM server OK, answer: {test[:80]!r}", flush=True)
        created: List[SingleAgentLLMCurriculum] = []

        def factory():
            curriculum = SingleAgentLLMCurriculum(llm, revision_freq=args.revision_freq or args.plan_len,
                                                   nb_obj_prompt=args.plan_len, log_path=log_path,
                                                   use_rules=args.rules)
            created.append(curriculum)
            return curriculum

        t0 = time.time()
        print(f"strategy={name}  seed={seed}  total_timesteps={args.total_timesteps}", flush=True)
        history = train_one_run(factory, seed, total_timesteps=args.total_timesteps, eval_freq=args.eval_freq,
                                n_episodes_per_zone=N_EVAL_EPISODES_PER_ZONE, verbose=0,
                                goal_log_path=goal_prefix, snapshot_every=args.snapshot_every)
        results[name] = history
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        with open(stats_path, "w") as f:
            json.dump(created[0].llm_stats(), f, indent=2)
        print(f"success rate (mean over tasks): {_overall_success(history[-1]):.2f}")
        print(f"seed {seed} done in {time.time() - t0:.0f}s -> {results_path}")
    finally:
        stop_vllm(server)


if __name__ == "__main__":
    main()