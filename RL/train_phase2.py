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

from train_phase1 import (Curriculum, CURRICULUMS, TASK_NAMES, TASK_SPECS, N_TASKS, SEEDS, VALIDATION_SEEDS, TOTAL_TIMESTEPS,
                          EVAL_FREQ, N_EVAL_EPISODES_PER_ZONE, SNAPSHOT_EVERY, train_one_run,
                          _overall_success, _slug, enforce_held_out_exclusion)


def start_vllm(model: str, port: int, log_file: str = "vllm.log", timeout: int = 1200,
               gpu_memory_utilization: float = 0.8, max_model_len: int = 8192,
               max_num_seqs: int = 8) -> subprocess.Popen:
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
    def __init__(self, base_url: str, model: str = "Qwen/Qwen2.5-14B-Instruct", api_key: str = "not-needed",
                 temperature: float = 0.2, max_tokens: int = 700, max_retries: int = 3, timeout: float = 120.0):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def chat(self, messages: List[Dict[str, str]]) -> str:
        kwargs = dict(model=self.model, messages=messages,
                      temperature=self.temperature, max_tokens=self.max_tokens,
                      extra_body={"chat_template_kwargs": {"enable_thinking": False}})
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

    def generate_json(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])


LADDER_DESCRIPTION = (
    "A robot arm is trained on 5 manipulation tasks: reach, push, slide, pickandplace, stack. "
    "reach/push/slide use a fixed, non-grasping gripper. pickandplace and stack require grasping. "
    "stack is harder than pickandplace: it needs two successive grasp-and-place actions, so it can only "
    "be learned once pickandplace is reliable. Each task is split into 3 distance levels (near, medium, far) "
    "= how far the target is from the centre of that task's target region. Every (task, distance_level) pair "
    "is a 'bin'. Success = the object/gripper ends within a small threshold of the target."
)

INTERFACE_FACTS = (
    "How your plan is used: the bins you list are consumed one per episode, in order, and cyclically, until "
    "your next plan. The list is therefore the allocation of the next episodes: a bin listed several times "
    "gets several episodes, a bin that is not listed gets none.\n"
    "How you are evaluated: after training, the policy is tested on held-out target zones spread over the whole "
    "target range of every task (near, medium and far), and the score is the average success over the 5 tasks. "
    "One single neural policy is shared by all tasks and bins; your plans are the only thing you control. "
    "A task that is no longer trained can be forgotten."
)

EPISODE_STEPS = 50


def gate_text(allowed: List[str], status: Dict[str, str]) -> str:
    locked = [t for t in TASK_NAMES if t not in allowed]
    lines = [f"Tasks you may train now: {', '.join(allowed)}."]
    if locked:
        lines.append("Locked tasks (any bin of these tasks in your plan is rejected): "
                     + "; ".join(f"{t} ({status[t]})" for t in locked) + ".")
    return "\n".join(lines)


def sample_on_shell(spec, frac: float, rng: np.random.Generator) -> np.ndarray:
    low, high = spec.goal_low, spec.goal_high
    free = (high - low) > 1e-9
    center = (low + high) / 2.0
    half_diag = np.linalg.norm(((high - low) / 2.0)[free])
    g = rng.normal(size=int(free.sum()))
    direction = np.zeros(len(low))
    direction[free] = g / max(np.linalg.norm(g), 1e-9)
    return np.clip(center + direction * frac * half_diag, low, high).astype(np.float32)


def _rate(h: List[bool], min_n: int) -> Optional[float]:
    return round(float(np.mean(h)), 2) if len(h) >= min_n else None


class SingleAgentLLMCurriculum(Curriculum):
    DIFFICULTY_LEVELS = ["near", "medium", "far"]
    DIFFICULTY_TO_FRACTION = [0.20, 0.55, 0.80]
    WINDOW = 20
    YIELD_WINDOW = 100
    YIELD_MIN = 30
    IMPROVE_DELTA = 0.05
    COVERAGE_HORIZON = 1000
    MAX_LLM_ERRORS = 20
    MEMORY_SIZE = 4
    FAMILY = ["reach", "push", "slide"]
    FAMILY_THRESHOLD = 0.70
    STACK_THRESHOLD = 0.50

    def __init__(self, llm_client: LLMClient, revision_freq: int = 100, band: float = 0.15,
                 nb_obj_prompt: int = 20, log_path: Optional[str] = None, total_timesteps: Optional[int] = None,
                 gate: bool = True, gate_budget_frac: Optional[float] = None, use_yield: bool = True,
                 use_coverage: bool = True, use_probe: bool = True, max_tool_calls: int = 3,
                 max_probe_episodes: int = 10):
        self.llm_client = llm_client
        self.revision_freq = revision_freq
        self.band = band
        self.nb_obj_prompt = nb_obj_prompt
        self.log_path = log_path
        self.total_timesteps = total_timesteps
        self.gate = gate
        self.gate_budget_frac = gate_budget_frac
        self.use_yield = use_yield
        self.use_coverage = use_coverage
        self.use_probe = use_probe
        self.max_tool_calls = max_tool_calls
        self.max_probe_episodes = max_probe_episodes

        self.bins = [(t, l) for t in TASK_NAMES for l in self.DIFFICULTY_LEVELS]
        self.bin_index = {b: i for i, b in enumerate(self.bins)}
        self.history: Dict[int, List[bool]] = {i: [] for i in range(len(self.bins))}
        self.bin_episodes: Dict[int, List[int]] = {i: [] for i in range(len(self.bins))}
        self.task_history: Dict[str, List[bool]] = {t: [] for t in TASK_NAMES}
        self.best_rate: Dict[str, float] = {t: -1.0 for t in TASK_NAMES}
        self.improved_at: Dict[str, int] = {t: 0 for t in TASK_NAMES}
        self.recent_tasks: List[str] = []
        self.last_probe: Dict[int, Dict] = {}

        self.pnp_unlocked = not gate
        self.stack_unlocked = not gate
        self.model = None
        self.probe_env = None

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
        self.n_rejected_locked = 0
        self.n_llm_errors = 0
        self.n_tool_calls = 0
        self.n_probe_episodes = 0
        self.n_probe_steps = 0

    def attach(self, model, probe_env) -> None:
        self.model = model
        self.probe_env = probe_env

    def _rolling(self, task: str) -> Optional[float]:
        h = self.task_history[task]
        return float(np.mean(h[-self.WINDOW:])) if len(h) >= self.WINDOW else None

    def _progress(self) -> Optional[float]:
        if not self.total_timesteps:
            return None
        return min(self.episode_count * EPISODE_STEPS / self.total_timesteps, 1.0)

    def _update_gate(self) -> None:
        if not self.gate:
            return
        if not self.pnp_unlocked:
            fam = [self._rolling(t) for t in self.FAMILY]
            mastered = all(r is not None and r >= self.FAMILY_THRESHOLD for r in fam)
            p = self._progress()
            by_budget = self.gate_budget_frac is not None and p is not None and p >= self.gate_budget_frac
            if mastered or by_budget:
                self.pnp_unlocked = True
        if self.pnp_unlocked and not self.stack_unlocked:
            r = self._rolling("pickandplace")
            if r is not None and r >= self.STACK_THRESHOLD:
                self.stack_unlocked = True

    def allowed_tasks(self) -> List[str]:
        allowed = list(self.FAMILY)
        if self.pnp_unlocked:
            allowed.append("pickandplace")
        if self.stack_unlocked:
            allowed.append("stack")
        return allowed

    def _gate_status(self) -> Dict[str, str]:
        def fmt(r):
            return "not enough episodes yet" if r is None else f"{r:.2f}"
        fam = ", ".join(f"{t} {fmt(self._rolling(t))}" for t in self.FAMILY)
        pnp = (f"unlocks when reach, push and slide each reach {self.FAMILY_THRESHOLD:.2f} training success over "
               f"their last {self.WINDOW} episodes; now: {fam}")
        if self.gate_budget_frac is not None:
            pnp += f", or when {100 * self.gate_budget_frac:.0f}% of the budget is used"
        stack = (f"unlocks when pickandplace reaches {self.STACK_THRESHOLD:.2f} training success over its last "
                 f"{self.WINDOW} episodes; now: {fmt(self._rolling('pickandplace'))}")
        return {"reach": "", "push": "", "slide": "", "pickandplace": pnp, "stack": stack}

    def _task_yield(self) -> Dict[str, Dict]:
        last_tasks = self.recent_tasks[-self.COVERAGE_HORIZON:]
        out = {}
        for t in TASK_NAMES:
            h = self.task_history[t]
            w = self.YIELD_WINDOW
            last, prev = h[-w:], h[-2 * w:-w]
            s_last, s_prev = _rate(last, self.YIELD_MIN), _rate(prev, self.YIELD_MIN)
            change, significant = None, None
            if s_last is not None and s_prev is not None:
                change = round(s_last - s_prev, 2)
                n1, n2 = len(last), len(prev)
                pooled = (sum(last) + sum(prev)) / (n1 + n2)
                se = np.sqrt(max(pooled * (1 - pooled), 1e-9) * (1 / n1 + 1 / n2))
                significant = bool(abs(s_last - s_prev) / se > 1.96)
            out[t] = {"episodes_total": len(h),
                      "share_of_last_1000_episodes": round(last_tasks.count(t) / len(last_tasks), 2) if last_tasks else 0.0,
                      "success_last_100": s_last, "success_previous_100": s_prev,
                      "change": change, "change_is_significant": significant,
                      "episodes_of_this_task_since_last_improvement": (len(h) - self.improved_at[t]) if h else None}
        return out

    def _coverage(self) -> Dict[str, Dict]:
        out = {}
        horizon_start = self.episode_count - self.COVERAGE_HORIZON
        for i, (t, l) in enumerate(self.bins):
            eps = self.bin_episodes[i]
            recent = sum(1 for e in eps[-self.COVERAGE_HORIZON:] if e >= horizon_start)
            probe = self.last_probe.get(i)
            out[f"{t}/{l}"] = {
                "trained_in_last_1000_episodes": recent,
                "episodes_since_last_trained": (self.episode_count - eps[-1]) if eps else None,
                "training_success_last_20": _rate(self.history[i][-self.WINDOW:], 5),
                "last_probe": ({"success": probe["success"], "episodes": probe["episodes"],
                                "episodes_ago": self.episode_count - probe["at"]} if probe else None),
            }
        return out

    def _task_success_now(self) -> Dict[str, Optional[float]]:
        return {t: _rate(self.task_history[t][-self.WINDOW:], 5) for t in TASK_NAMES}

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

    def _budget_text(self) -> str:
        p = self._progress()
        if p is None:
            return ""
        left = max(self.total_timesteps - self.episode_count * EPISODE_STEPS, 0) // EPISODE_STEPS
        return f"Training budget: {100 * p:.0f}% used, about {left} episodes left.\n\n"

    def _probe_available(self) -> bool:
        return self.use_probe and self.model is not None and self.probe_env is not None

    def _tools_text(self) -> str:
        if not self._probe_available():
            return ""
        return (
            "Tool available before you submit your plan:\n"
            '{"tool": "probe", "task": "<task>", "level": "<near|medium|far>", "episodes": <integer>} tests the '
            "current policy, without training it, on that many episodes of that bin, and returns its success rate. "
            "Use it to check what the statistics cannot show, for example whether a task that is no longer "
            "trained has been forgotten, or whether a level that was never trained already works. Any task can "
            f"be probed, locked or not. For this plan you have at most {self.max_tool_calls} tool calls and "
            f"{self.max_probe_episodes} probe episodes in total.\n"
            "At each turn, reply with ONE JSON object: either a tool call, or your final plan.\n\n"
        )

    @staticmethod
    def _lines(d: Dict[str, Dict]) -> str:
        return "\n".join(f"{k}: {json.dumps(v)}" for k, v in d.items())

    def build_prompt(self) -> str:
        previous = [f"{self.bins[i][0]}/{self.bins[i][1]}" for i in self.plan]
        parts = [LADDER_DESCRIPTION, "", INTERFACE_FACTS, ""]
        if self.gate:
            parts += [gate_text(self.allowed_tasks(), self._gate_status()), ""]
        parts.append(f"Training episodes so far: {self.episode_count}.")
        text = "\n".join(parts) + "\n" + self._budget_text()
        if self.use_yield:
            text += ("Task yield (training success of each task over its last 100 episodes and the 100 before; "
                     "change_is_significant = the change is unlikely to be noise; "
                     "episodes_of_this_task_since_last_improvement = how long the task has stalled):\n"
                     f"{self._lines(self._task_yield())}\n\n")
        else:
            text += f"Task training success over the last {self.WINDOW} episodes: {json.dumps(self._task_success_now())}\n\n"
        if self.use_coverage:
            text += ("Coverage of each bin (held-out zones lie at near, medium AND far, so an untrained or untested "
                     "level is a risk for the score):\n"
                     f"{self._lines(self._coverage())}\n\n")
        text += (f"Your previous plan, already executed: {json.dumps(previous)}\n\n"
                 "Outcome of your most recent plans, oldest first (change = success now minus success when the "
                 "plan started, null if not measurable):\n"
                 f"{self._memory_text()}\n\n"
                 f"{self._tools_text()}"
                 f"Final plan: the plan for the next {self.revision_freq} episodes, as a list of exactly "
                 f"{self.nb_obj_prompt} bins used cyclically, to maximise the final score. "
                 'Final plan format: {"analysis": "<at most two sentences>", '
                 f'"bins": [<exactly {self.nb_obj_prompt} strings of the form "task/level">]}}, for example '
                 f'["reach/near", "reach/medium", "push/near"]. Valid tasks: {", ".join(self.allowed_tasks())}. '
                 "Valid levels: near, medium, far.")
        return text

    @staticmethod
    def _norm(x) -> str:
        return re.sub(r"[^a-z]", "", str(x).lower())

    @staticmethod
    def _parse_obj(raw: str):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.M).strip()
        for candidate in (text, *re.findall(r"\{.*\}", text, flags=re.S)):
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
        return None

    def _pair_from_item(self, item) -> Optional[Tuple[str, str]]:
        if isinstance(item, dict):
            task, level = item.get("task", ""), item.get("distance_level", item.get("level", ""))
        elif isinstance(item, str) and "/" in item:
            task, level = item.split("/", 1)
        else:
            return None
        return self._norm(task), self._norm(level)

    def _parse_bins(self, raw: str, obj) -> List[int]:
        items = obj.get("bins") if isinstance(obj, dict) else obj if isinstance(obj, list) else None
        if isinstance(items, list):
            pairs = [self._pair_from_item(item) for item in items]
        else:
            text = raw or ""
            cut = text.find('"bins"')
            tail = text[cut:] if cut >= 0 else text
            found = re.findall(r'"([A-Za-z_ ]+)/(near|medium|far)"', tail)
            pairs = [(self._norm(t), self._norm(l)) for t, l in found]
        allowed = set(self.allowed_tasks())
        idxs: List[int] = []
        for pair in pairs:
            if pair not in self.bin_index:
                self.n_invalid_items += 1
            elif pair[0] not in allowed:
                self.n_rejected_locked += 1
            else:
                idxs.append(self.bin_index[pair])
        return idxs

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        self.n_llm_calls += 1
        try:
            return self.llm_client.chat(messages)
        except (APIConnectionError, APITimeoutError, RateLimitError):
            self.n_llm_errors += 1
            if self.n_llm_errors >= self.MAX_LLM_ERRORS:
                raise RuntimeError(f"{self.n_llm_errors} LLM calls failed: is the vLLM server still up?")
            return ""

    def _probe(self, task: str, level: str, n: int) -> Dict:
        name = TASK_NAMES[[self._norm(t) for t in TASK_NAMES].index(task)] if task in [self._norm(t) for t in TASK_NAMES] else None
        if name is None or level not in self.DIFFICULTY_LEVELS:
            return {"error": "unknown task or level"}
        if n <= 0:
            return {"error": "no probe episodes left"}
        spec = TASK_SPECS[name]
        frac_c = self.DIFFICULTY_TO_FRACTION[self.DIFFICULTY_LEVELS.index(level)]
        rng = self.probe_env.rng
        successes = []
        for _ in range(n):
            frac = rng.uniform(max(frac_c - self.band, 0.0), min(frac_c + self.band, 1.0))
            target = enforce_held_out_exclusion(name, sample_on_shell(spec, frac, rng), rng)
            self.probe_env.set_forced_episode(spec.task_id, target)
            obs, info = self.probe_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _r, terminated, truncated, info = self.probe_env.step(action)
                self.n_probe_steps += 1
                done = terminated or truncated
            successes.append(bool(info["is_success"]))
        self.n_probe_episodes += n
        rate = round(float(np.mean(successes)), 2)
        self.last_probe[self.bin_index[(name, level)]] = {"success": rate, "episodes": n, "at": self.episode_count}
        return {"task": name, "level": level, "episodes": n, "success": rate}

    def _run_tool(self, obj: Dict, probe_left: int) -> Dict:
        if self._norm(obj.get("tool", "")) != "probe":
            return {"error": "unknown tool; the only tool is probe"}
        try:
            n = int(obj.get("episodes", obj.get("n", 5)))
        except (TypeError, ValueError):
            n = 5
        n = max(0, min(n, probe_left))
        return self._probe(self._norm(obj.get("task", "")), self._norm(obj.get("level", "")), n)

    def _fallback_plan(self) -> List[int]:
        allowed = self.allowed_tasks()
        weakest = sorted(allowed, key=lambda t: (self._rolling(t) if self._rolling(t) is not None else 0.0))
        order = [self.bin_index[(t, l)] for t in weakest for l in self.DIFFICULTY_LEVELS]
        return [order[k % len(order)] for k in range(self.nb_obj_prompt)]

    def _run_dialogue(self, prompt: str, allow_tools: bool = True) -> Tuple[str, object, List[Dict]]:
        messages = [{"role": "user", "content": prompt}]
        tool_calls, probe_left = 0, self.max_probe_episodes
        transcript: List[Dict] = []
        raw, obj = "", None
        retried = False
        for _ in range(self.max_tool_calls + 3):
            raw = self._call_llm(messages)
            obj = self._parse_obj(raw)
            if obj is None and not retried:
                retried = True
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Your reply was not a valid JSON object. Reply with ONE "
                                                            "JSON object only: a tool call or your final plan."})
                continue
            is_tool = isinstance(obj, dict) and "tool" in obj and "bins" not in obj
            if not is_tool:
                break
            messages.append({"role": "assistant", "content": raw})
            if not allow_tools or not self._probe_available() or tool_calls >= self.max_tool_calls:
                messages.append({"role": "user", "content": "No more tool calls are allowed. Reply now with your "
                                                            "final plan JSON only."})
                continue
            tool_calls += 1
            self.n_tool_calls += 1
            result = self._run_tool(obj, probe_left)
            probe_left -= int(result.get("episodes", 0))
            transcript.append({"call": obj, "result": result})
            messages.append({"role": "user", "content":
                             f"Tool result: {json.dumps(result)}. Tool calls left: {self.max_tool_calls - tool_calls}; "
                             f"probe episodes left: {probe_left}. Call the tool again or reply with your final plan."})
        return raw, obj, transcript

    def _install_plan(self, idxs: List[int]) -> bool:
        fallback = not idxs
        if idxs:
            self.plan = idxs[: self.nb_obj_prompt]
        else:
            self.n_fallbacks += 1
            if not self.plan or any(self.bins[i][0] not in self.allowed_tasks() for i in self.plan):
                self.plan = self._fallback_plan()
        self.n_revisions += 1
        self.pos = 0
        self.cur_alloc = Counter()
        self.start_success = self._task_success_now()
        return fallback

    def _revise(self) -> None:
        self._update_gate()
        self._close_memory()
        prompt = self.build_prompt()
        raw, obj, transcript = self._run_dialogue(prompt, allow_tools=True)
        idxs = self._parse_bins(raw, obj)
        fallback = self._install_plan(idxs)
        self._append_log(prompt, raw, idxs, fallback, transcript)
        if self.n_revisions <= 3 or self.n_revisions % 50 == 0:
            names = [f"{self.bins[i][0]}/{self.bins[i][1]}" for i in self.plan]
            print(f"[llm] revision {self.n_revisions} (episode {self.episode_count}, fallback={fallback}, "
                  f"tools={len(transcript)}): {names}", flush=True)

    def _append_log(self, prompt: str, raw: str, idxs: List[int], fallback: bool, transcript: List[Dict]) -> None:
        if not self.log_path:
            return
        found = re.search(r'"analysis"\s*:\s*"((?:[^"\\]|\\.)*)', raw or "")
        record = {"episode": self.episode_count, "revision": self.n_revisions, "prompt": prompt, "raw": raw,
                  "analysis": found.group(1) if found else None, "tools": transcript,
                  "plan": [f"{self.bins[i][0]}/{self.bins[i][1]}" for i in self.plan],
                  "parsed": [f"{self.bins[i][0]}/{self.bins[i][1]}" for i in idxs], "fallback": fallback,
                  "allowed_tasks": self.allowed_tasks(), "n_llm_calls": self.n_llm_calls,
                  "n_fallbacks": self.n_fallbacks, "n_invalid_items": self.n_invalid_items,
                  "n_rejected_locked": self.n_rejected_locked, "n_probe_episodes": self.n_probe_episodes}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

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
            self.bin_episodes[self.last_bin].append(self.episode_count)
            task = self.bins[self.last_bin][0]
            h = self.task_history[task]
            h.append(bool(success))
            if len(h) >= self.YIELD_MIN:
                r = float(np.mean(h[-self.YIELD_WINDOW:]))
                if r > self.best_rate[task] + self.IMPROVE_DELTA:
                    self.best_rate[task] = r
                    self.improved_at[task] = len(h)
            self.recent_tasks.append(task)
            self.cur_alloc[task] += 1
            if len(self.recent_tasks) > 2 * self.COVERAGE_HORIZON:
                self.recent_tasks = self.recent_tasks[-self.COVERAGE_HORIZON:]
        self.episode_count += 1

    def snapshot(self) -> Dict:
        return {"plan": [f"{self.bins[i][0]}/{self.bins[i][1]}" for i in self.plan],
                "allowed_tasks": self.allowed_tasks(), "n_llm_calls": self.n_llm_calls,
                "n_revisions": self.n_revisions, "n_fallbacks": self.n_fallbacks,
                "n_tool_calls": self.n_tool_calls, "n_probe_episodes": self.n_probe_episodes}

    def llm_stats(self) -> Dict:
        return {"n_llm_calls": self.n_llm_calls, "n_revisions": self.n_revisions,
                "n_fallbacks": self.n_fallbacks, "n_invalid_items": self.n_invalid_items,
                "n_rejected_locked": self.n_rejected_locked, "n_llm_errors": self.n_llm_errors,
                "n_tool_calls": self.n_tool_calls, "n_probe_episodes": self.n_probe_episodes,
                "n_probe_steps": self.n_probe_steps, "episodes": self.episode_count,
                "pnp_unlocked": self.pnp_unlocked, "stack_unlocked": self.stack_unlocked,
                "revision_freq": self.revision_freq}


CURRICULUMS["single-agent LLM"] = SingleAgentLLMCurriculum
STRATEGIES = list(CURRICULUMS.keys())

NAME = "single-agent LLM"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS + VALIDATION_SEEDS)
    parser.add_argument("--llm-url", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--plan-len", type=int, default=20)
    parser.add_argument("--revision-freq", type=int, default=100)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--gate-budget-frac", type=float, default=None)
    parser.add_argument("--no-yield", action="store_true")
    parser.add_argument("--no-coverage", action="store_true")
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--max-probe-episodes", type=int, default=10)
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    parser.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY)
    args = parser.parse_args()
    seed = args.seed
    flags = [("no gate", "nogate", args.no_gate), ("no yield", "noyield", args.no_yield),
             ("no coverage", "nocoverage", args.no_coverage), ("no probe", "noprobe", args.no_probe)]
    name = NAME + "".join(f" ({label})" for label, _, on in flags if on)
    suffix = "".join(f"_{tag}" for _, tag, on in flags if on)

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
                            max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs)

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
            curriculum = SingleAgentLLMCurriculum(
                llm, revision_freq=args.revision_freq, nb_obj_prompt=args.plan_len, log_path=log_path,
                total_timesteps=args.total_timesteps, gate=not args.no_gate,
                gate_budget_frac=args.gate_budget_frac, use_yield=not args.no_yield,
                use_coverage=not args.no_coverage, use_probe=not args.no_probe,
                max_tool_calls=args.max_tool_calls, max_probe_episodes=args.max_probe_episodes)
            created.append(curriculum)
            return curriculum

        if os.path.exists(log_path):
            os.remove(log_path)
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