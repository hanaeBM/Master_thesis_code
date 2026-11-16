import argparse
import json
import os
import re
import threading
import time
from collections import Counter
from typing import Dict, List, Optional, TypedDict

import numpy as np
from langgraph.graph import StateGraph, START, END

from train_phase1 import (SEEDS, VALIDATION_SEEDS, TOTAL_TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES_PER_ZONE,
                          SNAPSHOT_EVERY, train_one_run, _overall_success, _slug)
from train_phase2 import SingleAgentLLMCurriculum, LLMClient, start_vllm, stop_vllm

ResponseState = TypedDict("ResponseState", {
    "Domain Expert": Optional[List[str]],
    "Risk Expert": Optional[List[str]],
    "RL expert": Optional[List[str]],
    "Aggregator": Optional[List[str]],
})

ROLE_RL = (
    "You are the RL PERFORMANCE MONITOR. Read the statistics: which tasks are improving, which have stalled, "
    "which bins are untrained or untested, and propose the bins that best convert practice into success. You are "
    "the only agent allowed to use the probe tool; its results are shared with the other agents."
)

ROLE_DOMAIN = (
    "You are the DOMAIN EXPERT. Using manipulation reasoning (kinematics, prerequisites between tasks, which "
    "targets are physically harder: e.g. far slide targets need precise momentum control, a lifted pickandplace "
    "goal needs a stable grasp), propose the bins the robot should practise next."
)

ROLE_RISK = (
    "You are the UNCERTAINTY AND RISK agent. Judge the situation under uncertainty: avoid bins that are likely "
    "wasteful right now (targets needing a skill the statistics show is absent, or bins with too little data to "
    "trust), and watch for the risk of forgetting tasks or levels that are no longer trained. Propose the bins you "
    "consider safest and most informative."
)

ROLE_AGGREGATOR = (
    "You are the AGGREGATOR. Three agents with different roles each proposed a plan. Weigh their proposals against "
    "the statistics and produce the single final plan."
)


class SystemMultiAgent(SingleAgentLLMCurriculum):
    NODE_MAPPING = {
        "RLExpertNode": "RL expert",
        "DomainExpertNode": "Domain Expert",
        "RiskExpertNode": "Risk Expert",
    }

    def __init__(self, llm_client: LLMClient, process: str = "aggregator", **kwargs):
        if process not in ("aggregator", "voting"):
            raise NotImplementedError(f"process={process!r}: only 'aggregator' and 'voting' are implemented")
        super().__init__(llm_client, **kwargs)
        self.process = process
        self.lock = threading.Lock()
        self.roles = {"RL expert": ROLE_RL, "Domain Expert": ROLE_DOMAIN, "Risk Expert": ROLE_RISK}
        self.initial_state = {"Domain Expert": None, "Risk Expert": None, "RL expert": None, "Aggregator": None}
        self._tools_in_prompt = False
        self.graph = self._build_graph()

    def _tools_text(self) -> str:
        return super()._tools_text() if self._tools_in_prompt else ""

    def build_agent_prompt(self, role: str, proposals: Optional[Dict] = None, tools: bool = False) -> str:
        self._tools_in_prompt = tools
        try:
            base = self.build_prompt()
        finally:
            self._tools_in_prompt = False
        if proposals is not None:
            proposal_text = (
                "The three agents proposed these plans:\n"
                f"{json.dumps(proposals, indent=2)}\n"
                "Choose the best plan or merge them into one better plan.\n\n"
            )
            cut = base.find("Final plan:")
            base = base[:cut] + proposal_text + base[cut:] if cut >= 0 else base + "\n\n" + proposal_text
        return f"{role}\n\n{base}"

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        with self.lock:
            self.n_llm_calls += 1
        try:
            return self.llm_client.chat(messages)
        except Exception as exc:
            if type(exc).__name__ not in ("APIConnectionError", "APITimeoutError", "RateLimitError"):
                raise
            with self.lock:
                self.n_llm_errors += 1
                too_many = self.n_llm_errors >= self.MAX_LLM_ERRORS
                n = self.n_llm_errors
            if too_many:
                raise RuntimeError(f"{n} LLM calls failed: is the vLLM server still up?")
            return ""

    def _parse_bins(self, raw: str, obj) -> List[int]:
        with self.lock:
            return super()._parse_bins(raw, obj)

    def _names(self, idxs: List[int]) -> List[str]:
        return [f"{self.bins[i][0]}/{self.bins[i][1]}" for i in idxs]

    def _names_to_idxs(self, names) -> List[int]:
        allowed = set(self.allowed_tasks())
        out = []
        for item in names or []:
            pair = self._pair_from_item(item)
            if pair in self.bin_index and pair[0] in allowed:
                out.append(self.bin_index[pair])
        return out

    def _vote(self, state) -> List[int]:
        counts: Counter = Counter()
        for agent in self.roles:
            counts.update(self._names_to_idxs(state.get(agent)))
        total = sum(counts.values())
        if total == 0:
            return []
        n = self.nb_obj_prompt
        quotas = {i: n * c / total for i, c in counts.items()}
        alloc = {i: int(np.floor(q)) for i, q in quotas.items()}
        rest = n - sum(alloc.values())
        for i in sorted(quotas, key=lambda i: (-(quotas[i] - alloc[i]), -counts[i], i))[:rest]:
            alloc[i] += 1
        slots = [((j + 0.5) / k, i) for i, k in alloc.items() if k > 0 for j in range(k)]
        return [i for _, i in sorted(slots)]

    def _log(self, agent: str, prompt: str, raw: str, idxs: List[int], transcript: List[Dict]) -> None:
        if not self.log_path:
            return
        found = re.search(r'"analysis"\s*:\s*"((?:[^"\\]|\\.)*)', raw or "")
        record = {"episode": self.episode_count, "revision": self.n_revisions, "agent": agent,
                  "process": self.process, "prompt": prompt, "raw": raw,
                  "analysis": found.group(1) if found else None, "tools": transcript,
                  "parsed": self._names(idxs), "allowed_tasks": self.allowed_tasks(),
                  "n_llm_calls": self.n_llm_calls, "n_fallbacks": self.n_fallbacks,
                  "n_invalid_items": self.n_invalid_items, "n_rejected_locked": self.n_rejected_locked,
                  "n_probe_episodes": self.n_probe_episodes}
        with self.lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def _expert_node(self, state, node: str):
        agent = self.NODE_MAPPING[node]
        use_tools = agent == "RL expert"
        prompt = self.build_agent_prompt(self.roles[agent], tools=use_tools)
        raw, obj, transcript = self._run_dialogue(prompt, allow_tools=use_tools)
        idxs = self._parse_bins(raw, obj)
        self._log(agent, prompt, raw, idxs, transcript)
        return {agent: self._names(idxs)}

    def _aggregator_node(self, state):
        if self.process == "voting":
            idxs = self._vote(state)
            self._log("Aggregator (vote)", "", "", idxs, [])
        else:
            proposals = {agent: state.get(agent) for agent in self.roles}
            prompt = self.build_agent_prompt(ROLE_AGGREGATOR, proposals=proposals, tools=False)
            raw, obj, transcript = self._run_dialogue(prompt, allow_tools=False)
            idxs = self._parse_bins(raw, obj)
            self._log("Aggregator", prompt, raw, idxs, transcript)
            if not idxs:
                idxs = self._vote(state)
        fallback = self._install_plan(idxs)
        if self.n_revisions <= 3 or self.n_revisions % 50 == 0:
            print(f"[llm] revision {self.n_revisions} (episode {self.episode_count}, process={self.process}, "
                  f"fallback={fallback}): {self._names(self.plan)}", flush=True)
        return {"Aggregator": self._names(self.plan)}

    def _build_graph(self):
        workflow = StateGraph(ResponseState)
        for node in self.NODE_MAPPING:
            workflow.add_node(node, lambda state, n=node: self._expert_node(state, n))
        workflow.add_node("AggregatorNode", self._aggregator_node)
        workflow.add_edge(START, "RLExpertNode")
        workflow.add_edge("RLExpertNode", "DomainExpertNode")
        workflow.add_edge("RLExpertNode", "RiskExpertNode")
        workflow.add_edge(list(self.NODE_MAPPING), "AggregatorNode")
        workflow.add_edge("AggregatorNode", END)
        return workflow.compile()

    def _revise(self) -> None:
        self._update_gate()
        self._close_memory()
        self.graph.invoke(dict(self.initial_state))

    def llm_stats(self):
        stats = super().llm_stats()
        stats["process"] = self.process
        return stats


NAME = "multi-agent LLM"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS + VALIDATION_SEEDS)
    parser.add_argument("--process", type=str, default="aggregator", choices=["aggregator", "voting"])
    parser.add_argument("--llm-url", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.8-27B-FP8")
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
    name = f"{NAME} ({args.process})" + "".join(f" ({label})" for label, _, on in flags if on)
    suffix = f"_{args.process}" + "".join(f"_{tag}" for _, tag, on in flags if on)

    results_path = os.path.join(args.out_dir, "results", f"results_llm_multi{suffix}_seed{seed}.json")
    stats_path = os.path.join(args.out_dir, "results", f"llm_stats_multi{suffix}_seed{seed}.json")
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
        created: List[SystemMultiAgent] = []

        def factory():
            curriculum = SystemMultiAgent(
                llm, process=args.process, revision_freq=args.revision_freq, nb_obj_prompt=args.plan_len,
                log_path=log_path, total_timesteps=args.total_timesteps, gate=not args.no_gate,
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