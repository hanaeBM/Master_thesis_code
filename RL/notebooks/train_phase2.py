import time
import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
import panda_gym 

from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple



_trapz = getattr(np, "trapezoid", None) or np.trapz

print("gymnasium:", gym.__version__)
from train_phase1 import *


# SINGLE LLM AGENT 

from openai import OpenAI
class LLMClient:
    def __init__(self, base_url: str, model: str = "Qwen/Qwen2.5-14B-Instruct",
                 api_key: str = "not-needed", temperature: float = 0.2, max_tokens: int = 200):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_json(self, prompt: str) -> str:
        response=self.client.chat.completion.create(model=self.model,message=[{'role':'user','content':prompt}],temperature=self.temperature,max_tokens=self.max_tokens)
        return response.choice[0].message.content


class SingleAgentLLMCurriculum(Curriculum):
    DIFFICULTY_LEVELS = ["near", "medium", "far"]
    N_BINS = N_TASKS * len(DIFFICULTY_LEVELS)
    WINDOW = 20                               
    DIFFICULTY_TO_FRACTION = [0.20, 0.55, 0.80]

    def __init__(self, llm_client: "LLMClient", revision_freq: int = 50, band: float = 0.15,
                 log_path: Optional[str] = None):
        self.llm_client=llm_client
        self.revision_freq=revision_freq
        self.band=band
        self.bins = [(name, level) for name in TASK_NAMES for level in self.DIFFICULTY_LEVELS]
        self.history: Dict[int, List[bool]] = {i: [] for i in range(self.N_BINS)}
        self.episode_count = 0
        self.current_bin = 0                    
        self.n_llm_calls = 0
        # malformed/invalid LLM output
        self.n_fallbacks = 0                     
        self.log_path = log_path
        self._last_bin: Optional[int] = None
        self._revise(force=True)  
        self.nb_obj_prompt=10

    def bin_stats(self) -> Dict[str, Dict]:
        """Per-bin rolling success rate """

        stats = {}
        for i, (name, level) in enumerate(self.bins):
            h = self.history[i][-self.WINDOW:]
            stats[f"{name}/{level}"] = {
                "n_episodes": len(self.history[i]),
                "success_rate": round(float(np.mean(h)), 3) if h else None,
            }
        return stats


    def build_prompt(self) -> str:
        stats = self._bin_stats()
        return (
            "You manage a training curriculum for a robot arm on 5 manipulation tasks "
            "(reach, push, slide, pickandplace, stack), each split into 3 difficulty "
            "levels (near, medium, far distance from that task's own default target "
            "region). reach/push/slide use a fixed, non-grasping gripper; pickandplace "
            "and stack require grasping and are typically only learnable once basic "
            "reaching/pushing competence already exists.\n\n"
            f"Rolling success rate per (task, level) bin, over at most the last "
            f"{self.WINDOW} episodes attempted in that bin (null = not attempted yet):\n"
            f"{json.dumps(stats, indent=2)}\n\n"
            f"Propose a LIST of {self.nb_obj_prompt} (task, distance_level) bins to spend the next batch of "
            "episodes on, to make the fastest overall progress across all 5 tasks. "
            "Respond with ONLY a JSON object, no other text:\n"
            '[{"task": "<one of reach|push|slide|pickandplace|stack>", '
            '"distance_level": "<one of near|medium|far>", "reason": "<one short sentence>"},...]'
        )

    def parse_llm_choice(self, raw_text: str):
        response=json.loads(raw_text)
        R=[]
        try:
            for dict in response:
                task,level=dict["task"],dict["distance_level"]
                if (task,level) in self.bins:
                    R.append(self.bins.index(((task,level))))
            return R
                             
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None


    def revise(self, force: bool = False) -> None:
        prompt=self.build_prompt
        raw=self.llm_client.generate_json(prompt)
        self.n_llm_calls+=1
        Idxs = self._parse_llm_choice(raw)
        # nb of invalid goal proposed by LLM
        self.n_fallbacks=self.nb_obj_prompt-len(Idxs)
        step= self.episode_count%self.nb_obj_prompt
        self.current_bin = Idxs[step]
        if self.log_path:
            self.append_log(prompt, raw, bin_idx)

    def append_log(self, prompt: str, raw: str, bin_idx: int) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps({
                "episode": self.episode_count,
                "prompt": prompt,
                "raw_response": raw,
                "chosen_bin": list(self.bins[bin_idx]),
                "n_llm_calls": self.n_llm_calls,
                "n_fallbacks": self.n_fallbacks,
            }) + "\n")


    def propose(self, rng: np.random.Generator) -> Tuple[int, np.ndarray]:
        name, level = self.bins[self.current_bin]
        spec = TASK_SPECS[name]
        center = (spec.goal_low + spec.goal_high) / 2.0
        frac_center = self.DIFFICULTY_TO_FRACTION[self.DIFFICULTY_LEVELS.index(level)]
        frac = rng.uniform(max(frac_center - self.band, 0.0), min(frac_center + self.band, 1.0))
        low = center - frac * (center - spec.goal_low)
        high = center + frac * (spec.goal_high - center)
        target = rng.uniform(low, high)
        self._last_bin = self.current_bin
        return spec.task_id, target.astype(np.float32)

    def notify_result(self, task_id: int, success: bool) -> None:
        self.history[self._last_bin].append(success)
        self.episode_count += 1
        self.revise()


CURRICULUMS["single-agent LLM"]=SingleAgentLLMCurriculum     
        
STRATEGIES = list(CURRICULUMS.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS,
                         help="Which seed to run (one job = one seed).")
    parser.add_argument("--out-dir", type=str, default=".",
                         help="Directory to write results_phase1_seed<SEED>.json into.")
    args = parser.parse_args()
    seed = args.seed

    results_path = os.path.join(args.out_dir, f"results/results_phase1_seed{seed}.json")

    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {}

    t0 = time.time()
        name="single-agent LLM"
        print(f"skip strategy={name!r} seed={seed} (already done)")
        print(f"success rate (mean over tasks): {_overall_success(results[name][-1]):.2f}")
            
        print(f"strategy={name}  seed={seed}  ({time.time() - t0:.0f}s elapsed)")
        history = train_one_run(curriculum_cls, seed, total_timesteps=TOTAL_TIMESTEPS, eval_freq=EVAL_FREQ,
                                 n_episodes_per_zone=N_EVAL_EPISODES_PER_ZONE, verbose=0)
        results[name] = history
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"success rate (mean over tasks): {_overall_success(history[-1]):.2f}")

    print(f"seed {seed} done in {time.time() - t0:.0f}s -> {results_path}")




if __name__ == "__main__":
    main()


                         
                    
            
            
    

            