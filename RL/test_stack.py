import argparse
import json
import os
import time

import numpy as np

from train_phase1 import (
    Curriculum,
    TASK_SPECS,
    TOTAL_TIMESTEPS,
    EVAL_FREQ,
    N_EVAL_EPISODES_PER_ZONE,
    SNAPSHOT_EVERY,
    train_one_run,
)

NAME = "stack only"


class StackOnlyCurriculum(Curriculum):
    goal_scale = 1.0

    def propose(self, rng):
        spec = TASK_SPECS["stack"]
        target = rng.uniform(spec.goal_low, spec.goal_high) * self.goal_scale
        return spec.task_id, target.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=str, default="tests_stack")
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    parser.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY)
    parser.add_argument("--goal-scale", type=float, default=1.0)
    args = parser.parse_args()
    seed = args.seed

    StackOnlyCurriculum.goal_scale = args.goal_scale

    suffix = "" if args.goal_scale == 1.0 else f"_scale{args.goal_scale}"
    results_path = os.path.join(args.out_dir, "results", f"results_stack_only{suffix}_seed{seed}.json")
    goal_prefix = os.path.join(args.out_dir, "results", "goals", f"stack_only{suffix}_seed{seed}")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    os.makedirs(os.path.dirname(goal_prefix), exist_ok=True)

    if os.path.exists(results_path):
        print(f"already done: {results_path}")
        return

    t0 = time.time()
    print(f"strategy={NAME}  seed={seed}  steps={args.total_timesteps}  goal_scale={args.goal_scale}")
    history = train_one_run(
        StackOnlyCurriculum,
        seed,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        n_episodes_per_zone=N_EVAL_EPISODES_PER_ZONE,
        verbose=0,
        goal_log_path=goal_prefix,
        snapshot_every=args.snapshot_every,
    )

    with open(results_path, "w") as f:
        json.dump({NAME: history}, f, indent=2)

    stack_curve = [(h["timesteps"], h["success_stack"]) for h in history]
    best = max(v for _, v in stack_curve)
    print(f"stack success: last={stack_curve[-1][1]:.2f}  best={best:.2f}")
    print(f"done in {time.time() - t0:.0f}s -> {results_path}")


if __name__ == "__main__":
    main()