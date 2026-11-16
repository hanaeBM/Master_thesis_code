"""
Run this when all 5 seed jobs (job_phase1.sh, array 10-14) have finished.

Usage:
    python3 merge_results.py
    python3 merge_results.py --dir /path/to/results --out-dir /path/to/plots
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASK_NAMES = ["stack"]

COLORS = {
    "random": "#2a78d6",
    
}
LABELS = {
    "random": "Random goals",
   
}
STRATEGIES = list(COLORS.keys())

SEED_FILE_RE = re.compile(r"results.+_seed(\d+)\.json$")


def merge_seed_files(results_dir: str) -> dict:
    """{strategy: {seed_str: history}} merged from every seed file found."""
    results: dict = {s: {} for s in STRATEGIES}
    pattern = os.path.join(results_dir, "results_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No results_phase1_seed*.json files found in {results_dir!r}. "
            "Make sure the array job (job_phase1.sh) has produced output there."
        )

    for fp in files:
        m = SEED_FILE_RE.search(os.path.basename(fp))
        if not m:
            continue
        seed = m.group(1)
        with open(fp) as f:
            per_seed = json.load(f)  # {strategy: history}
        for strategy, history in per_seed.items():
            results.setdefault(strategy, {})[seed] = history
        print(f"merged {fp}  (seed={seed}, strategies={list(per_seed.keys())})")

    return results


def build_dataframe(results: dict) -> pd.DataFrame:
    rows = []
    for strategy, seeds in results.items():
        for seed_key, history in seeds.items():
            for entry in history:
                for task in TASK_NAMES:
                    rows.append({
                        "strategy": strategy,
                        "seed": int(seed_key),
                        "timesteps": entry["timesteps"],
                        "task": task,
                        "success": entry[f"success_{task}"],
                    })
    return pd.DataFrame(rows)


def plot_global(df: pd.DataFrame, out_path: str) -> None:
    df_global = (
        df.groupby(["strategy", "seed", "timesteps"], as_index=False)["success"].mean()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy in STRATEGIES:
        sub = df_global[df_global["strategy"] == strategy]
        if sub.empty:
            continue
        agg = sub.groupby("timesteps").agg(mean=("success", "mean"), std=("success", "std")).reset_index()
        color, label = COLORS[strategy], LABELS[strategy]
        ax.plot(agg["timesteps"], agg["mean"], label=label, color=color, linewidth=2)
        if agg["std"].notna().any() and sub["seed"].nunique() > 1:
            ax.fill_between(agg["timesteps"],
                             (agg["mean"] - agg["std"]).clip(lower=0),
                             (agg["mean"] + agg["std"]).clip(upper=1),
                             color=color, alpha=0.15)

    ax.set_xlabel("Training steps")
    ax.set_ylabel("Success rate (mean over the 5 tasks)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Overall held-out success rate vs. training time, by curriculum")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_per_task(df: pd.DataFrame, out_path: str, task: str = "stack") -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sub_task = df[df["task"] == task]
 
    for strategy in STRATEGIES:
        sub = sub_task[sub_task["strategy"] == strategy]
        if sub.empty:
            continue
        agg = (sub.groupby("timesteps")
                  .agg(mean=("success", "mean"), std=("success", "std"))
                  .reset_index())
        color, label = COLORS[strategy], LABELS[strategy]
        ax.plot(agg["timesteps"], agg["mean"], label=label, color=color, linewidth=2)
        if sub["seed"].nunique() > 1:
            ax.fill_between(agg["timesteps"],
                            (agg["mean"] - agg["std"]).clip(lower=0),
                            (agg["mean"] + agg["std"]).clip(upper=1),
                            color=color, alpha=0.15)
 
    ax.set_title(f"Held-out success rate vs. training time: {task}")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Held-out success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
 
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="results", help="Where the results_phase1_seed*.json files live.")
    parser.add_argument("--out-dir", type=str, default="graphs", help="Where to write the combined json + plots.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = merge_seed_files(args.dir)
    combined_path = os.path.join(args.out_dir, "results_phase1_combined.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"combined results written to {combined_path}")

    df = build_dataframe(results)
    print(f"dataframe shape: {df.shape}")

    
    plot_per_task(df, os.path.join(args.out_dir, "random stack.png"))


if __name__ == "__main__":
    main()
