"""
Run this when all 5 seed jobs (job_phase1.sh, array 10-14) have finished.

Usage:
    python3 merge_results.py
    python3 merge_results.py --dir /path/to/results --out-dir /path/to/plots
    python3 merge_results.py --baselines B1,B3 --tasks reach,push
    python3 merge_results.py --tasks stack
"""
import argparse
import glob
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASK_NAMES = ["reach", "push", "slide", "pickandplace", "stack"]

COLORS = {
    "random": "#2a78d6",
    "hand crafted 1(naive)": "#eb6834",
    "hand crafted 2 (domain-informed)": "#1baf7a",
    "single-agent LLM (+hint) (+floor)": "#eda100",
    #"InformedAdaptiveCurriculum":"#1a1a1a"
}
LABELS = {
    "random": "Random goals",
    "hand crafted 1(naive)": "Hand-crafted 1 (naive)",
    "hand crafted 2 (domain-informed)": "Hand-crafted 2 (domain-informed)",
    "single-agent LLM (+hint) (+floor)": "Single LLM",
    #"InformedAdaptiveCurriculum":"Informed Adaptive Curriculum"
}
STRATEGIES = list(COLORS.keys())

BASELINE_CODES = {
    "B1": "random",
    "B2": "hand crafted 1(naive)",
    "B3": "hand crafted 2 (domain-informed)",
    "B4": "single-agent LLM (+hint) (+floor)",
}

SEED_FILE_RE = re.compile(r"results.+_seed(\d+)\.json$")


def choose_baselines(baselines: str) -> list:
    """'B1,B3' -> ['random', 'hand crafted 2 (domain-informed)']"""
    if baselines == "all":
        return list(STRATEGIES)
    codes = [b.strip() for b in baselines.split(",") if b.strip()]
    unknown = [c for c in codes if c not in BASELINE_CODES]
    if unknown:
        raise ValueError(f"Unknown baseline(s) {unknown}. Choose from {list(BASELINE_CODES)}")
    return [BASELINE_CODES[c] for c in codes]


def choose_tasks(tasks: str) -> list:
    """'reach,push' -> ['reach', 'push']"""
    if tasks == "all":
        return list(TASK_NAMES)
    chosen = [t.strip() for t in tasks.split(",") if t.strip()]
    unknown = [t for t in chosen if t not in TASK_NAMES]
    if unknown:
        raise ValueError(f"Unknown task(s) {unknown}. Choose from {TASK_NAMES}")
    return chosen


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


def build_dataframe(results: dict, tasks: list) -> pd.DataFrame:
    rows = []
    for strategy, seeds in results.items():
        for seed_key, history in seeds.items():
            for entry in history:
                for task in tasks:
                    rows.append({
                        "strategy": strategy,
                        "seed": int(seed_key),
                        "timesteps": entry["timesteps"],
                        "task": task,
                        "success": entry[f"success_{task}"],
                    })
    return pd.DataFrame(rows)


def _plot_strategy(ax, sub: pd.DataFrame, strategy: str) -> None:
    """One curve (mean over seeds) + shaded band (std over seeds)."""
    agg = sub.groupby("timesteps").agg(mean=("success", "mean"), std=("success", "std")).reset_index()
    color, label = COLORS[strategy], LABELS[strategy]
    ax.plot(agg["timesteps"], agg["mean"], label=label, color=color, linewidth=2)
    if agg["std"].notna().any() and sub["seed"].nunique() > 1:
        ax.fill_between(agg["timesteps"],
                        (agg["mean"] - agg["std"]).clip(lower=0),
                        (agg["mean"] + agg["std"]).clip(upper=1),
                        color=color, alpha=0.15)


def plot_global(df: pd.DataFrame, out_path: str, strategies: list, tasks: list) -> None:
    df_global = (
        df.groupby(["strategy", "seed", "timesteps"], as_index=False)["success"].mean()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy in strategies:
        sub = df_global[df_global["strategy"] == strategy]
        if sub.empty:
            continue
        _plot_strategy(ax, sub, strategy)

    ax.set_xlabel("Training steps")
    if len(tasks) == 1:
        ax.set_ylabel(f"Success rate ({tasks[0]})")
    else:
        ax.set_ylabel(f"Success rate (mean over the {len(tasks)} tasks)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Overall held-out success rate vs. training time, by curriculum")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_per_task(df: pd.DataFrame, out_path: str, strategies: list, tasks: list) -> None:
    n = len(tasks)
    ncols = min(3, n)                 
    nrows = math.ceil(n / ncols)

    
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(5 * ncols, 9), 4 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()

    for i, task in enumerate(tasks):
        ax = axes[i]
        sub_task = df[df["task"] == task]
        for strategy in strategies:
            sub = sub_task[sub_task["strategy"] == strategy]
            if sub.empty:
                continue
            _plot_strategy(ax, sub, strategy)
        ax.set_title(task)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

        if i % ncols == 0:
            ax.set_ylabel("Success rate")
        if i + ncols >= n:
            ax.set_xlabel("Training steps")
            ax.tick_params(labelbottom=True)

    for ax in axes[n:]:
        ax.axis("off")

    handles, labels_ = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_, loc="lower center", ncol=min(4, len(handles)),
                   fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Held-out success rate per task vs. training time, by curriculum", fontsize=13)
    bottom = 0.12 if nrows == 1 else 0.06  
    fig.tight_layout(rect=[0, bottom, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

def plot_per_seed(df: pd.DataFrame, out_path: str, strategies: list, tasks: list):
    df_global = (
        df.groupby(["strategy", "seed", "timesteps"], as_index=False)["success"].mean()
    )
    seeds = sorted(df_global["seed"].unique())   
    n = len(seeds)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
 
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(5 * ncols, 9), 4 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
 
    for i, seed in enumerate(seeds):
        ax = axes[i]
        sub_seed = df_global[df_global["seed"] == seed]
        for strategy in strategies:
            sub = sub_seed[sub_seed["strategy"] == strategy]
            if sub.empty:
                continue
            _plot_strategy(ax, sub, strategy)   
        ax.set_title(f"seed {seed}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
 
        if i % ncols == 0:
            ax.set_ylabel("Success rate")
        if i + ncols >= n:
            ax.set_xlabel("Training steps")
            ax.tick_params(labelbottom=True)
 
    for ax in axes[n:]:
        ax.axis("off")
 
    handles, labels_ = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_, loc="lower center", ncol=min(4, len(handles)),
                   fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Held-out success rate per seeds vs. training time, by curriculum", fontsize=13)
    bottom = 0.12 if nrows == 1 else 0.06  
    fig.tight_layout(rect=[0, bottom, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", type=str, default="all",
                        help="Comma-separated baselines among B1,B2,B3,B4 (e.g. B1,B3), or 'all'.")
    parser.add_argument("--tasks", type=str, default="all",
                        help=f"Comma-separated tasks among {','.join(TASK_NAMES)} (e.g. reach,push), or 'all'.")
    parser.add_argument("--dir", type=str, default="results", help="Where the results_phase1_seed*.json files live.")
    parser.add_argument("--out-dir", type=str, default="graphs", help="Where to write the combined json + plots.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    strategies = choose_baselines(args.baselines)
    tasks = choose_tasks(args.tasks)
    print(f"strategies: {strategies}")
    print(f"tasks: {tasks}")

    results = merge_seed_files(args.dir)
    combined_path = os.path.join(args.out_dir, "results_phase1_combined.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"combined results written to {combined_path}")

    df = build_dataframe(results, tasks)
    print(f"dataframe shape: {df.shape}")

    plot_global(df, os.path.join(args.out_dir, "phase1_global.png"), strategies, tasks)
    plot_per_task(df, os.path.join(args.out_dir, "phase1_per_task.png"), strategies, tasks)
    plot_per_seed(df, os.path.join(args.out_dir, "phase1_per_seeds.png"), strategies, tasks)


if __name__ == "__main__":
    main()