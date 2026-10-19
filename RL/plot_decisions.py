
import json
import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore", message="Mean of empty slice")
_KFMT = FuncFormatter(lambda v, _: f"{int(v / 1000)}k" if v < 1_000_000 else f"{v / 1e6:g}M")

TASKS = ["reach", "push", "slide", "pickandplace", "stack"]
TASK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def load_runs(goal_dir: str, strategies: List[str], seeds: List[int]) -> Dict[str, List[Dict]]:
    """{strategy: [one dict of arrays per seed that exists]}; missing files are skipped."""
    runs: Dict[str, List[Dict]] = {}
    for name in strategies:
        for seed in seeds:
            path = os.path.join(goal_dir, f"{_slug(name)}_seed{seed}.npz")
            if os.path.exists(path):
                d = np.load(path)
                runs.setdefault(name, []).append({k: d[k] for k in d.files})
    return runs


def _edges(runs, bin_steps):
    last = max(int(r["step"].max()) for rs in runs.values() for r in rs)
    return np.arange(0, last + bin_steps, bin_steps)


def task_share_per_window(run: Dict, edges: np.ndarray) -> np.ndarray:
    """(n_windows, 5): share of the EPISODES started in each window that were on each task."""
    idx = np.clip(np.searchsorted(edges, run["step"], side="right") - 1, 0, len(edges) - 2)
    counts = np.zeros((len(edges) - 1, len(TASKS)))
    np.add.at(counts, (idx, run["task"].astype(int)), 1)
    total = counts.sum(1, keepdims=True)
    return np.where(total > 0, counts / np.maximum(total, 1), np.nan)


def difficulty_per_window(run: Dict, edges: np.ndarray) -> np.ndarray:
    """(n_windows, 5): mean dist_frac of the goals chosen (NaN if the task was not sampled)."""
    idx = np.clip(np.searchsorted(edges, run["step"], side="right") - 1, 0, len(edges) - 2)
    s = np.zeros((len(edges) - 1, len(TASKS)))
    n = np.zeros_like(s)
    np.add.at(s, (idx, run["task"].astype(int)), run["dist_frac"])
    np.add.at(n, (idx, run["task"].astype(int)), 1)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.set_major_formatter(_KFMT)


def plot_task_share(runs: Dict[str, List[Dict]], bin_steps: int = 20_000, save_to: Optional[str] = None):
    """One panel per curriculum: stacked share of episodes per task over training (mean over seeds)."""
    names = list(runs)
    edges = _edges(runs, bin_steps)
    mid = (edges[:-1] + edges[1:]) / 2
    ncols = min(len(names), 4)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, name in zip(axes.ravel(), names):
        share = np.nanmean(np.stack([task_share_per_window(r, edges) for r in runs[name]]), axis=0)
        share = np.nan_to_num(share)
        ax.stackplot(mid, share.T, colors=TASK_COLORS, edgecolor=SURFACE, linewidth=1.0, labels=TASKS)
        ax.set_title(f"{name}  (n={len(runs[name])} seeds)", fontsize=9, color=INK)
        ax.set_ylim(0, 1)
        _style(ax)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Training steps", fontsize=8, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of episodes", fontsize=8, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(TASKS), frameon=False, fontsize=9)
    fig.suptitle("Which task each curriculum chooses, over training", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    if save_to:
        fig.savefig(save_to, dpi=150, facecolor=SURFACE)
    return fig


def plot_difficulty(runs: Dict[str, List[Dict]], bin_steps: int = 20_000, save_to: Optional[str] = None):
    """One heatmap per curriculum: rows = tasks, columns = time windows, colour = mean distance of the
    chosen goals from the task centre (fraction of the half-diagonal). Blank = task not sampled."""
    names = list(runs)
    edges = _edges(runs, bin_steps)
    ncols = min(len(names), 4)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.6 * nrows + 0.6), sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    im = None
    for ax, name in zip(axes.ravel(), names):
        d = np.nanmean(np.stack([difficulty_per_window(r, edges) for r in runs[name]]), axis=0)
        im = ax.imshow(d.T, aspect="auto", origin="upper", cmap=SEQ_CMAP, vmin=0, vmax=1,
                       extent=(edges[0], edges[-1], len(TASKS), 0))
        ax.set_yticks(np.arange(len(TASKS)) + 0.5)
        ax.set_yticklabels(TASKS, fontsize=8, color=MUTED)
        ax.set_title(name, fontsize=9, color=INK)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.set_major_formatter(_KFMT)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Training steps", fontsize=8, color=MUTED)
    fig.suptitle("How far from the centre the chosen goals are (0 = centre, 1 = corner)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 0.93, 0.95))
    cax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=8, colors=MUTED)
    if save_to:
        fig.savefig(save_to, dpi=150, facecolor=SURFACE)
    return fig


def first_event(goal_dir: str, strategy: str, seed: int, key: str) -> Optional[int]:
    """First training step at which a boolean entry of the curriculum snapshot is True
    (e.g. 'family_done' or 'stack_unlocked' for Hand-crafted 2). Resolution = --snapshot-every."""
    path = os.path.join(goal_dir, f"{_slug(strategy)}_seed{seed}_snapshots.json")
    for snap in json.load(open(path)):
        if snap.get(key):
            return int(snap["step"])
    return None


def main():
    STRATEGIES  = [
    "random",
    "hand crafted 1(naive)",
    "hand crafted 2 (domain-informed)",
    "single-agent LLM",
    #"InformedAdaptiveCurriculum"
    ]
    
    SEEDS = [10, 11, 12, 13, 14]

    runs=load_runs("results/goals", STRATEGIES, SEEDS)
    plot_task_share(runs, bin_steps=20_000, save_to= "graphs/plot_task_share")
    plot_difficulty(runs, bin_steps=20_000, save_to = "graphs/plot_difficulty")


if __name__=="__main__":
    main()
