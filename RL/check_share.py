import glob
import sys

import numpy as np

TASKS = ["reach", "push", "slide", "pickandplace", "stack"]


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "results/goals/*_seed*.npz"
    bin_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no file matches {pattern!r}")
        print("files in results/goals:", sorted(glob.glob("results/goals/*"))[:20])
        return
    for path in paths:
        d = np.load(path)
        step, task = d["step"], d["task"].astype(int)
        last = int(step.max())
        print(f"\n{path}  ({len(step)} episodes, {last} steps)")
        print("window".ljust(16) + "".join(t.rjust(14) for t in TASKS))
        for a in range(0, last + 1, bin_steps):
            m = (step >= a) & (step < a + bin_steps)
            if m.sum() == 0:
                continue
            share = np.bincount(task[m], minlength=len(TASKS)) / m.sum()
            print(f"{a // 1000}k-{(a + bin_steps) // 1000}k".ljust(16) + "".join(f"{s:14.2f}" for s in share))
        if "dist_frac" in d.files:
            print("mean goal distance by task:",
                  {t: round(float(d["dist_frac"][task == i].mean()), 2) if (task == i).any() else None
                   for i, t in enumerate(TASKS)})


if __name__ == "__main__":
    main()
