import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import panda_gym  # noqa: F401 (registers the Panda-* envs)

from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

_trapz = getattr(np, "trapezoid", None) or np.trapz


TASK_NAMES = ["reach", "push", "slide", "pickandplace", "stack"]
TASK_ENV_IDS = {
    "reach": "PandaReach-v3",
    "push": "PandaPush-v3",
    "slide": "PandaSlide-v3",
    "pickandplace": "PandaPickAndPlace-v3",
    "stack": "PandaStack-v3",
}

MAX_ROBOT_OBS_DIM = 7
MAX_GOAL_DIM = 6
N_TASKS = len(TASK_NAMES)

SEEDS = [10, 11, 12, 13, 14]


@dataclass
class TaskSpec:
    name: str
    task_id: int
    env_id: str
    block_gripper: bool
    goal_dim: int
    threshold: float
    goal_low: np.ndarray
    goal_high: np.ndarray
    stack_z: Optional[Tuple[float, float]] = None


def build_task_specs() -> Dict[str, TaskSpec]:
    specs = {}

    specs["reach"] = TaskSpec(
        "reach", 0, TASK_ENV_IDS["reach"], block_gripper=True, goal_dim=3, threshold=0.05,
        goal_low=np.array([-0.15, -0.15, 0.0]), goal_high=np.array([0.15, 0.15, 0.3]),
    )

    base = np.array([0.0, 0.0, 0.02])
    specs["push"] = TaskSpec(
        "push", 1, TASK_ENV_IDS["push"], block_gripper=True, goal_dim=3, threshold=0.05,
        goal_low=base + np.array([-0.15, -0.15, 0.0]), goal_high=base + np.array([0.15, 0.15, 0.0]),
    )

    base = np.array([0.0, 0.0, 0.03])
    specs["slide"] = TaskSpec(
        "slide", 2, TASK_ENV_IDS["slide"], block_gripper=True, goal_dim=3, threshold=0.05,
        goal_low=base + np.array([-0.15 + 0.4, -0.15, 0.0]), goal_high=base + np.array([0.15 + 0.4, 0.15, 0.0]),
    )

    base = np.array([0.0, 0.0, 0.02])
    specs["pickandplace"] = TaskSpec(
        "pickandplace", 3, TASK_ENV_IDS["pickandplace"], block_gripper=False, goal_dim=3, threshold=0.05,
        goal_low=base + np.array([-0.15, -0.15, 0.0]), goal_high=base + np.array([0.15, 0.15, 0.2]),
    )

    specs["stack"] = TaskSpec(
        "stack", 4, TASK_ENV_IDS["stack"], block_gripper=False, goal_dim=2, threshold=0.1,
        goal_low=np.array([-0.15, -0.15]), goal_high=np.array([0.15, 0.15]),
        stack_z=(0.02, 0.06),
    )
    return specs


TASK_SPECS = build_task_specs()


def expand_stack_goal(xy_shift: np.ndarray, spec: TaskSpec) -> np.ndarray:
    z1, z2 = spec.stack_z
    return np.array([xy_shift[0], xy_shift[1], z1, xy_shift[0], xy_shift[1], z2], dtype=np.float32)


def native_goal_dim(name: str) -> int:
    return 6 if name == "stack" else TASK_SPECS[name].goal_dim


# 2. Held-out evaluation

HELD_OUT_SEED = 20260921
N_ZONES_PER_TASK = 5
ZONE_RADIUS_FRACTION = 0.08
ZONE_DIFFICULTY_LEVELS = ["near", "near", "medium", "medium", "far"]
DIFFICULTY_TO_FRACTION = {"near": 0.20, "medium": 0.55, "far": 0.85}


def _sample_direction(rng: np.random.RandomState, dim: int) -> np.ndarray:
    v = rng.normal(size=dim)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def build_held_out_zones() -> Dict[str, List[Dict]]:
    rng = np.random.RandomState(HELD_OUT_SEED)
    zones: Dict[str, List[Dict]] = {}
    for name in TASK_NAMES:
        spec = TASK_SPECS[name]
        low, high = spec.goal_low, spec.goal_high
        free = (high - low) > 1e-9
        center = (low + high) / 2.0
        half_extent = (high - low) / 2.0
        half_diag = np.linalg.norm(half_extent[free]) if free.any() else 0.0
        radius = ZONE_RADIUS_FRACTION * half_diag if half_diag > 0 else 1e-3
        radius_eval = radius / 2
        task_zones = []
        for level in ZONE_DIFFICULTY_LEVELS:
            frac = DIFFICULTY_TO_FRACTION[level]
            direction = np.zeros(spec.goal_dim)
            direction[free] = _sample_direction(rng, int(free.sum()))
            point = center + direction * frac * half_diag
            margin = np.where(free, radius * 0.5, 0.0)
            point = np.clip(point, low + margin, high - margin)
            task_zones.append({"center": point.astype(np.float32), "radius": float(radius),
                                "radius_eval": float(radius_eval), "level": level})
        zones[name] = task_zones
    return zones


HELD_OUT_ZONES = build_held_out_zones()


def in_held_out_zone(name: str, point: np.ndarray) -> bool:
    for zone in HELD_OUT_ZONES[name]:
        if np.linalg.norm(point - zone["center"]) < zone["radius"]:
            return True
    return False


def enforce_held_out_exclusion(name: str, point: np.ndarray, rng: np.random.Generator,
                                max_tries: int = 50) -> np.ndarray:
    spec = TASK_SPECS[name]
    low, high = spec.goal_low, spec.goal_high
    p = point.copy()
    jitter_scale = 2.0 * max(z["radius"] for z in HELD_OUT_ZONES[name])
    for _ in range(max_tries):
        if not in_held_out_zone(name, p):
            return p
        p = point + rng.normal(scale=jitter_scale, size=point.shape)
        p = np.clip(p, low, high)
    for _ in range(max_tries):
        p = rng.uniform(low, high)
        if not in_held_out_zone(name, p):
            return p
    raise RuntimeError(f"Could not find a valid point outside held-out zones for task {name}")


# 3. Shared obs/reward helpers

SHARED_ACTION_DIM = 4


def pad_goal(g: np.ndarray, name: str) -> np.ndarray:
    g = np.asarray(g, dtype=np.float32).reshape(-1)
    if g.shape[0] == MAX_GOAL_DIM:
        return g
    new_g = np.zeros(MAX_GOAL_DIM, dtype=np.float32)
    new_g[: g.shape[0]] = g
    return new_g


def pad_robot_obs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] == MAX_ROBOT_OBS_DIM:
        return x
    new_x = np.zeros(MAX_ROBOT_OBS_DIM, dtype=np.float32)
    new_x[: x.shape[0]] = x
    return new_x


def one_hot_task(task_id: int) -> np.ndarray:
    tab = np.zeros(N_TASKS, dtype=np.float32)
    tab[task_id] = 1.0
    return tab


def shared_compute_reward(achieved_goal, desired_goal, info) -> np.ndarray:
    achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
    desired_goal = np.asarray(desired_goal, dtype=np.float32)
    single = desired_goal.ndim == 1
    if single:
        achieved_goal = achieved_goal[None, :]
        desired_goal = desired_goal[None, :]
        info = [info]
    n = desired_goal.shape[0]
    rewards = np.zeros(n, dtype=np.float32)
    for i in range(n):
        info_i = info[i]
        task_id = int(info_i["task_id"]) if isinstance(info_i, dict) else int(info_i)
        name = TASK_NAMES[task_id]
        gd = native_goal_dim(name)
        thr = TASK_SPECS[name].threshold
        d = np.linalg.norm(achieved_goal[i, :gd] - desired_goal[i, :gd])
        rewards[i] = -float(d > thr)
    return rewards[0] if single else rewards


def shared_is_success(achieved_goal_padded, desired_goal_padded, task_id: int) -> bool:
    name = TASK_NAMES[task_id]
    gd = native_goal_dim(name)
    thr = TASK_SPECS[name].threshold
    d = np.linalg.norm(np.asarray(achieved_goal_padded)[:gd] - np.asarray(desired_goal_padded)[:gd])
    return bool(d < thr)


def _jsonable(x):
    """numpy -> plain python so curriculum snapshots can go into a JSON file."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# 4. Non-LLM curriculum baselines
class Curriculum:
    def propose(self, rng: np.random.Generator) -> Tuple[int, np.ndarray]:
        raise NotImplementedError

    def notify_result(self, task_id: int, success: bool) -> None:
        pass

    def snapshot(self) -> Dict:
        """[GOAL LOG] Internal state worth recording every `snapshot_every` steps (stage, unlock flags,
        bandit counts...). Default: nothing."""
        return {}


class RandomCurriculum(Curriculum):
    def propose(self, rng: np.random.Generator) -> Tuple[int, np.ndarray]:
        selected_task = int(rng.integers(0, N_TASKS))
        name = TASK_NAMES[selected_task]
        spec = TASK_SPECS[name]
        target = rng.uniform(spec.goal_low, spec.goal_high)
        return spec.task_id, target.astype(np.float32)


class HandCraftedCurriculum(Curriculum):
    ADVANCE_THRESHOLD = 0.70
    WINDOW = 20

    def __init__(self):
        self.order = ["reach", "push", "slide", "pickandplace", "stack"]
        self.stage = 0
        self.history: List[bool] = []

    def _rolling_success(self) -> float:
        if not self.history:
            return 0.0
        recent = self.history[-self.WINDOW:]
        return float(np.mean(recent))

    def propose(self, rng: np.random.Generator) -> Tuple[int, np.ndarray]:
        task = self.order[self.stage]
        spec = TASK_SPECS[task]
        low, high = spec.goal_low, spec.goal_high
        center = (low + high) / 2.0
        progress = np.clip(self._rolling_success() / self.ADVANCE_THRESHOLD, 0.0, 1.0)
        frac = max(1.0 - progress, 0.15)
        goal_low = center - frac * (center - low)
        goal_high = center + frac * (high - center)
        target = rng.uniform(goal_low, goal_high)
        return spec.task_id, target.astype(np.float32)

    def notify_result(self, task_id: int, success: bool) -> None:
        name = TASK_NAMES[task_id]
        if name != self.order[self.stage]:
            return
        self.history.append(success)
        if self._rolling_success() >= self.ADVANCE_THRESHOLD and len(self.history) >= self.WINDOW:
            if self.stage < len(self.order) - 1:
                self.stage += 1
                self.history = []

    def snapshot(self) -> Dict:
        return {"stage": self.order[self.stage], "stage_index": self.stage,
                "rolling_success": self._rolling_success()}


class HandCraftedCurriculum2(Curriculum):
    ADVANCE_THRESHOLD = 0.70
    THRESHOLD_START_STACK = 0.50
    WINDOW = 20
    NON_GRASPING = ["reach", "push", "slide"]
    STACK_RAMP_EPISODES = 100
    STACK_MAX_SHARE = 0.70

    def __init__(self):
        self.order = ["reach", "push", "slide", "pickandplace", "stack"]
        self.history: Dict[str, List[bool]] = {name: [] for name in TASK_NAMES}
        self.family_done = False
        self.stack_unlocked = False
        self.stack_episode_count = 0

    def _rolling_success(self, name: str) -> float:
        h = self.history[name]
        if not h:
            return 0.0
        recent = h[-self.WINDOW:]
        return float(np.mean(recent))

    def _is_mastered(self, name: str, threshold: float) -> bool:
        return self._rolling_success(name) >= threshold and len(self.history[name]) >= self.WINDOW

    def propose(self, rng: np.random.Generator) -> Tuple[int, np.ndarray]:
        if not self.family_done:
            name = self.NON_GRASPING[int(rng.integers(0, len(self.NON_GRASPING)))]
        elif not self.stack_unlocked:
            name = "pickandplace"
        else:
            stack_share = min(self.stack_episode_count / self.STACK_RAMP_EPISODES, self.STACK_MAX_SHARE)
            name = "stack" if rng.random() < stack_share else "pickandplace"

        spec = TASK_SPECS[name]
        low, high = spec.goal_low, spec.goal_high
        center = (low + high) / 2.0
        progress = np.clip(self._rolling_success(name) / self.ADVANCE_THRESHOLD, 0.0, 1.0)
        frac = max(1.0 - progress, 0.15)
        goal_low = center - frac * (center - low)
        goal_high = center + frac * (high - center)
        target = rng.uniform(goal_low, goal_high)
        return spec.task_id, target.astype(np.float32)

    def notify_result(self, task_id: int, success: bool) -> None:
        name = TASK_NAMES[task_id]
        self.history[name].append(success)
        if not self.family_done and all(self._is_mastered(n, self.ADVANCE_THRESHOLD) for n in self.NON_GRASPING):
            self.family_done = True
        if self.family_done and not self.stack_unlocked and self._is_mastered("pickandplace", self.THRESHOLD_START_STACK):
            self.stack_unlocked = True
        if self.stack_unlocked:
            self.stack_episode_count += 1

    def snapshot(self) -> Dict:
        return {"family_done": self.family_done, "stack_unlocked": self.stack_unlocked,
                "stack_episode_count": self.stack_episode_count,
                "rolling_success": {n: self._rolling_success(n) for n in TASK_NAMES}}


class UCBAdaptiveCurriculum(Curriculum):
    DIFFICULTY_LEVELS = ["near", "medium", "far"]
    UCB_C = 0.1
    WINDOW = 40
    LP_Z = 1.0
    N_BINS = N_TASKS * len(DIFFICULTY_LEVELS)
    DIFFICULTY_TO_FRACTION = [0.20, 0.55, 0.80]
    BAND = 0.15

    def __init__(self):
        self.bins = [(name, level) for name in TASK_NAMES for level in self.DIFFICULTY_LEVELS]
        self.n_pulls = np.zeros(self.N_BINS, dtype=np.int64)
        self.history = {i: [] for i in range(self.N_BINS)}
        self.lp = np.zeros(self.N_BINS)
        self.total_pulls = 0
        self._last_bin = None

    def _refresh_learning_progress(self, bin_idx: int) -> None:
        h = self.history[bin_idx]
        half = self.WINDOW // 2
        if len(h) < self.WINDOW :
            return
        old = np.mean(h[-self.WINDOW:-half])
        new = np.mean(h[-half:])
        pooled = 0.5 * (old + new)
        se = np.sqrt(max(pooled * (1.0 - pooled), 1e-9) * 2.0 / half)
        self.lp[bin_idx] = max(0.0, abs(new - old) - self.LP_Z * se)

    def _learning_progress(self, bin_idx: int) -> float:
        return float(self.lp[bin_idx])

    def _ucb_score(self, bin_idx: int) -> float:
        if self.n_pulls[bin_idx] == 0:
            return float("inf")
        return self._learning_progress(bin_idx) + self.UCB_C * np.sqrt(np.log(self.total_pulls) / self.n_pulls[bin_idx])

    def propose(self, rng: np.random.Generator) -> Tuple[int, np.ndarray]:
        bin_idx = int(np.argmax([self._ucb_score(i) for i in range(self.N_BINS)]))
        name, level = self.bins[bin_idx]
        spec = TASK_SPECS[name]
        low, high = spec.goal_low, spec.goal_high
        free = (high - low) > 1e-9
        center = (low + high) / 2.0
        half_diag = np.linalg.norm(((high - low) / 2.0)[free])

        frac_center = self.DIFFICULTY_TO_FRACTION[self.DIFFICULTY_LEVELS.index(level)]
        frac = rng.uniform(max(frac_center - self.BAND, 0.0), min(frac_center + self.BAND, 1.0))
        g = rng.normal(size=int(free.sum()))
        direction = np.zeros(len(low))
        direction[free] = g / max(np.linalg.norm(g), 1e-9)
        target = np.clip(center + direction * frac * half_diag, low, high)

        self._last_bin = bin_idx
        return spec.task_id, target.astype(np.float32)

    def notify_result(self, task_id: int, success: bool) -> None:
        self.n_pulls[self._last_bin] += 1
        self.total_pulls += 1
        self.history[self._last_bin].append(bool(success))
        self._refresh_learning_progress(self._last_bin)

    def snapshot(self) -> Dict:
        return {"n_pulls": {f"{n}/{l}": int(self.n_pulls[i]) for i, (n, l) in enumerate(self.bins)},
                "learning_progress": {f"{n}/{l}": float(self.lp[i]) for i, (n, l) in enumerate(self.bins)}}


class InformedAdaptiveCurriculum(Curriculum):
   
 
    ADVANCE_THRESHOLD = 0.70
    THRESHOLD_START_STACK = 0.50
    WINDOW = 20
    NON_GRASPING = ["reach", "push", "slide"]
    STACK_RAMP_EPISODES = 100
    STACK_MAX_SHARE = 0.70
 
    ALLOC_MASTERY = 0.90    
    MIN_WEIGHT = 0.15       
    PROBE_EPISODES = 2000  
    PROBE_WINDOW = 200      
    PROBE_SUCCESS = 0.08    
    PROBE_WEIGHT = 0.04     
    MAX_FAMILY_EPISODES = 4000   
    MAX_SHARE = 0.60       
    BLOCK = 100             
    STALL_BLOCKS = 3        
    STALL_DELTA = 0.03      
    MIN_FAMILY_EPISODES = 600
    FRAC_NEAR, FRAC_FAR = 0.25, 0.85   
    BAND = 0.15
    COVERAGE_SHARE = 0.25    
 
    def __init__(self):
        self.history: Dict[str, List[bool]] = {n: [] for n in TASK_NAMES}
        self.family_done = False
        self.p_and_p_unlocked = False
        self.stack_unlocked = False
        self.stack_episode_count = 0
        self._family_episodes = 0
        self._best = {n: 0.0 for n in self.NON_GRASPING}
        self._stalled_blocks = 0
        self.unlock_reason = None
 
    def _rolling_success(self, name: str) -> float:
        h = self.history[name]
        return float(np.mean(h[-self.WINDOW:])) if h else 0.0
 
    def _is_mastered(self, name: str, threshold: float) -> bool:
        return self._rolling_success(name) >= threshold and len(self.history[name]) >= self.WINDOW
 
    def _active_tasks(self) -> List[str]:
        active = list(self.NON_GRASPING)
        if self.p_and_p_unlocked:
            active.append("pickandplace")
        if self.stack_unlocked:
            active.append("stack")
        return active
 
    def _task_probabilities(self) -> Tuple[List[str], np.ndarray]:
        active = self._active_tasks()
        w = np.array([max(self.MIN_WEIGHT, 1.0 - self._rolling_success(n) / self.ALLOC_MASTERY) for n in active])
        for n in ("pickandplace", "stack"):   # no visible progress -> probe share only
            if n in active and len(self.history[n]) >= self.PROBE_EPISODES and float(np.mean(self.history[n][-self.PROBE_WINDOW:])) < self.PROBE_SUCCESS:
                w[active.index(n)] = self.PROBE_WEIGHT
        if "stack" in active:   # same ramp as HC2 for the newly introduced stack
            stack_share = min(self.stack_episode_count / self.STACK_RAMP_EPISODES, self.STACK_MAX_SHARE)
            w[active.index("stack")] *= stack_share
        p = w / w.sum()
        for _ in range(3):      # cap + renormalise (a few passes is enough for 5 tasks)
            p = np.minimum(p, self.MAX_SHARE)
            p = p / p.sum()
        return active, p
 
    def propose(self, rng: np.random.Generator) -> Tuple[int, np.ndarray]:
        active, p = self._task_probabilities()
        name = active[int(rng.choice(len(active), p=p))]
        spec = TASK_SPECS[name]
        low, high = spec.goal_low, spec.goal_high
        free = (high - low) > 1e-9
        center = (low + high) / 2.0
        half_diag = np.linalg.norm(((high - low) / 2.0)[free])
 
        if rng.random() < self.COVERAGE_SHARE:
            frac = rng.uniform(0.10, 0.90)
        else:
            skill = float(np.clip(self._rolling_success(name) / self.ADVANCE_THRESHOLD, 0.0, 1.0))
            c = self.FRAC_NEAR + (self.FRAC_FAR - self.FRAC_NEAR) * skill
            frac = rng.uniform(max(c - self.BAND, 0.0), min(c + self.BAND, 1.0))
        g = rng.normal(size=int(free.sum()))
        direction = np.zeros(len(low))
        direction[free] = g / max(np.linalg.norm(g), 1e-9)
        target = np.clip(center + direction * frac * half_diag, low, high)
        return spec.task_id, target.astype(np.float32)
 
    def notify_result(self, task_id: int, success: bool) -> None:
        name = TASK_NAMES[task_id]
        self.history[name].append(bool(success))
 
        if not self.p_and_p_unlocked:
            if name in self.NON_GRASPING:
                self._family_episodes += 1
                if self._family_episodes % self.BLOCK == 0:
                    improved = False
                    for n in self.NON_GRASPING:
                        r = self._rolling_success(n)
                        if r > self._best[n] + self.STALL_DELTA:
                            improved = True
                        self._best[n] = max(self._best[n], r)
                    self._stalled_blocks = 0 if improved else self._stalled_blocks + 1
            if all(self._is_mastered(n, self.ADVANCE_THRESHOLD) for n in self.NON_GRASPING):
                self.family_done, self.p_and_p_unlocked, self.unlock_reason = True, True, "mastered"
            elif self._family_episodes >= self.MIN_FAMILY_EPISODES and self._stalled_blocks >= self.STALL_BLOCKS:
                self.p_and_p_unlocked, self.unlock_reason = True, "stalled"
            elif self._family_episodes >= self.MAX_FAMILY_EPISODES:
                self.p_and_p_unlocked, self.unlock_reason = True, "family_budget"
 
        if self.p_and_p_unlocked and not self.stack_unlocked and self._is_mastered("pickandplace", self.THRESHOLD_START_STACK):
            self.stack_unlocked = True
        if self.stack_unlocked:
            self.stack_episode_count += 1
 
    def snapshot(self) -> Dict:
        active, p = self._task_probabilities()
        return {"family_done": self.family_done, "p_and_p_unlocked": self.p_and_p_unlocked,
                "stack_unlocked": self.stack_unlocked, "unlock_reason": self.unlock_reason,
                "family_episodes": self._family_episodes,
                "task_probability": {n: float(x) for n, x in zip(active, p)},
                "rolling_success": {n: self._rolling_success(n) for n in TASK_NAMES}}
 
# 5. MultiTaskCurriculumEnv

class MultiTaskCurriculumEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, curriculum: Curriculum, seed: int = 0, held_out_eval: bool = False,
                 log_goals: bool = False, snapshot_every: int = 0):
        super().__init__()
        self.curriculum = curriculum
        self.held_out_eval = held_out_eval
        self._forced_episode: Optional[Tuple[int, np.ndarray]] = None
        self.rng = np.random.default_rng(seed)

        # [GOAL LOG] one record per TRAINING episode: where/when the curriculum sent the robot, and the outcome
        self.log_goals = log_goals
        self.snapshot_every = snapshot_every
        self.total_steps = 0                       # env steps taken = SB3 num_timesteps
        self.goal_log: List[Dict] = []
        self.snapshots: List[Dict] = []
        self._current: Optional[Dict] = None

        self.envs = {name: gym.make(TASK_ENV_IDS[name]) for name in TASK_NAMES}
        for e in self.envs.values():
            e.reset(seed=int(self.rng.integers(0, 2**31 - 1)))

        self.action_space = spaces.Box(-1.0, 1.0, shape=(SHARED_ACTION_DIM,), dtype=np.float32)
        obs_dim = MAX_ROBOT_OBS_DIM + N_TASKS
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-10.0, 10.0, shape=(obs_dim,), dtype=np.float32),
                achieved_goal=spaces.Box(-10.0, 10.0, shape=(MAX_GOAL_DIM,), dtype=np.float32),
                desired_goal=spaces.Box(-10.0, 10.0, shape=(MAX_GOAL_DIM,), dtype=np.float32),
            )
        )
        self.compute_reward = shared_compute_reward

        self.active_task_id: Optional[int] = None
        self._elapsed = 0

    def _set_goal(self, name: str, curriculum_target: np.ndarray) -> None:
        env = self.envs[name].unwrapped
        spec = TASK_SPECS[name]
        if name == "stack":
            goal6 = expand_stack_goal(curriculum_target, spec)
            env.task.goal = goal6
            env.sim.set_base_pose("target1", goal6[:3], np.array([0.0, 0.0, 0.0, 1.0]))
            env.sim.set_base_pose("target2", goal6[3:], np.array([0.0, 0.0, 0.0, 1.0]))
        else:
            env.task.goal = curriculum_target.astype(np.float32)
            env.sim.set_base_pose("target", curriculum_target, np.array([0.0, 0.0, 0.0, 1.0]))

    def _build_obs(self, name: str) -> Dict[str, np.ndarray]:
        task_id = TASK_SPECS[name].task_id
        env = self.envs[name].unwrapped
        x = pad_robot_obs(env.robot.get_obs()[: (7 if not TASK_SPECS[name].block_gripper else 6)])
        achieved = pad_goal(env.task.get_achieved_goal(), name)
        desired = pad_goal(env.task.get_goal(), name)
        return {
            "observation": np.concatenate([x, one_hot_task(task_id)]).astype(np.float32),
            "achieved_goal": achieved,
            "desired_goal": desired,
        }

    def set_forced_episode(self, task_id: int, target: np.ndarray) -> None:
        self._forced_episode = (task_id, target)

    def _record_goal(self, name: str, task_id: int, target: np.ndarray) -> None:
        """[GOAL LOG] Start of a record. `dist_frac` = distance of the goal from the task's centre, as a
        fraction of its half-diagonal (0 = centre, 1 = corner): comparable across the 5 tasks."""
        spec = TASK_SPECS[name]
        free = (spec.goal_high - spec.goal_low) > 1e-9
        center = (spec.goal_low + spec.goal_high) / 2.0
        half_diag = np.linalg.norm(((spec.goal_high - spec.goal_low) / 2.0)[free])
        frac = float(np.linalg.norm((np.asarray(target) - center)[free]) / half_diag)
        self._current = {"step": self.total_steps, "task": int(task_id),
                         "target": [float(v) for v in np.asarray(target).reshape(-1)], "dist_frac": frac}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if self._forced_episode is not None:
            task_id, target = self._forced_episode
            self._forced_episode = None
        else:
            task_id, target = self.curriculum.propose(self.rng)
            name = TASK_NAMES[task_id]
            target = enforce_held_out_exclusion(name, target, self.rng)
            if self.log_goals:
                self._record_goal(name, task_id, target)   # the goal actually used (after the exclusion)

        name = TASK_NAMES[task_id]
        self.active_task_id = task_id
        self.envs[name].reset()
        self._set_goal(name, target)
        obs = self._build_obs(name)
        self._elapsed = 0
        success = shared_is_success(obs["achieved_goal"], obs["desired_goal"], task_id)
        info = {"is_success": success, "task_id": task_id}
        return obs, info

    def step(self, action: np.ndarray):
        name = TASK_NAMES[self.active_task_id]
        spec = TASK_SPECS[name]
        native_action_dim = 3 if spec.block_gripper else 4
        sliced = np.asarray(action, dtype=np.float32)[:native_action_dim]
        _native_obs, _native_reward, terminated, truncated, _native_info = self.envs[name].step(sliced)

        obs = self._build_obs(name)
        reward = float(shared_compute_reward(obs["achieved_goal"], obs["desired_goal"], {"task_id": self.active_task_id}))
        success = reward > -0.5
        terminated = bool(success)
        info = {"is_success": success, "task_id": self.active_task_id}
        self._elapsed += 1
        self.total_steps += 1

        if terminated or truncated:
            self.curriculum.notify_result(self.active_task_id, success)
            if self.log_goals and self._current is not None:       # [GOAL LOG] close the record
                self._current["success"] = bool(success)
                self._current["length"] = self._elapsed
                self.goal_log.append(self._current)
                self._current = None
        if self.snapshot_every and self.total_steps % self.snapshot_every == 0:   # [GOAL LOG]
            self.snapshots.append({"step": self.total_steps, **_jsonable(self.curriculum.snapshot())})
        return obs, reward, terminated, truncated, info

    def save_goal_log(self, path_prefix: str) -> None:
        """[GOAL LOG] <prefix>.npz (one row per episode) + <prefix>_snapshots.json (curriculum state)."""
        log = self.goal_log
        targets = np.zeros((len(log), 3), dtype=np.float32)       # stack has 2 free dims -> padded with 0
        for i, r in enumerate(log):
            targets[i, : len(r["target"])] = r["target"]
        os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
        np.savez_compressed(
            path_prefix + ".npz",
            step=np.array([r["step"] for r in log], dtype=np.int64),
            task=np.array([r["task"] for r in log], dtype=np.int8),
            target=targets,
            dist_frac=np.array([r["dist_frac"] for r in log], dtype=np.float32),
            success=np.array([r["success"] for r in log], dtype=bool),
            length=np.array([r["length"] for r in log], dtype=np.int32),
        )
        with open(path_prefix + "_snapshots.json", "w") as f:
            json.dump(self.snapshots, f)

    def close(self):
        for e in self.envs.values():
            e.close()


# 6. Training / evaluation harness

N_EVAL_EPISODES_PER_ZONE = 5
EVAL_FREQ = 1000
TOTAL_TIMESTEPS =2_000_000
SNAPSHOT_EVERY = 10_000      # [GOAL LOG] curriculum state saved every N training steps
HER_KWARGS = dict(n_sampled_goal=4, goal_selection_strategy="future", copy_info_dict=True)
SAC_KWARGS = dict(
    learning_rate=1e-3,
    buffer_size=TOTAL_TIMESTEPS,
    learning_starts=1_000,
    batch_size=256,
    train_freq=1,
    gradient_steps=1,
    gamma=0.95,
    tau=0.05,
    policy_kwargs=dict(net_arch=[256, 256]),
)


def evaluate_held_out(model, eval_env: "MultiTaskCurriculumEnv", n_episodes_per_zone: int = N_EVAL_EPISODES_PER_ZONE,
                       deterministic: bool = True) -> Dict[str, float]:
    per_task: Dict[str, float] = {}
    for name in TASK_NAMES:
        spec = TASK_SPECS[name]
        free = (spec.goal_high - spec.goal_low) > 1e-9
        successes = []
        for zone in HELD_OUT_ZONES[name]:
            for _ in range(n_episodes_per_zone):
                direction = np.zeros(spec.goal_dim)
                direction[free] = _sample_direction(eval_env.rng, int(free.sum()))
                magnitude = eval_env.rng.uniform(0.0, zone["radius_eval"])
                target = (zone["center"] + direction * magnitude).astype(np.float32)

                eval_env.set_forced_episode(spec.task_id, target)
                obs, info = eval_env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                successes.append(bool(info["is_success"]))
        per_task[name] = float(np.mean(successes))
    return per_task


class HeldOutEvalCallback(BaseCallback):
    def __init__(self, eval_env: "MultiTaskCurriculumEnv", eval_freq: int,
                 n_episodes_per_zone: int = N_EVAL_EPISODES_PER_ZONE, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_episodes_per_zone = n_episodes_per_zone
        self.history: List[Dict] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            per_task = evaluate_held_out(self.model, self.eval_env, self.n_episodes_per_zone)
            entry = {"timesteps": self.num_timesteps, **{f"success_{k}": v for k, v in per_task.items()}}
            self.history.append(entry)
            if self.verbose:
                print(f"  [held-out eval @ {self.num_timesteps:>7d}] " +
                      " ".join(f"{k}={v:.2f}" for k, v in per_task.items()))
        return True


def make_model(env: "MultiTaskCurriculumEnv", seed: int):
    return SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=HER_KWARGS,
        seed=seed,
        verbose=0,
        **SAC_KWARGS,
    )


def train_one_run(curriculum_factory, seed: int, total_timesteps: int, eval_freq: int,
                   n_episodes_per_zone: int = N_EVAL_EPISODES_PER_ZONE, verbose: int = 0,
                   goal_log_path: Optional[str] = None, snapshot_every: int = SNAPSHOT_EVERY) -> List[Dict]:
    """goal_log_path: prefix of the files where the curriculum's decisions are saved (None = no logging)."""
    set_random_seed(seed)

    train_env = MultiTaskCurriculumEnv(curriculum_factory(), seed=seed,
                                       log_goals=goal_log_path is not None, snapshot_every=snapshot_every)
    eval_env = MultiTaskCurriculumEnv(RandomCurriculum(), seed=seed + 10_000, held_out_eval=True)

    model = make_model(train_env, seed=seed)
    cb = HeldOutEvalCallback(eval_env, eval_freq=eval_freq, n_episodes_per_zone=n_episodes_per_zone, verbose=verbose)
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=False)

    if goal_log_path is not None:
        train_env.save_goal_log(goal_log_path)

    train_env.close()
    eval_env.close()
    return cb.history


def _overall_success(entry: Dict) -> float:
    return float(np.mean([v for k, v in entry.items() if k.startswith("success_")]))


# 7. Per-seed main:

CURRICULUMS = {
    "random": RandomCurriculum,
    "hand crafted 1(naive)": HandCraftedCurriculum,
    "hand crafted 2 (domain-informed)": HandCraftedCurriculum2,
   # "Adaptive": UCBAdaptiveCurriculum,
    #"InformedAdaptiveCurriculum":InformedAdaptiveCurriculum
}
STRATEGIES = list(CURRICULUMS.keys())


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS,
                         help="Which seed to run (one job = one seed).")
    parser.add_argument("--out-dir", type=str, default=".",
                         help="Directory to write results_phase1_seed<SEED>.json into.")
    parser.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY,
                         help="Save the curriculum's internal state every N training steps.")
    args = parser.parse_args()
    seed = args.seed

    results_path = os.path.join(args.out_dir, f"results/results_phase1_seed{seed}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)      # [FIX] was missing: crash after hours of training

    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {}

    t0 = time.time()
    for name, curriculum_cls in CURRICULUMS.items():
        if name in results:
            print(f"skip strategy={name!r} seed={seed} (already done)")
            print(f"success rate (mean over tasks): {_overall_success(results[name][-1]):.2f}")
            continue
        print(f"strategy={name}  seed={seed}  ({time.time() - t0:.0f}s elapsed)")
        goal_prefix = os.path.join(args.out_dir, "results", "goals", f"{_slug(name)}_seed{seed}")
        history = train_one_run(curriculum_cls, seed, total_timesteps=TOTAL_TIMESTEPS, eval_freq=EVAL_FREQ,
                                 n_episodes_per_zone=N_EVAL_EPISODES_PER_ZONE, verbose=0,
                                 goal_log_path=goal_prefix, snapshot_every=args.snapshot_every)
        results[name] = history
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"success rate (mean over tasks): {_overall_success(history[-1]):.2f}")

    print(f"seed {seed} done in {time.time() - t0:.0f}s -> {results_path}")


if __name__ == "__main__":
    main()