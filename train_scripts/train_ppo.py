"""Train the ontology-guided MAPPO agent with curriculum learning.

Curriculum stages (auto-advance at 80% completion over 100 episodes):
  Stage 1 — banana + mango           (2 items, living_room only)
  Stage 2 — + orange + grapes        (4 items, full main floor)
  Stage 3 — all 8 items              (full house)

Run:
    python train_ppo.py

Resume from checkpoint:
    Set RESUME_FROM to the saved .zip path.
"""

import csv
import os
import re
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer

from envs.environment_sb3 import HouseEnvSB3
from helper_functions.goals import GOALS
from mappo_policy import MAPPOPolicy
from mappo import MAPPO

ONTOLOGY_PATH  = "/Users/m.manso/Downloads/thesisont_updated-2.owl"
GOAL_NAME      = "collect_food"
MAX_EP_STEPS   = 4000
TOTAL_STEPS    = 45_000_000
SAVE_EVERY_EPS = 200
_HERE          = os.path.dirname(os.path.abspath(__file__))
RESUME_FROM    = None
RESUME_STAGE   = 0
LOG_CSV        = os.path.join(_HERE, "checkpoints_ppo", "training_log.csv")

CURRICULUM_STAGES = [
    ["banana"],
    ["banana", "mango"],
    ["banana", "mango", "orange", "grapes"],
    None,
]
STAGE_MAX_STEPS = [2000, 2000, 3000, 4000]

STAGE_ADVANCE_RATE   = 0.80
STAGE_ADVANCE_WINDOW = 100

os.makedirs(os.path.join(_HERE, "checkpoints_ppo"), exist_ok=True)
active_goal = GOALS[GOAL_NAME]
N_ITEMS     = len(active_goal.target_items)
_GLOBAL_DIM = N_ITEMS * 4

_START_EP = 0
if RESUME_FROM:
    _m = re.search(r"ep(\d+)", RESUME_FROM)
    if _m:
        _START_EP = int(_m.group(1))


def make_env(active_items=None, max_steps=None):
    def _init():
        env = HouseEnvSB3(
            ontology_path=ONTOLOGY_PATH,
            goal=active_goal,
            num_agents=3,
            max_steps=max_steps or STAGE_MAX_STEPS[0],
            active_items=active_items,
        )
        return Monitor(env)
    return _init


_initial_stage = RESUME_STAGE if RESUME_FROM else 0
_raw_env = DummyVecEnv([make_env(active_items=CURRICULUM_STAGES[_initial_stage],
                                  max_steps=STAGE_MAX_STEPS[_initial_stage])])
vec_env  = VecNormalize(_raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

model = MAPPO(
    MAPPOPolicy,
    vec_env,
    verbose=0,
    learning_rate=1e-4,
    n_steps=8000,
    batch_size=800,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,
    vf_coef=0.3,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256], n_global=_GLOBAL_DIM),
    tensorboard_log=None,
)

if RESUME_FROM and os.path.exists(RESUME_FROM):
    vecnorm_path = RESUME_FROM.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, _raw_env)
    model = MAPPO.load(RESUME_FROM, env=vec_env)
    model.n_steps    = 4 * STAGE_MAX_STEPS[RESUME_STAGE]
    model.batch_size = model.n_steps // 10
    model.vf_coef    = 0.3
    model.ent_coef   = 0.01
    model.rollout_buffer = RolloutBuffer(
        model.n_steps, model.observation_space, model.action_space,
        device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
    )
    house_env = vec_env.venv.envs[0].env
    _resume_active = CURRICULUM_STAGES[RESUME_STAGE]
    house_env.active_items = set(_resume_active) if _resume_active is not None else None
    house_env.max_steps    = STAGE_MAX_STEPS[RESUME_STAGE]
    print(f"Resumed from {RESUME_FROM}  (episode offset: {_START_EP})")
    print(f"Resuming on Stage {RESUME_STAGE}: {_resume_active or 'ALL'}")


_CSV_FIELDS = [
    "ep", "reward", "avg50", "balls", "done", "seen", "rooms", "steps", "time_s",
    "expl", "guide", "pickup", "pen", "delivery", "obs", "lava", "completion",
    "stage", "reasoner_calls", "reasoner_time_ep", "reasoner_ms_per_call",
]


class PPOCallback(BaseCallback):
    """One-line terminal output per episode; full reward breakdown logged to CSV."""

    SEP = "─" * 76

    def __init__(self, save_every: int = 100, start_ep: int = 0):
        super().__init__()
        self.save_every = save_every
        self.ep_count   = start_ep
        self.recent50: list[float] = []
        self.start_time = time.time()
        self._stage     = 0

        self._comp_buf: dict[str, list[float]] = {
            k: [] for k in ("expl", "guide", "pickup", "pen",
                            "delivery", "obs", "lava", "completion")
        }

        csv_exists = os.path.exists(LOG_CSV)
        self._csv_fh = open(LOG_CSV, "a", newline="")
        self._csv_w  = csv.DictWriter(self._csv_fh, fieldnames=_CSV_FIELDS)
        if not csv_exists:
            self._csv_w.writeheader()

        print(f"\nGoal : {active_goal.name}")
        print(f"Items: {N_ITEMS}  |  target: {active_goal.target_room}  |  agents: 3")
        print(f"Curriculum: {len(CURRICULUM_STAGES)} stages")
        for i, s in enumerate(CURRICULUM_STAGES):
            n = len(s) if s else N_ITEMS
            print(f"  Stage {i+1}: {n} items — {s or 'ALL'}")
        if start_ep:
            print(f"Resuming from episode {start_ep}")
        active = CURRICULUM_STAGES[self._stage]
        n_start = len(active) if active else N_ITEMS
        print(f"\nStarting Stage {self._stage + 1}: {n_start} items — {active or 'ALL'}")
        print(self.SEP)
        print(f"{'Ep':>5}  {'Reward':>9}  {'Avg50':>9}  {'Balls':>5}  {'Done':>4}  "
              f"{'Seen':>5}  {'Rooms':>5}  {'Steps':>10}  {'Time':>7}  {'Stage':>5}")
        print(self.SEP)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info is None:
                continue

            self.ep_count += 1
            ep_r = ep_info["r"]
            self.recent50.append(ep_r)
            if len(self.recent50) > 50:
                self.recent50.pop(0)

            avg50 = np.mean(self.recent50)
            balls = info.get("balls_delivered", 0)
            done  = "✓" if info.get("task_complete", False) else "✗"
            seen  = info.get("items_observed", 0)
            rooms = info.get("rooms_explored", 0)
            secs  = int(time.time() - self.start_time)
            comp  = info.get("reward_components") or {}
            rstat = info.get("reasoner_stats") or {}
            stage = self._stage + 1

            s_items = CURRICULUM_STAGES[self._stage]
            n_active = len(s_items) if s_items else N_ITEMS

            print(f"{self.ep_count:5d}  {ep_r:9.2f}  {avg50:9.2f}  "
                  f"{balls:>2}/{n_active:<2}  {done:>4}  "
                  f"{seen:>2}/{n_active:<2}  {rooms:>5}  "
                  f"{self.num_timesteps:>10}  {secs:>6}s  {stage:>5}")

            self._csv_w.writerow({
                "ep": self.ep_count, "reward": round(ep_r, 2),
                "avg50": round(avg50, 2), "balls": balls,
                "done": int(done == "✓"), "seen": seen, "rooms": rooms,
                "steps": self.num_timesteps, "time_s": secs, "stage": stage,
                "reasoner_calls":       rstat.get("reasoner_calls_ep", 0),
                "reasoner_time_ep":     rstat.get("reasoner_time_ep", 0.0),
                "reasoner_ms_per_call": rstat.get("reasoner_ms_per_call", 0.0),
                **{k: round(comp.get(k, 0.0), 2) for k in self._comp_buf},
            })
            self._csv_fh.flush()

            for k in self._comp_buf:
                self._comp_buf[k].append(comp.get(k, 0.0))

            if self.ep_count % self.save_every == 0:
                path = os.path.join(_HERE, "checkpoints_ppo",
                                    f"ppo_ep{self.ep_count:04d}_s{stage}")
                self.model.save(path)
                env = self.model.get_env()
                if isinstance(env, VecNormalize):
                    env.save(path + "_vecnorm.pkl")
                n = self.save_every
                m = {k: np.mean(v[-n:]) for k, v in self._comp_buf.items()}
                print(self.SEP)
                print(f"  [ep {self.ep_count}]  stage={stage}  avg50={avg50:+.1f}  "
                      f"steps={self.num_timesteps}  saved → {path}.zip")
                print(f"  Mean reward breakdown (last {n} eps):")
                print(f"    expl={m['expl']:+.1f}  guide={m['guide']:+.1f}  "
                      f"delivery={m['delivery']:+.1f}  completion={m['completion']:+.1f}")
                print(f"  Reasoner (last ep): {rstat.get('reasoner_calls_ep', 0)} calls, "
                      f"{rstat.get('reasoner_time_ep', 0.0):.2f}s, "
                      f"{rstat.get('reasoner_ms_per_call', 0.0):.1f} ms/call")
                print(self.SEP)

        return True

    def _on_training_end(self) -> None:
        self._csv_fh.close()


class CurriculumCallback(PPOCallback):
    """PPOCallback + automatic curriculum stage advancement."""

    def __init__(self, save_every: int = 100, start_ep: int = 0, start_stage: int = 0):
        super().__init__(save_every=save_every, start_ep=start_ep)
        self._stage      = start_stage
        self._stage_hist: list[int] = []

    def _on_step(self) -> bool:
        result = super()._on_step()

        for info in self.locals.get("infos", []):
            if info.get("episode") is None:
                continue

            self._stage_hist.append(int(info.get("task_complete", False)))
            if len(self._stage_hist) > STAGE_ADVANCE_WINDOW:
                self._stage_hist.pop(0)

            if (self._stage < len(CURRICULUM_STAGES) - 1
                    and len(self._stage_hist) >= STAGE_ADVANCE_WINDOW
                    and np.mean(self._stage_hist) >= STAGE_ADVANCE_RATE):
                self._advance_stage()

        return result

    def _advance_stage(self) -> None:
        self._stage += 1
        new_active   = CURRICULUM_STAGES[self._stage]
        new_maxsteps = STAGE_MAX_STEPS[self._stage]

        vec_env   = self.model.get_env()
        house_env = vec_env.venv.envs[0].env
        house_env.active_items = set(new_active) if new_active is not None else None
        house_env.max_steps    = new_maxsteps

        new_n_steps = 4 * new_maxsteps
        self.model.n_steps    = new_n_steps
        self.model.batch_size = new_n_steps // 10
        self.model.rollout_buffer = RolloutBuffer(
            new_n_steps, self.model.observation_space, self.model.action_space,
            device=self.model.device, gamma=self.model.gamma,
            gae_lambda=self.model.gae_lambda, n_envs=self.model.n_envs,
        )

        path = os.path.join(_HERE, "checkpoints_ppo",
                            f"ppo_stage{self._stage}_ep{self.ep_count:04d}")
        self.model.save(path)
        env = self.model.get_env()
        if isinstance(env, VecNormalize):
            env.save(path + "_vecnorm.pkl")

        self._stage_hist = []

        n_new = len(new_active) if new_active else N_ITEMS
        print(f"\n{'='*76}")
        print(f"  CURRICULUM ADVANCE  →  Stage {self._stage + 1}/{len(CURRICULUM_STAGES)}")
        print(f"  Items: {n_new}  —  {new_active or 'ALL 8'}")
        print(f"  Max episode steps: {new_maxsteps}")
        print(f"  Ep {self.ep_count}  |  steps {self.num_timesteps}")
        print(f"  Checkpoint saved → {path}.zip")
        print(f"{'='*76}\n")


model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=CurriculumCallback(save_every=SAVE_EVERY_EPS, start_ep=_START_EP,
                               start_stage=RESUME_STAGE if RESUME_FROM else 0),
    reset_num_timesteps=(RESUME_FROM is None),
)

model.save(os.path.join(_HERE, "checkpoints_ppo", "ppo_final"))
vec_env.close()
print("Training complete.")
