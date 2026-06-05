"""Evaluate a trained MAPPO checkpoint on the collect-food task.

Runs N episodes with the trained policy and reports delivery / completion
statistics and the reward-component breakdown.  Optionally renders.

Examples:
    # Evaluate the latest Stage-3 checkpoint over 50 episodes
    python evaluate_ppo.py --checkpoint checkpoints_ppo/ppo_ep20800_s3 --stage 2 --episodes 50

    # Watch 5 episodes render in a window
    python evaluate_ppo.py --checkpoint checkpoints_ppo/ppo_ep20800_s3 --stage 2 --episodes 5 --render

    # Sample actions instead of deterministic argmax
    python evaluate_ppo.py --checkpoint checkpoints_ppo/ppo_ep20800_s3 --stage 2 --stochastic
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.environment_sb3 import HouseEnvSB3
from helper_functions.goals import GOALS
from mappo_policy import MAPPOPolicy  # noqa: F401  (registers custom policy for load)
from mappo import MAPPO

# ── config ────────────────────────────────────────────────────────────
ONTOLOGY_PATH = "/Users/m.manso/Downloads/thesisont_updated-2.owl"
GOAL_NAME     = "collect_food"
_HERE         = os.path.dirname(os.path.abspath(__file__))

# Must match train_ppo.py
CURRICULUM_STAGES = [
    ["banana"],                                   # Stage 0
    ["banana", "mango"],                          # Stage 1
    ["banana", "mango", "orange", "grapes"],      # Stage 2
    None,                                         # Stage 3: all 8
]
STAGE_MAX_STEPS = [2000, 2000, 3000, 4000]
# ──────────────────────────────────────────────────────────────────────


def make_env(goal, active_items, max_steps, render_mode=None):
    def _init():
        env = HouseEnvSB3(
            ontology_path=ONTOLOGY_PATH,
            goal=goal,
            num_agents=3,
            max_steps=max_steps,
            active_items=active_items,
            render_mode=render_mode,
        )
        return Monitor(env)
    return _init


def evaluate(checkpoint, stage, episodes, deterministic, render):
    goal         = GOALS[GOAL_NAME]
    active_items = CURRICULUM_STAGES[stage]
    max_steps    = STAGE_MAX_STEPS[stage]
    n_target     = len(active_items) if active_items is not None else len(goal.target_items)

    render_mode = "human" if render else None
    raw_env = DummyVecEnv([make_env(goal, active_items, max_steps, render_mode)])

    # Load VecNormalize stats if present; disable updating + reward norm for eval
    vecnorm_path = checkpoint.replace(".zip", "") + "_vecnorm.pkl"
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, raw_env)
        env.training    = False    # freeze running stats
        env.norm_reward = False    # report RAW rewards
        print(f"Loaded VecNormalize stats from {os.path.basename(vecnorm_path)}")
    else:
        env = raw_env
        print("No VecNormalize stats found — using raw env")

    model = MAPPO.load(checkpoint, env=env)

    print(f"\nEvaluating: {os.path.basename(checkpoint)}")
    print(f"Stage {stage}: {active_items or 'ALL'}  ({n_target} items)  |  "
          f"max_steps={max_steps}  |  {'deterministic' if deterministic else 'stochastic'}")
    print("─" * 72)
    print(f"{'Ep':>4}  {'Reward':>9}  {'Delivered':>9}  {'Complete':>8}  {'Steps':>6}")
    print("─" * 72)

    results = []
    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = np.array([False])
        ep_reward = 0.0
        last_info = {}
        steps = 0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, infos = env.step(action)
            ep_reward += float(reward[0])
            last_info = infos[0]
            steps += 1
            if render:
                env.render()

        delivered = last_info.get("balls_delivered", 0)
        complete  = bool(last_info.get("task_complete", False))
        comps     = last_info.get("reward_components") or {}
        results.append({
            "reward": ep_reward, "delivered": delivered,
            "complete": complete, "steps": steps, "comps": comps,
        })
        print(f"{ep:>4}  {ep_reward:>9.1f}  {delivered:>5}/{n_target:<3}  "
              f"{'✓' if complete else '✗':>8}  {steps:>6}")

    # ── summary ────────────────────────────────────────────────────────
    rewards   = [r["reward"] for r in results]
    delivered = [r["delivered"] for r in results]
    completes = [r["complete"] for r in results]

    print("─" * 72)
    print(f"\nSUMMARY over {episodes} episodes (Stage {stage}, {n_target} items)")
    print(f"  Completion rate : {sum(completes)}/{episodes} = {np.mean(completes):.1%}")
    print(f"  Avg delivered   : {np.mean(delivered):.2f} / {n_target}")
    print(f"  Max delivered   : {max(delivered)} / {n_target}")
    print(f"  Avg reward      : {np.mean(rewards):.1f}  (std {np.std(rewards):.1f})")
    print(f"  Reward range    : {min(rewards):.1f} … {max(rewards):.1f}")

    # delivery histogram
    print(f"\n  Delivery distribution:")
    for k in range(n_target + 1):
        c = sum(1 for d in delivered if d == k)
        bar = "█" * int(40 * c / episodes)
        print(f"    {k}/{n_target}: {c:>3} ({c/episodes:>5.1%}) {bar}")

    # mean component breakdown (only over episodes that logged components)
    keys = ["expl", "guide", "pickup", "pen", "delivery", "completion"]
    comp_means = {k: np.mean([r["comps"].get(k, 0.0) for r in results if r["comps"]])
                  for k in keys if any(r["comps"] for r in results)}
    if comp_means:
        print(f"\n  Mean reward components:")
        for k, v in comp_means.items():
            print(f"    {k:<11}: {v:+.1f}")

    env.close()
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained MAPPO checkpoint.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint .zip (with or without .zip)")
    p.add_argument("--stage", type=int, default=2,
                   help="Curriculum stage index (0-3). Default 2 (4 items).")
    p.add_argument("--episodes", type=int, default=50, help="Number of eval episodes")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions instead of deterministic argmax")
    p.add_argument("--render", action="store_true", help="Render episodes in a window")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt = args.checkpoint if args.checkpoint.endswith(".zip") else args.checkpoint + ".zip"
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(_HERE, ckpt) if not os.path.exists(ckpt) else ckpt
    if not os.path.exists(ckpt):
        sys.exit(f"Checkpoint not found: {ckpt}")
    evaluate(
        checkpoint=ckpt,
        stage=args.stage,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
    )
