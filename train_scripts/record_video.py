"""Record a video of a trained MAPPO checkpoint running the eval task.

Same env/checkpoint loading as evaluate_ppo.py, but instead of opening a live
pygame window it grabs ``rgb_array`` frames from the underlying multigrid env
and writes them to an .mp4 (or .gif). Works headless (no display needed), so it
runs fine on the cloud VM where the checkpoints live.

Requires a video backend:  pip install imageio imageio-ffmpeg
(imageio-ffmpeg bundles its own ffmpeg binary, no system ffmpeg needed).
For .gif output only imageio + pillow are needed.

Examples:
    # Record one episode of the run1 checkpoint at stage 3 -> mp4
    python train_scripts/record_video.py \
        --checkpoint checkpoints_run1_ont_food_prox5/ppo_ep6800_s3 \
        --stage 3 --episodes 1 --out run1.mp4

    # Three episodes, sampled actions, slower playback
    python train_scripts/record_video.py --checkpoint ppo_final \
        --stage 3 --episodes 3 --stochastic --fps 8 --out run3.mp4
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

# config (env-overridable so it matches whichever run is being evaluated)
ONTOLOGY_PATH = os.environ.get("ONTOLOGY_PATH", "/Users/m.manso/Downloads/thesisont_updated-2.owl")
GOAL_NAME     = os.environ.get("GOAL_NAME", "collect_food")
USE_ONTOLOGY  = os.environ.get("USE_ONTOLOGY", "1") not in ("0", "false", "False")
PROXIMITY     = int(os.environ.get("PROXIMITY", "5"))
NUM_AGENTS    = 2
_HERE         = os.path.dirname(os.path.abspath(__file__))

# Must match train_ppo.py / evaluate_ppo.py; curriculum is goal-specific.
_CURRICULUM_BY_GOAL = {
    "collect_food": (
        [["banana"], ["banana", "mango"],
         ["banana", "mango", "orange", "grapes"], None],
        [2000, 2000, 4500, 4000],
    ),
    "collect_trash": (
        [["plastic bottle"], ["plastic bottle", "trash bag"],
         ["plastic bottle", "trash bag", "old newspaper", "empty can"], None],
        [2000, 2000, 4500, 6000],
    ),
}
CURRICULUM_STAGES, STAGE_MAX_STEPS = _CURRICULUM_BY_GOAL[GOAL_NAME]


def make_env(goal, active_items, max_steps, render_mode):
    def _init():
        env = HouseEnvSB3(
            ontology_path=ONTOLOGY_PATH,
            goal=goal,
            num_agents=NUM_AGENTS,
            max_steps=max_steps,
            active_items=active_items,
            render_mode=render_mode,
            proximity_threshold=PROXIMITY,
            use_ontology=USE_ONTOLOGY,
        )
        return Monitor(env)
    return _init


def _base_env(vec_env):
    """Drill through VecNormalize -> DummyVecEnv -> Monitor to the raw env."""
    e = vec_env
    if hasattr(e, "venv"):          # VecNormalize wrapper
        e = e.venv
    return e.envs[0]                # DummyVecEnv; Monitor.render() forwards down


def _open_writer(out_path, fps):
    import imageio.v2 as imageio
    if out_path.lower().endswith(".gif"):
        return imageio.get_writer(out_path, mode="I", duration=1.0 / fps)
    # mp4 / other ffmpeg formats; macro_block_size=None avoids resize warnings
    return imageio.get_writer(out_path, fps=fps, macro_block_size=None)


def record(checkpoint, stage, episodes, deterministic, out_path, fps):
    goal         = GOALS[GOAL_NAME]
    active_items = CURRICULUM_STAGES[stage]
    max_steps    = STAGE_MAX_STEPS[stage]
    n_target     = len(active_items) if active_items is not None else len(goal.target_items)

    raw_env = DummyVecEnv([make_env(goal, active_items, max_steps, "rgb_array")])

    vecnorm_path = checkpoint.replace(".zip", "") + "_vecnorm.pkl"
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, raw_env)
        env.training    = False
        env.norm_reward = False
        print(f"Loaded VecNormalize stats from {os.path.basename(vecnorm_path)}")
    else:
        env = raw_env
        print("No VecNormalize stats found, using raw env")

    model = MAPPO.load(checkpoint, env=env)
    base  = _base_env(env)

    try:
        writer = _open_writer(out_path, fps)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Could not open video writer for {out_path}: {e}\n"
                 f"Install a backend:  pip install imageio imageio-ffmpeg")

    print(f"\nRecording {episodes} episode(s) of {os.path.basename(checkpoint)} "
          f"(stage {stage}, {n_target} items) -> {out_path}")

    total_frames = 0
    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = np.array([False])
        ep_reward, steps = 0.0, 0
        writer.append_data(base.render())   # initial frame
        while not done[0]:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, infos = env.step(action)
            ep_reward += float(reward[0])
            steps += 1
            writer.append_data(base.render())
            total_frames += 1
        delivered = infos[0].get("balls_delivered", 0)
        complete  = bool(infos[0].get("task_complete", False))
        print(f"  ep {ep}: reward={ep_reward:7.1f}  delivered={delivered}/{n_target}  "
              f"complete={'✓' if complete else '✗'}  steps={steps}")

    writer.close()
    env.close()
    print(f"\nWrote {total_frames} frames at {fps} fps -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Record a video of a trained MAPPO checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .zip (with or without .zip)")
    p.add_argument("--stage", type=int, default=3, help="Curriculum stage index (0-3). Default 3 (full task).")
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to record")
    p.add_argument("--stochastic", action="store_true", help="Sample actions instead of deterministic argmax")
    p.add_argument("--out", default="eval.mp4", help="Output path (.mp4 or .gif)")
    p.add_argument("--fps", type=int, default=12, help="Playback frames per second")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt = args.checkpoint if args.checkpoint.endswith(".zip") else args.checkpoint + ".zip"
    if not os.path.isabs(ckpt) and not os.path.exists(ckpt):
        ckpt = os.path.join(_HERE, ckpt)
    if not os.path.exists(ckpt):
        sys.exit(f"Checkpoint not found: {ckpt}")
    record(
        checkpoint=ckpt,
        stage=args.stage,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        out_path=args.out,
        fps=args.fps,
    )
