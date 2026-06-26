import os
import time
from envs.environment_multi import HouseEnv
from helper_functions.goals import GOALS
from agent import DQNAgent

ONTOLOGY_PATH = "/Users/m.manso/Downloads/thesisont_updated-2.owl"
GOAL_NAME     = "collect_food"
EPISODES      = 5000
MAX_STEPS     = 3500
SAVE_EVERY    = 200
EPS_DECAY     = 3_000_000
RESUME_FROM   = "checkpoints/dqn_ep0800.pt"   # set to None to start fresh

os.makedirs("checkpoints", exist_ok=True)
active_goal = GOALS[GOAL_NAME]

env   = HouseEnv(num_agents=3, goal=active_goal, max_steps=MAX_STEPS, width=32, height=23)
agent = DQNAgent(ONTOLOGY_PATH, active_goal, eps_decay=EPS_DECAY, buffer_size=100_000)

if RESUME_FROM and os.path.exists(RESUME_FROM):
    agent.load(RESUME_FROM)
    print(f"Resumed from {RESUME_FROM}  (total_steps={agent.total_steps}, ε={agent.epsilon():.3f})")
    START_EP = agent.total_steps // MAX_STEPS + 1
else:
    START_EP = 1

N_ITEMS  = len(active_goal.target_items)
N_AGENTS = 3
SEP      = "-" * 72

print(f"\nGoal : {active_goal.name}")
print(f"Items: {N_ITEMS}  |  target: {active_goal.target_room}  |  agents: {N_AGENTS}")
print(SEP)
print(f"{'Ep':>5}  {'Reward':>8}  {'Loss':>8}  {'ε':>6}  "
      f"{'Balls':>5}  {'Done':>4}  {'Steps':>5}  {'Rooms':>5}  {'Seen':>5}  {'Time':>6}  ETA")
print(SEP)

train_start    = time.time()
recent_rewards = []
ep_count       = 0  # episodes run this session (for ETA)

for ep in range(START_EP, EPISODES + 1):

      ep_count += 1
      ep_start  = time.time()
      s         = agent.run_episode(env)
      ep_secs   = time.time() - ep_start
      elapsed   = time.time() - train_start
      eta_secs  = (elapsed / ep_count) * (EPISODES - ep)
      eta_str   = f"{int(eta_secs // 3600):02d}h{int((eta_secs % 3600) // 60):02d}m"

      recent_rewards.append(s["total_reward"])
      if len(recent_rewards) > 50:
            recent_rewards.pop(0)

      ball_str = f"{s['balls_delivered']}/{N_ITEMS}"
      seen_str = f"{s['items_observed']}/{N_ITEMS}"
      done     = "✓" if s["task_complete"] else "✗"

      print(f"{ep:5d}  {s['total_reward']:8.2f}  {s['avg_loss']:8.4f}  {s['epsilon']:6.3f}  "
            f"{ball_str:>5}  {done:>4}  {s['steps']:>5}  {s['rooms_explored']:>5}  "
            f"{seen_str:>5}  {ep_secs:5.1f}s  ETA:{eta_str}")

      if ep % SAVE_EVERY == 0:
            agent.save(f"checkpoints/dqn_ep{ep:04d}.pt")
            avg50 = sum(recent_rewards) / len(recent_rewards)
            ont_cls   = agent.get_goal_classification()
            ont_items = sorted(s["ont_goal_items"] or [])
            ont_rooms = sorted(s["ont_rooms"] or [])
            items_str = ", ".join(ont_items[:12]) + ("…" if len(ont_items) > 12 else "")
            print(SEP)
            print(f"  [ep {ep}]  avg50 reward: {avg50:+.1f}  |  ε={s['epsilon']:.3f}")
            print(f"  [ep {ep}]  shaping: expl{s['rc_explore']:+.1f}  pick{s['rc_pickup']:+.1f}  "
                  f"guide{s['rc_guidance']:+.1f}  pen{s['rc_penalties']:+.1f}  "
                  f"deliv{s['rc_delivery']:+.1f}  done{s['rc_complete']:+.1f}")
            print(f"  [ep {ep}]  seen ({s['items_observed']}/{N_ITEMS}): {items_str or 'none'}")
            print(f"  [ep {ep}]  rooms ({s['rooms_explored']}): {', '.join(ont_rooms) or 'none'}")
            print(SEP)

agent.save("checkpoints/dqn_final.pt")
env.close()
