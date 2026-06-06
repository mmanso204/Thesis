"""Smoke test for the in-loop reasoner + competency questions.

Run on a machine with a working JVM (HermiT needs Java):
    python train_scripts/smoke_reasoner.py

Validates:
  1. HermiT actually runs inside the env step (no JVM/owlapy crash)
  2. Real reasoner cost  → ms/call  (decides whether per-room-change is affordable)
  3. The ontology answers the 8 competency questions via DL reasoning
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.environment_sb3 import HouseEnvSB3
from helper_functions.goals import GOALS

ONT = os.environ.get("ONTOLOGY_PATH", "/Users/m.manso/Downloads/thesisont_updated-2.owl")
N_STEPS = 300

env = HouseEnvSB3(ontology_path=ONT, goal=GOALS["collect_food"], num_agents=3,
                  max_steps=N_STEPS, active_items=["banana", "mango"])
obs, _ = env.reset()

t0 = time.perf_counter()
reach_changes = 0
prev_reach = env._ont._ont_reachable_cached
for _ in range(N_STEPS):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if env._ont._ont_reachable_cached != prev_reach:
        reach_changes += 1
        prev_reach = env._ont._ont_reachable_cached
    if term or trunc:
        break
wall = time.perf_counter() - t0

stats = env._ont.reasoner_stats()
print("\n── reasoner timing ──────────────────────────────")
print(f"  wall time for {N_STEPS} steps : {wall:.1f}s")
print(f"  reasoner calls (this ep)     : {stats['reasoner_calls_ep']}")
print(f"  reasoner time (this ep)      : {stats['reasoner_time_ep']:.2f}s "
      f"({100*stats['reasoner_time_ep']/max(wall,1e-9):.0f}% of wall)")
print(f"  ms per reasoner call         : {stats['reasoner_ms_per_call']:.1f}")
print(f"  reachability value changes   : {reach_changes}")

print("\n── competency questions (DL reasoning) ──────────")
report = env._ont.competency_report()
for k in sorted(report):
    print(f"  {k:18s}: {report[k]}")
print()
