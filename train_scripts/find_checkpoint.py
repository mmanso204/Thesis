#!/usr/bin/env python3
"""Find the saved checkpoint closest to a target step count.

Checkpoints are written every SAVE_EVERY_EPS (=200) episodes as
    ppo_ep{ep:04d}_s{stage}.zip   (+ a matching _vecnorm.pkl)
so to evaluate a run "at 20M steps" you need the checkpoint whose logged
`steps` is nearest the target. This scans each run's training_log.csv and
prints that filename.

Usage (run on the VM where the checkpoints live):
    python train_scripts/find_checkpoint.py                 # default: run1 + run2 @ 20M
    python train_scripts/find_checkpoint.py --target 20000000 \
        train_scripts/checkpoints_run1_ont_food_prox5 \
        train_scripts/checkpoints_run2_vanilla_food
"""
import argparse
import csv
import os
import sys

SAVE_EVERY_EPS = 200  # must match train_ppo.py


def nearest_checkpoint(ckpt_dir: str, target: int):
    csv_path = os.path.join(ckpt_dir, "training_log.csv")
    if not os.path.isfile(csv_path):
        return None, f"no training_log.csv in {ckpt_dir}"

    best = None  # (abs_diff, ep, stage, steps)
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                ep = int(row["ep"])
                steps = int(row["steps"])
                stage = int(row["stage"])
            except (KeyError, ValueError):
                continue
            if ep % SAVE_EVERY_EPS != 0:      # only episodes that were saved
                continue
            diff = abs(steps - target)
            if best is None or diff < best[0]:
                best = (diff, ep, stage, steps)

    if best is None:
        return None, f"no checkpoint-episode rows in {csv_path}"

    _, ep, stage, steps = best
    fname = f"ppo_ep{ep:04d}_s{stage}"
    zip_path = os.path.join(ckpt_dir, fname + ".zip")
    vec_path = os.path.join(ckpt_dir, fname + "_vecnorm.pkl")
    exists = " (on disk)" if os.path.isfile(zip_path) else " (MISSING on disk!)"
    return (
        f"  ep={ep}  stage={stage}  steps={steps:,}  (diff={best[0]:,})\n"
        f"    zip    : {zip_path}{exists}\n"
        f"    vecnorm: {vec_path}",
        None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=20_000_000,
                    help="target step count (default 20,000,000)")
    ap.add_argument("dirs", nargs="*", default=[
        "train_scripts/checkpoints_run1_ont_food_prox5",
        "train_scripts/checkpoints_run2_vanilla_food",
    ], help="checkpoint dirs to scan")
    args = ap.parse_args()

    print(f"Target: {args.target:,} steps\n")
    rc = 0
    for d in args.dirs:
        print(f"{d}:")
        result, err = nearest_checkpoint(d, args.target)
        if err:
            print(f"  ! {err}")
            rc = 1
        else:
            print(result)
        print()
    sys.exit(rc)


if __name__ == "__main__":
    main()
