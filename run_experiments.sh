#!/usr/bin/env bash
# Train all thesis run variants back-to-back, each into its own checkpoint dir.
#
# Maps to RESEARCH_PLAN.md:
#   run1_ont_food_prox5   SQ1 ontology arm   (the "primary" run)
#   run2_vanilla_food     SQ1 vanilla baseline   (use_ontology=False)
#   run3_ont_food_prox0   SQ3 independent ABox   (proximity=0)
#   run4_ont_trash_prox5  SQ5 generalisation     (collect_trash)
#
# Each run writes to:  train_scripts/checkpoints_<RUN_NAME>/
#   - training_log.csv        (full per-episode metrics for plots)
#   - ppo_ep*_s*.zip          (periodic checkpoints + _vecnorm.pkl)
#   - ppo_final.zip           (final model)
# A combined console log per run goes to:  logs/<RUN_NAME>.log
#
# Usage:
#   ONTOLOGY_PATH=~/Thesis/thesisont_updated-2.owl N_ENVS=$(nproc) bash run_experiments.sh
#
# Run a subset:  RUNS="run2 run3" bash run_experiments.sh
# Resume the whole sweep after a crash: it SKIPS any run that already has a
# ppo_final.zip, so just re-invoke the same command.

set -uo pipefail

cd "$(dirname "$0")"

# ── shared settings (identical across all runs → fair comparison) ─────────────
export ONTOLOGY_PATH="${ONTOLOGY_PATH:-$HOME/Thesis/thesisont_updated-2.owl}"
export N_ENVS="${N_ENVS:-$(nproc 2>/dev/null || echo 1)}"
export SEED="${SEED:-0}"
export TOTAL_STEPS="${TOTAL_STEPS:-45000000}"
export ENT_START="${ENT_START:-0.06}"
export ENT_END="${ENT_END:-0.065}"

PY="${PYTHON:-python}"
mkdir -p logs

# ── run table: name | GOAL_NAME | USE_ONTOLOGY | PROXIMITY ───────────────────
# (edit / comment out lines to change what trains. run1 is included for a clean
#  same-seed sweep; comment it if you keep your existing checkpoints_ppo run.)
declare -A RUN_GOAL RUN_ONT RUN_PROX
RUN_ORDER=(run1 run2 run3 run4)

RUN_GOAL[run1]="collect_food";  RUN_ONT[run1]="1"; RUN_PROX[run1]="5"
RUN_GOAL[run2]="collect_food";  RUN_ONT[run2]="0"; RUN_PROX[run2]="5"
RUN_GOAL[run3]="collect_food";  RUN_ONT[run3]="1"; RUN_PROX[run3]="0"
RUN_GOAL[run4]="collect_trash"; RUN_ONT[run4]="1"; RUN_PROX[run4]="5"

declare -A RUN_LABEL
RUN_LABEL[run1]="ont_food_prox5"
RUN_LABEL[run2]="vanilla_food"
RUN_LABEL[run3]="ont_food_prox0"
RUN_LABEL[run4]="ont_trash_prox5"

# Allow RUNS="run2 run3" to select a subset.
SELECTED=(${RUNS:-${RUN_ORDER[@]}})

echo "=================================================================="
echo " Experiment sweep"
echo "   ontology : $ONTOLOGY_PATH"
echo "   n_envs   : $N_ENVS   seed: $SEED   total_steps: $TOTAL_STEPS"
echo "   entropy  : $ENT_START -> $ENT_END"
echo "   runs     : ${SELECTED[*]}"
echo "=================================================================="

for r in "${SELECTED[@]}"; do
    name="${r}_${RUN_LABEL[$r]}"
    ckpt="train_scripts/checkpoints_${name}"

    if [ -f "${ckpt}/ppo_final.zip" ]; then
        echo ">>> SKIP ${name}  (already finished: ${ckpt}/ppo_final.zip)"
        continue
    fi

    echo ""
    echo ">>> START ${name}   goal=${RUN_GOAL[$r]} use_ontology=${RUN_ONT[$r]} proximity=${RUN_PROX[$r]}"
    echo ">>> $(date)   log -> logs/${name}.log"

    RUN_NAME="${name}" \
    GOAL_NAME="${RUN_GOAL[$r]}" \
    USE_ONTOLOGY="${RUN_ONT[$r]}" \
    PROXIMITY="${RUN_PROX[$r]}" \
    "$PY" train_scripts/train_ppo.py 2>&1 | tee "logs/${name}.log"

    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
        echo "!!! ${name} exited with status ${status} — stopping sweep." >&2
        exit "$status"
    fi
    echo ">>> DONE ${name}   $(date)"
done

echo ""
echo "All selected runs finished. Checkpoints in train_scripts/checkpoints_*/"
