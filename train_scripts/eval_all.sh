#!/usr/bin/env bash
# Evaluate the final checkpoint from every training run and save results to
# results/eval/.  Run this on the machine where the checkpoints live.
#
# Usage:
#   bash train_scripts/eval_all.sh
#   EPISODES=100 bash train_scripts/eval_all.sh          # more episodes
#   RUNS="run1 run2" bash train_scripts/eval_all.sh      # subset
#   RENDER=1 bash train_scripts/eval_all.sh              # watch one run
#
# Each run writes its eval output to:
#   results/eval/<run_name>.txt

set -uo pipefail
cd "$(dirname "$0")/.."   # repo root

EPISODES="${EPISODES:-50}"
STAGE="${STAGE:-2}"       # 0-indexed: 0=1item 1=2items 2=4items 3=all. Default 2=4items
                         # (matched reporting stage; agents plateau at the 8-item stage)
PY="${PYTHON:-python}"
ONTOLOGY_PATH="${ONTOLOGY_PATH:-$HOME/Thesis/thesisont_updated-2.owl}"
OUT_DIR="results/eval"
mkdir -p "$OUT_DIR"

RENDER_FLAG=""
if [ "${RENDER:-0}" = "1" ]; then RENDER_FLAG="--render"; fi

# Match the original result files, which were generated with sampled (stochastic)
# actions. STOCHASTIC=1 (default) samples; STOCHASTIC=0 uses deterministic argmax.
STOCH_FLAG=""
if [ "${STOCHASTIC:-1}" = "1" ]; then STOCH_FLAG="--stochastic"; fi

echo "======================================================"
echo " Evaluation sweep"
echo "   episodes   : $EPISODES"
echo "   stage index: $STAGE (0=1item, 1=2items, 2=4items, 3=all)"
echo "   actions    : $([ "${STOCHASTIC:-1}" = "1" ] && echo stochastic || echo deterministic)"
echo "   ontology   : $ONTOLOGY_PATH"
echo "   output     : $OUT_DIR/"
echo "======================================================"

# ── Run table: same structure as run_experiments.sh ───────────────────────────
declare -A RUN_LABEL RUN_GOAL RUN_ONT RUN_PROX RUN_CKPT
RUN_ORDER=(run1 run2 run3 run4)

RUN_LABEL[run1]="ont_food_prox5";  RUN_GOAL[run1]="collect_food";  RUN_ONT[run1]="1"; RUN_PROX[run1]="5"
RUN_LABEL[run2]="vanilla_food";    RUN_GOAL[run2]="collect_food";  RUN_ONT[run2]="0"; RUN_PROX[run2]="5"
RUN_LABEL[run3]="ont_food_prox0";  RUN_GOAL[run3]="collect_food";  RUN_ONT[run3]="1"; RUN_PROX[run3]="0"
RUN_LABEL[run4]="ont_trash_prox5"; RUN_GOAL[run4]="collect_trash"; RUN_ONT[run4]="1"; RUN_PROX[run4]="5"

# Canonical matched-20M checkpoint per run (run1/run2 over-trained past 20M, so
# pin their nearest-20M zip; run3/run4 trained fresh to 20M -> ppo_final.zip).
# Override per run with e.g. CKPT_run1=ppo_epNNNN_s3.zip
RUN_CKPT[run1]="${CKPT_run1:-ppo_ep6800_s3.zip}"
RUN_CKPT[run2]="${CKPT_run2:-ppo_ep7800_s3.zip}"
RUN_CKPT[run3]="${CKPT_run3:-ppo_final.zip}"
RUN_CKPT[run4]="${CKPT_run4:-ppo_final.zip}"

SELECTED=(${RUNS:-${RUN_ORDER[@]}})

for r in "${SELECTED[@]}"; do
    name="${r}_${RUN_LABEL[$r]}"
    ckpt_dir="train_scripts/checkpoints_${name}"
    ckpt_zip="${ckpt_dir}/${RUN_CKPT[$r]}"

    if [ ! -f "$ckpt_zip" ]; then
        echo ">>> SKIP $name  (missing ${RUN_CKPT[$r]} in $ckpt_dir)"
        continue
    fi

    out_file="${OUT_DIR}/${name}.txt"
    echo ""
    echo ">>> EVAL $name"
    echo "    checkpoint : $ckpt_zip"
    echo "    goal       : ${RUN_GOAL[$r]}  use_ontology=${RUN_ONT[$r]}  proximity=${RUN_PROX[$r]}"
    echo "    output     : $out_file"

    ONTOLOGY_PATH="$ONTOLOGY_PATH" \
    GOAL_NAME="${RUN_GOAL[$r]}" \
    USE_ONTOLOGY="${RUN_ONT[$r]}" \
    PROXIMITY="${RUN_PROX[$r]}" \
    "$PY" train_scripts/evaluate_ppo.py \
        --checkpoint "$ckpt_zip" \
        --stage "$STAGE" \
        --episodes "$EPISODES" \
        $STOCH_FLAG \
        $RENDER_FLAG \
    2>&1 | tee "$out_file"

    echo ">>> DONE $name"
done

echo ""
echo "All evaluations complete. Results in $OUT_DIR/"
