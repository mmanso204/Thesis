#!/usr/bin/env bash
# Record an .mp4 of the canonical checkpoint from every training run, using the
# same run table / checkpoints as eval_all.sh. Runs headless, so it works on the
# cloud VM where the checkpoints live.
#
# Requires a video backend (once):  pip install imageio imageio-ffmpeg
#
# Usage:
#   bash train_scripts/record_all.sh
#   STAGE=2 EPISODES=2 bash train_scripts/record_all.sh     # 4-item stage, 2 eps
#   RUNS="run1 run3" bash train_scripts/record_all.sh       # subset
#   STOCHASTIC=0 bash train_scripts/record_all.sh           # deterministic argmax
#
# Each run writes:  results/videos/<run_name>.mp4

set -uo pipefail
cd "$(dirname "$0")/.."   # repo root

EPISODES="${EPISODES:-1}"
STAGE="${STAGE:-3}"        # 0=1item 1=2items 2=4items 3=all. Default 3 = full task
FPS="${FPS:-12}"
PY="${PYTHON:-python}"
ONTOLOGY_PATH="${ONTOLOGY_PATH:-$HOME/Thesis/thesisont_updated-2.owl}"
OUT_DIR="results/videos"
mkdir -p "$OUT_DIR"

# Default to sampled actions, matching the eval result files.
STOCH_FLAG=""
if [ "${STOCHASTIC:-1}" = "1" ]; then STOCH_FLAG="--stochastic"; fi

echo "======================================================"
echo " Video recording sweep"
echo "   episodes   : $EPISODES"
echo "   stage index: $STAGE (0=1item, 1=2items, 2=4items, 3=all)"
echo "   actions    : $([ "${STOCHASTIC:-1}" = "1" ] && echo stochastic || echo deterministic)"
echo "   ontology   : $ONTOLOGY_PATH"
echo "   output     : $OUT_DIR/"
echo "======================================================"

# Run table + canonical checkpoints: identical to eval_all.sh.
declare -A RUN_LABEL RUN_GOAL RUN_ONT RUN_PROX RUN_CKPT
RUN_ORDER=(run1 run2 run3 run4)

RUN_LABEL[run1]="ont_food_prox5";  RUN_GOAL[run1]="collect_food";  RUN_ONT[run1]="1"; RUN_PROX[run1]="5"
RUN_LABEL[run2]="vanilla_food";    RUN_GOAL[run2]="collect_food";  RUN_ONT[run2]="0"; RUN_PROX[run2]="5"
RUN_LABEL[run3]="ont_food_prox0";  RUN_GOAL[run3]="collect_food";  RUN_ONT[run3]="1"; RUN_PROX[run3]="0"
RUN_LABEL[run4]="ont_trash_prox5"; RUN_GOAL[run4]="collect_trash"; RUN_ONT[run4]="1"; RUN_PROX[run4]="5"

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

    out_file="${OUT_DIR}/${name}.mp4"
    echo ""
    echo ">>> RECORD $name"
    echo "    checkpoint : $ckpt_zip"
    echo "    goal       : ${RUN_GOAL[$r]}  use_ontology=${RUN_ONT[$r]}  proximity=${RUN_PROX[$r]}"
    echo "    output     : $out_file"

    ONTOLOGY_PATH="$ONTOLOGY_PATH" \
    GOAL_NAME="${RUN_GOAL[$r]}" \
    USE_ONTOLOGY="${RUN_ONT[$r]}" \
    PROXIMITY="${RUN_PROX[$r]}" \
    "$PY" train_scripts/record_video.py \
        --checkpoint "$ckpt_zip" \
        --stage "$STAGE" \
        --episodes "$EPISODES" \
        --fps "$FPS" \
        $STOCH_FLAG \
        --out "$out_file"

    echo ">>> DONE $name"
done

echo ""
echo "All recordings complete. Videos in $OUT_DIR/"
