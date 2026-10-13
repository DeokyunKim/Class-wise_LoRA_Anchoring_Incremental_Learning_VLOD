#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# Class-wise LoRA Anchoring (CLA) — incremental open-vocabulary training.
#
# Trains the per-class LoRA inventory sequentially over the benchmark's tasks
# with the per-class L2 anchor enabled, then runs zero-shot COCO + per-dataset
# evaluation (handled inside main.py).
#
# Usage:
#   bash train.sh <odinw13|odinwo> [cla|dithub] [seed] [gpu] [port]
#
#   benchmark : odinw13 (13 disjoint tasks) | odinwo (6 overlapping-class tasks)
#   method    : cla    -> per-class L2 anchor ON  (default; headline method)
#               dithub -> anchor OFF (gamma_L2=0) -> reproduces the DitHub baseline
#   seed      : task-order seed (paper uses 1, 2, 3 and reports mean +/- std)
#
# The Python interpreter is taken from $PYTHON (default: `python`); activate
# your environment first, or run e.g. `PYTHON=/path/to/venv/bin/python bash train.sh ...`.
# ─────────────────────────────────────────────────────────────────────────
set -e

BENCH="${1:-odinw13}"
METHOD="${2:-cla}"
SEED="${3:-1}"
GPU="${4:-0}"
PORT="${5:-29500}"
PYTHON="${PYTHON:-python}"

# DitHub module-merge coefficients (lambda_a / lambda_b). Both benchmarks use the
# DitHub defaults 0.3 / 0.7.
case "$BENCH" in
  odinw13) CFG="configs/test_odinw13"; GAMMA_L2=1.0; LAMBDA_A=0.3; LAMBDA_B=0.7 ;;   # disjoint classes -> strong anchor
  odinwo)  CFG="configs/test_odinwo";  GAMMA_L2=0.3; LAMBDA_A=0.3; LAMBDA_B=0.7 ;;    # overlapping classes -> weak anchor
  *) echo "unknown benchmark: '$BENCH' (use odinw13 | odinwo)"; exit 1 ;;
esac

if [ "$METHOD" = "dithub" ]; then
  GAMMA_L2=0.0                                         # baseline: no anchor
elif [ "$METHOD" != "cla" ]; then
  echo "unknown method: '$METHOD' (use cla | dithub)"; exit 1
fi

OUT="output/${BENCH}_${METHOD}_seed${SEED}"
mkdir -p "$OUT"
echo "[CLA] bench=$BENCH method=$METHOD seed=$SEED gpu=$GPU gamma_L2=$GAMMA_L2 lambda_a=$LAMBDA_A lambda_b=$LAMBDA_B -> $OUT"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" main.py \
  --config-file "$CFG" \
  --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_dithub.py \
  --model-checkpoint-path groundingdino_swint_ogc.pth \
  --num-gpus 1 --seed "$SEED" --output-dir "$OUT" --shuffle-tasks \
  --dist-url "tcp://127.0.0.1:${PORT}" \
  --dithub --lora-r 16 --lora-alpha 8 --lora-dropout 0.0 \
  --lora-out-min 128 --lora-lr 0.001 --lora-weight-decay 0.01 \
  --lambda-a "$LAMBDA_A" --lambda-b "$LAMBDA_B" \
  --l2-anchor-weight "$GAMMA_L2" --l2-anchor-target a_and_b \
  2>&1 | tee "${OUT}_run.log"
