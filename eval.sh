#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# Evaluate a trained CLA run (eval-only) from its saved `last_lora.pth`.
#
# Re-runs the full per-dataset + zero-shot COCO evaluation against the merged
# LoRA inventory stored in <output_dir>/last_lora.pth (written at train time).
#
# Usage:
#   bash eval.sh <odinw13|odinwo> <output_dir> [gpu] [port]
#
#   output_dir : a directory containing last_lora.pth (e.g. output/odinw13_cla_seed1)
#
# The Python interpreter is taken from $PYTHON (default: `python`).
# ─────────────────────────────────────────────────────────────────────────
set -e

BENCH="${1:-odinw13}"
OUT="${2:?usage: bash eval.sh <odinw13|odinwo> <output_dir> [gpu] [port]}"
GPU="${3:-0}"
PORT="${4:-29600}"
PYTHON="${PYTHON:-python}"

case "$BENCH" in
  odinw13) CFG="configs/test_odinw13" ;;
  odinwo)  CFG="configs/test_odinwo" ;;
  *) echo "unknown benchmark: '$BENCH' (use odinw13 | odinwo)"; exit 1 ;;
esac

if [ ! -f "$OUT/last_lora.pth" ]; then
  echo "ERROR: $OUT/last_lora.pth not found (train first, or point to a run dir)."; exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" main.py \
  --config-file "$CFG" \
  --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_dithub.py \
  --model-checkpoint-path groundingdino_swint_ogc.pth \
  --num-gpus 1 --output-dir "$OUT" --eval-only \
  --dist-url "tcp://127.0.0.1:${PORT}" \
  --dithub --lora-r 16 --lora-alpha 8 --lora-dropout 0.0 \
  --lora-out-min 128 --lora-lr 0.001 --lora-weight-decay 0.01 \
  2>&1 | tee "${OUT}_eval.log"
