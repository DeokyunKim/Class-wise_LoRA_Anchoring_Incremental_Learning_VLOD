#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# Dataset installation for CLA.
#
#   1. Downloads ODinW-35 (which contains the ODinW-13 subsets) into
#      datasets/odinw13/  via tools/download_odinw.py
#   2. Builds the ODinW-O (overlapping-class) annotations in place
#      via tools/overlapped_classes_dataset.py
#   3. Reminds you to provide COCO 2017 (used for zero-shot ZCOCO evaluation),
#      which is large and must be downloaded separately.
#
# Usage:   bash download_data.sh
# The Python interpreter is taken from $PYTHON (default: `python`).
# ─────────────────────────────────────────────────────────────────────────
set -e
PYTHON="${PYTHON:-python}"

mkdir -p datasets/odinw13

echo "[data] Downloading ODinW-35 into datasets/odinw13/ (this can take a while)..."
"$PYTHON" tools/download_odinw.py --dataset_path datasets/odinw13

echo "[data] Building ODinW-O (overlapping-class) annotations..."
"$PYTHON" tools/overlapped_classes_dataset.py

cat <<'EOF'

[data] ODinW-13 / ODinW-O are ready under datasets/odinw13/.

[data] COCO 2017 is required for zero-shot evaluation (ZCOCO) but is NOT fetched
       by this script (it is large). Place or symlink it at datasets/coco with the
       standard layout:
           datasets/coco/{train2017, val2017, annotations}
       Download: https://cocodataset.org/#download
EOF
