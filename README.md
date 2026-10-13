<h1 align="center">Class-wise LoRA Anchoring (CLA)</h1>
<p align="center"><b>Anchoring the Per-Class LoRA Inventory for Incremental Open-Vocabulary Object Detection</b></p>
<p align="center">Deokyun Kim, Giyoung Lee, Yookyung Kim, Jihun Cha</p>
<p align="center"><i>NeurIPS 2026 Workshop (CL4FMagents)</i></p>

**CLA** adds a per-class **L2 anchor** (a `γ_L2` stability–plasticity dial) to the
DitHub / Grounding DINO per-class LoRA inventory, preserving prior-task knowledge during
incremental open-vocabulary detection. On **ODinW-13** it reaches **hAP 54.39** vs DitHub
**53.26** (3 seeds), driven by higher zero-shot COCO (48.28 vs 46.85). An optional inter-class
repulsion (`--repulsion-weight`) helps in the class-overlap regime. Full results and ablations
are in the paper.

Requires Python ≥ 3.9, CUDA ≥ 11.8, and an NVIDIA GPU.

## Setup

```bash
conda create -n cla python=3.10 pip && conda activate cla
pip install -r requirements.txt
pip install -r requirements-dev.txt          # optional: pytest for unit tests

# build the deformable-attention CUDA op
git clone https://github.com/fundamentalvision/Deformable-DETR.git
cd Deformable-DETR/models/ops && sh ./make.sh && cd -

# pre-trained Grounding DINO weights (training starting point)
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## Data

```bash
bash download_data.sh        # downloads ODinW-35 and builds ODinW-O into datasets/odinw13/
```

COCO 2017 (for zero-shot ZCOCO evaluation) is **not** fetched by the script — place or symlink
it at `datasets/coco` with the standard layout (`{train2017, val2017, annotations}`); download
from <https://cocodataset.org/#download>.

## Train

```bash
# args:  bash train.sh <odinw13|odinwo> [cla|dithub] [seed] [gpu]
bash train.sh odinw13 cla    1 0     # CLA (ours), ODinW-13  (γ_L2 = 1.0)
bash train.sh odinwo  cla    1 0     # CLA (ours), ODinW-O   (γ_L2 = 0.3)
bash train.sh odinw13 dithub 1 0     # DitHub baseline       (anchor off)
```

Each run trains the benchmark's tasks sequentially, then evaluates per-dataset AP and
zero-shot COCO into `output/<bench>_<method>_seed<seed>/`. Results are reported as the
mean ± std over task-order seeds **1, 2, 3**. The core knob is `--l2-anchor-weight` (γ_L2);
run `python main.py --help` for all flags.

## Eval

```bash
# re-evaluate a trained run from its saved last_lora.pth
bash eval.sh odinw13 output/odinw13_cla_seed1 0

# aggregate one or more runs -> Avg / ZCOCO / hAP (mean ± std across seeds)
python tools/aggregate_results.py --glob 'output/odinw13_cla_seed*'

# optional: unit tests
pytest tests/
```

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Vendored third-party code keeps its original license
headers (mostly Apache-2.0; Microsoft `loralib` is MIT; PyTorch is BSD-3-Clause).

## Acknowledgements

Built on [DitHub](https://github.com/aimagelab/DitHub) (Cappellino *et al.*, NeurIPS 2025),
and reuses code from [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO),
[Detectron2](https://github.com/facebookresearch/detectron2),
[Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR),
[DINO](https://github.com/IDEA-Research/DINO) / [detrex](https://github.com/IDEA-Research/detrex),
[DETR](https://github.com/facebookresearch/detr) / [MDETR](https://github.com/ashkamath/mdetr),
[LoRA](https://github.com/microsoft/LoRA),
[Swin Transformer](https://github.com/microsoft/Swin-Transformer),
[mmdetection](https://github.com/open-mmlab/mmdetection), and
[Mask2Former](https://github.com/facebookresearch/Mask2Former). We thank the authors for
releasing their code.
