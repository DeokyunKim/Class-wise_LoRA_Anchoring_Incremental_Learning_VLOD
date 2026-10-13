"""Verify LoRA-rank wiring used by the CLA L2-anchor + inter-class repulsion:
per-class A has shape (r, in_features), shared B has (out_features, r),
and scaling == lora_alpha / r. Pre-registered α=r/2 keeps scaling=0.5 across ranks.
"""
import pytest
import torch

from groundingdino.models.GroundingDINO.modules.lora_pool import LinearPool
from groundingdino.util.task_memory import TaskMemory

IN_F, OUT_F, CLASSES = 256, 512, ("a", "b", "c")


def _fresh_pool(r, alpha):
    tm = TaskMemory()                       # singleton — reset to isolate the test
    tm._modules = {}
    tm._registered_instances = []
    tm._class_counter = {}
    tm.task_index = 0
    tm.current_task = None
    return LinearPool(in_features=IN_F, out_features=OUT_F, r=r, lora_alpha=alpha,
                      layer_name=f"L_r{r}_a{alpha}", classes=set(CLASSES))


@pytest.mark.parametrize("r,alpha", [(2, 1), (4, 2), (8, 4), (16, 8), (32, 16)])
def test_rank_shapes_and_scaling(r, alpha):
    pool = _fresh_pool(r, alpha)
    for c in CLASSES:
        A = pool.per_class_lora_A[f"class_{c}"]
        assert tuple(A.shape) == (r, IN_F), f"per-class A {tuple(A.shape)} != ({r},{IN_F})"
    assert tuple(pool.shared_lora_b.shape) == (OUT_F, r), "shared_lora_b rank mismatch"
    assert tuple(pool.warmup_lora_a.shape) == (r, IN_F), "warmup_lora_a rank mismatch"
    assert abs(pool.scaling - alpha / r) < 1e-9, f"scaling {pool.scaling} != {alpha/r}"


def test_constant_scaling_policy():
    # pre-registered: alpha = r/2 -> scaling 0.5 for every rank (isolates capacity from scaling)
    for r in (2, 4, 8, 16, 32):
        pool = _fresh_pool(r, r / 2)
        assert abs(pool.scaling - 0.5) < 1e-9, f"r={r}: scaling {pool.scaling} != 0.5"


def test_dithub_release_rank_is_8_not_16():
    # documents the repo(r16) vs released-ckpt(r8) discrepancy the sweep is built around
    pool16 = _fresh_pool(16, 8)
    pool8 = _fresh_pool(8, 4)
    assert pool16.per_class_lora_A["class_a"].shape[0] == 16
    assert pool8.per_class_lora_A["class_a"].shape[0] == 8
