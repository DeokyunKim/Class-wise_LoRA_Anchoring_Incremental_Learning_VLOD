"""Unit tests: CLA-selective L2-anchor snapshot (``skip_warmup`` + ``skip_new_per_class``).

Verifies ``TaskMemory.l2_anchor_snapshot(skip_warmup, skip_new_per_class)``:
  - ``skip_warmup``        -> warmup_lora_a excluded
  - ``skip_new_per_class`` -> per-class A of counter==0 (NEW) classes excluded;
                              counter>0 (existing) classes kept
  - shared_lora_b kept under the ``a_and_b`` target
"""
import torch
import torch.nn as nn

from groundingdino.util.task_memory import TaskMemory


class Dummy(nn.Module):
    """Module whose parameter names mimic the LoRA pool (per-class A + shared B)."""

    def __init__(self) -> None:
        super().__init__()
        self.warmup_lora_a = nn.Parameter(torch.randn(2, 4))
        self.per_class_lora_A = nn.ParameterDict({
            "class_dog": nn.Parameter(torch.randn(2, 4)),   # existing (counter > 0)
            "class_cat": nn.Parameter(torch.randn(2, 4)),   # NEW (counter == 0)
        })
        self.shared_lora_b = nn.Parameter(torch.randn(4, 2))


def _has(keys, substring: str) -> bool:
    return any(substring in k for k in keys)


def test_full_anchor_includes_everything() -> None:
    tm = TaskMemory()
    tm._class_counter = {"dog": 3, "cat": 0}    # dog seen before, cat brand-new
    tm.l2_anchor_snapshot(Dummy(), target="a_and_b")
    full = set(tm._l2_anchor_dict)
    assert _has(full, "warmup_lora_a")
    assert _has(full, "class_dog") and _has(full, "class_cat")
    assert _has(full, "shared_lora_b")


def test_selective_skips_warmup_and_new_per_class() -> None:
    tm = TaskMemory()
    tm._class_counter = {"dog": 3, "cat": 0}
    tm.l2_anchor_snapshot(Dummy(), target="a_and_b",
                          skip_warmup=True, skip_new_per_class=True)
    sel = set(tm._l2_anchor_dict)
    assert not _has(sel, "warmup_lora_a")        # warm-up excluded
    assert not _has(sel, "class_cat")            # NEW class (counter==0) excluded
    assert _has(sel, "class_dog")                # existing class (counter>0) kept
    assert _has(sel, "shared_lora_b")            # shared B kept (a_and_b)


def test_skip_warmup_only_keeps_all_per_class() -> None:
    tm = TaskMemory()
    tm._class_counter = {"dog": 3, "cat": 0}
    tm.l2_anchor_snapshot(Dummy(), target="a_and_b",
                          skip_warmup=True, skip_new_per_class=False)
    w_only = set(tm._l2_anchor_dict)
    assert not _has(w_only, "warmup_lora_a")
    assert _has(w_only, "class_cat") and _has(w_only, "class_dog")
