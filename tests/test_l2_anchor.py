"""Unit tests for the per-class L2 anchor (``TaskMemory.l2_anchor_*``).

Covers the snapshot target selection (``a_only`` vs ``a_and_b``), the
mean-over-tensors loss definition, snapshot detachment, and gradient flow.
"""
import torch
import torch.nn as nn

from groundingdino.util.task_memory import TaskMemory


class Toy(nn.Module):
    """Minimal module whose parameter names mimic the LoRA pool naming."""

    def __init__(self) -> None:
        super().__init__()
        self.per_class_lora_A_class_cat = nn.Parameter(torch.zeros(4, 8))  # 'lora_' in name
        self.warmup_lora_a = nn.Parameter(torch.zeros(4, 8))
        self.shared_lora_b = nn.Parameter(torch.zeros(6, 4))
        self.other = nn.Parameter(torch.zeros(3))                          # not a lora_ param

    def named_parameters(self, *args, **kwargs):  # noqa: D401 - mimic nn.Module signature
        return [
            ("per_class_lora_A.class_cat", self.per_class_lora_A_class_cat),
            ("warmup_lora_a", self.warmup_lora_a),
            ("shared_lora_b", self.shared_lora_b),
            ("other", self.other),
        ]

    def parameters(self, *args, **kwargs):
        return [p for _, p in self.named_parameters()]


def test_a_and_b_snapshot_includes_a_and_b_excludes_non_lora() -> None:
    tm = TaskMemory()
    n = tm.l2_anchor_snapshot(Toy(), target="a_and_b")
    assert n == 3  # per-class A + warmup A + shared B
    assert "shared_lora_b" in tm._l2_anchor_dict
    assert "other" not in tm._l2_anchor_dict


def test_a_only_excludes_shared_b() -> None:
    tm = TaskMemory()
    n = tm.l2_anchor_snapshot(Toy(), target="a_only")
    assert n == 2  # per-class A + warmup A only
    assert "shared_lora_b" not in tm._l2_anchor_dict


def test_loss_is_zero_at_snapshot() -> None:
    tm = TaskMemory()
    m = Toy()
    tm.l2_anchor_snapshot(m, target="a_and_b")
    assert float(tm.l2_anchor_loss(m)) == 0.0


def test_loss_is_mean_squared_deviation_over_tensors() -> None:
    tm = TaskMemory()
    m = Toy()
    tm.l2_anchor_snapshot(m, target="a_and_b")
    with torch.no_grad():
        m.shared_lora_b.add_(2.0)               # delta = 2 over 6*4 = 24 elements
    expected = (2.0 ** 2 * 24) / 3              # sum of squared deviations / n (= 3 tensors)
    assert abs(float(tm.l2_anchor_loss(m)) - expected) < 1e-5


def test_snapshot_is_detached_clone() -> None:
    tm = TaskMemory()
    m = Toy()
    tm.l2_anchor_snapshot(m, target="a_and_b")
    with torch.no_grad():
        m.warmup_lora_a.add_(5.0)               # mutating the live param must not touch the snapshot
    assert float(tm._l2_anchor_dict["warmup_lora_a"].abs().sum()) == 0.0


def test_grad_flows_to_params() -> None:
    tm = TaskMemory()
    m = Toy()
    tm.l2_anchor_snapshot(m, target="a_and_b")
    m.shared_lora_b.data.add_(1.0)
    loss = tm.l2_anchor_loss(m)
    loss.backward()
    assert m.shared_lora_b.grad is not None
    assert m.shared_lora_b.grad.abs().sum() > 0
