"""Shared pytest fixtures for the CLA unit tests.

``TaskMemory`` is a process-wide singleton, so its state must be reset between
tests to keep them isolated.
"""
import pytest

from groundingdino.util.task_memory import TaskMemory


@pytest.fixture(autouse=True)
def reset_task_memory():
    """Reset the TaskMemory singleton to a clean state before each test."""
    tm = TaskMemory()
    tm._l2_anchor_dict = {}
    tm._repulsion_anchor = {}
    tm._class_counter = {}
    tm._modules = {}
    tm._memory = {}
    tm._registered_instances = []
    tm._current_classes = []
    tm.l2_anchor_weight = 0.0
    tm.l2_anchor_target = "a_and_b"
    tm.repulsion_weight = 0.0
    tm.task_index = 0
    tm.is_warm_up = True
    tm.current_task = None
    yield
