"""
场景测试框架

提供流式测试构建器，用于按预定牌墙顺序和出牌节奏测试状态机。
"""

from tests.scenario.context import ScenarioContext, StepConfig, TestResult
from tests.scenario.builder import ScenarioBuilder
from tests.scenario.executor import TestExecutor
from tests.scenario.validators import (
    hand_count_equals,
    hand_contains,
    wall_count_equals,
    discard_pile_contains,
    state_is,
    meld_count_equals,
    action_mask_contains,
)

__all__ = [
    "ScenarioContext",
    "StepConfig",
    "TestResult",
    "ScenarioBuilder",
    "TestExecutor",
    "hand_count_equals",
    "hand_contains",
    "wall_count_equals",
    "discard_pile_contains",
    "state_is",
    "meld_count_equals",
    "action_mask_contains",
]
