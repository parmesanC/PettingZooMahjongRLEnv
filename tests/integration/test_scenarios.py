"""使用场景框架的集成测试"""

import pytest
from tests.scenario.builder import ScenarioBuilder
from src.mahjong_rl.core.constants import GameStateType, ActionType


def test_scenario_builder_basic():
    """测试基本的场景构建和执行"""
    result = (
        ScenarioBuilder("基本场景测试")
        .description("测试场景构建器的基本功能")
        .run()
    )

    # 空场景应该成功
    assert result.success is True


def test_scenario_with_wall():
    """测试带牌墙配置的场景"""
    wall = [0] * 136  # 简化：全部是1万

    result = (
        ScenarioBuilder("牌墙配置测试")
        .with_wall(wall)
        .run()
    )

    assert result.success is True

