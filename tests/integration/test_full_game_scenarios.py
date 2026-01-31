"""完整游戏流程场景测试

使用场景测试框架测试完整的对局流程。
"""

import pytest
from tests.scenario.builder import ScenarioBuilder
from src.mahjong_rl.core.constants import GameStateType, ActionType


def test_full_game_simple():
    """测试简单对局流程"""
    wall = []
    for tile_id in range(34):
        wall.extend([tile_id] * 4)

    result = (
        ScenarioBuilder("简单对局流程")
        .description("测试从发牌到游戏结束的基本流程")
        .with_wall(wall)
        .run()
    )

    assert result is not None
    assert result.scenario_name == "简单对局流程"
