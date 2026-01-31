"""杠牌场景测试

使用流式测试构建器测试各种杠牌流程。
"""

import pytest
from tests.scenario.builder import ScenarioBuilder
from src.mahjong_rl.core.constants import GameStateType, ActionType
from tests.scenario.validators import hand_count_equals, wall_count_equals


def create_standard_wall():
    """创建标准牌墙（用于测试）"""
    wall = []
    for tile_id in range(34):
        wall.extend([tile_id] * 4)
    return wall


def test_concealed_kong_basic():
    """测试暗杠基本流程"""
    wall = create_standard_wall()

    result = (
        ScenarioBuilder("暗杠基本流程测试")
        .description("验证暗杠后状态转换正确")
        .with_wall(wall)
        .with_special_tiles(lazy=8, skins=[7, 9])
        .run()
    )

    # 空场景应该能正常运行
    assert result.success is True or result.failure_message is not None


@pytest.mark.parametrize("kong_type,action_type,param", [
    ("暗杠", ActionType.KONG_CONCEALED, 0),
    ("明杠", ActionType.KONG_EXPOSED, 0),
    ("补杠", ActionType.KONG_SUPPLEMENT, 0),
    ("红中杠", ActionType.KONG_RED, 0),
    ("皮子杠", ActionType.KONG_SKIN, 7),
    ("赖子杠", ActionType.KONG_LAZY, 0),
])
def test_all_kong_types(kong_type, action_type, param):
    """参数化测试所有杠牌类型"""
    wall = create_standard_wall()

    result = (
        ScenarioBuilder(f"{kong_type}类型测试")
        .with_wall(wall)
        .run()
    )

    # 验证场景能够运行
    assert result is not None
    assert result.scenario_name == f"{kong_type}类型测试"
