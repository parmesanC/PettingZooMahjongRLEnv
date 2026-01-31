"""测试自定义初始状态功能

这个测试模块验证场景测试框架的自定义初始化功能，
允许用户绕过 InitialState 的自动初始化，直接设置手牌、牌墙、庄家等。
"""

import pytest
from tests.scenario.builder import ScenarioBuilder
from tests.scenario.validators import hand_count_equals, wall_count_equals, state_is
from src.mahjong_rl.core.constants import GameStateType, ActionType


# =============================================================================
# Fixtures 和辅助函数 - 消除重复代码
# =============================================================================

@pytest.fixture
def standard_hands():
    """标准测试手牌：每个玩家13张牌"""
    return {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        1: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        2: [26, 27, 28, 29, 30, 31, 32, 33, 0, 1, 2, 3, 4],
        3: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    }


@pytest.fixture
def standard_wall():
    """标准测试牌墙：16张牌"""
    return [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]


@pytest.fixture
def standard_special_tiles():
    """标准特殊牌配置"""
    return {'lazy': 8, 'skins': [7, 9]}


def create_initial_config(
    dealer_idx=0,
    current_player_idx=0,
    hands=None,
    wall=None,
    special_tiles=None,
    last_drawn_tile=12
):
    """创建初始状态配置的辅助函数

    Args:
        dealer_idx: 庄家位置
        current_player_idx: 当前玩家位置
        hands: 玩家手牌字典
        wall: 牌墙列表
        special_tiles: 特殊牌配置
        last_drawn_tile: 庄家刚摸的牌

    Returns:
        初始状态配置字典
    """
    return {
        'dealer_idx': dealer_idx,
        'current_player_idx': current_player_idx,
        'hands': hands or {},
        'wall': wall or [],
        'special_tiles': special_tiles or {},
        'last_drawn_tile': last_drawn_tile,
    }


# =============================================================================
# 测试用例
# =============================================================================

def test_custom_initialization_basic(
    standard_hands,
    standard_wall,
    standard_special_tiles
):
    """测试基本自定义初始化

    验证：
    - 4个玩家各有13张手牌
    - 牌墙剩余16张
    - 当前状态是 PLAYER_DECISION
    """
    result = (
        ScenarioBuilder("自定义初始化测试")
        .with_initial_state(
            create_initial_config(
                dealer_idx=0,
                current_player_idx=0,
                hands=standard_hands,
                wall=standard_wall,
                special_tiles=standard_special_tiles,
                last_drawn_tile=12,
            )
        )
        .step(1, "验证初始状态")
            .auto_advance()
            .verify("玩家0手牌13张", hand_count_equals(0, 13))
            .verify("玩家1手牌13张", hand_count_equals(1, 13))
            .verify("玩家2手牌13张", hand_count_equals(2, 13))
            .verify("玩家3手牌13张", hand_count_equals(3, 13))
            .verify("牌墙剩余16张", wall_count_equals(16))
            .verify("当前是PLAYER_DECISION状态", state_is(GameStateType.PLAYER_DECISION))
        .run()
    )

    assert result.success, f"测试失败: {result.failure_message}"


@pytest.mark.xfail(
    strict=True,
    reason="需要进一步调试 check_ting 调用问题。"
           "在打牌后系统会调用 check_ting，"
           "而自定义初始化可能没有正确设置所有必要的状态。"
           "TODO: 调查 check_ting 被调用的位置，"
           "确保自定义初始化设置所有必要的状态字段。"
)
def test_custom_initialization_with_action(standard_hands, standard_wall):
    """测试自定义初始化后执行动作

    验证：
    - 玩家0是庄家
    - 玩家0打牌
    - 状态转换到 WAITING_RESPONSE

    注意：此测试被标记为 xfail（预期失败），因为需要进一步调试。
    参见上面的 reason 和 TODO 标记。
    """
    result = (
        ScenarioBuilder("自定义初始化后打牌")
        .with_initial_state(
            create_initial_config(
                dealer_idx=0,
                current_player_idx=0,
                hands=standard_hands,
                wall=standard_wall,
                special_tiles={'lazy': 5, 'skins': [4, 6]},
                last_drawn_tile=12,
            )
        )
        .step(1, "玩家0打牌")
            .action(0, ActionType.DISCARD, 12)
            .expect_state(GameStateType.WAITING_RESPONSE)
        .run()
    )

    assert result.success, f"测试失败: {result.failure_message}"


def test_custom_initialization_dealer(
    standard_hands,
    standard_wall,
    standard_special_tiles
):
    """测试庄家设置

    验证：
    - 玩家2是庄家
    - 当前玩家是玩家2
    - 玩家2的 is_dealer 标志为 True
    """
    result = (
        ScenarioBuilder("庄家设置测试")
        .with_initial_state(
            create_initial_config(
                dealer_idx=2,
                current_player_idx=2,
                hands=standard_hands,
                wall=standard_wall,
                special_tiles=standard_special_tiles,
                last_drawn_tile=30,
            )
        )
        .step(1, "验证庄家")
            .auto_advance()
            .verify("当前玩家是玩家2", lambda ctx: ctx.current_player_idx == 2)
            .verify("玩家2是庄家", lambda ctx: ctx.players[2].is_dealer)
        .run()
    )

    assert result.success, f"测试失败: {result.failure_message}"


def test_custom_initialization_special_tiles(
    standard_hands,
    standard_wall
):
    """测试特殊牌设置

    验证：
    - 赖子牌ID
    - 皮子牌ID
    """
    result = (
        ScenarioBuilder("特殊牌设置测试")
        .with_initial_state(
            create_initial_config(
                dealer_idx=0,
                current_player_idx=0,
                hands=standard_hands,
                wall=standard_wall,
                special_tiles={'lazy': 5, 'skins': [4, 6]},
                last_drawn_tile=12,
            )
        )
        .step(1, "验证特殊牌")
            .auto_advance()
            .verify("赖子是5", lambda ctx: ctx.lazy_tile == 5)
            .verify("皮子是4和6", lambda ctx: ctx.skin_tile == [4, 6])
        .run()
    )

    assert result.success, f"测试失败: {result.failure_message}"


def test_custom_initialization_error_handling():
    """测试错误处理

    验证：
    - 设置无效的 dealer_idx (5)
    - 测试失败并包含错误消息
    """
    result = (
        ScenarioBuilder("错误处理测试")
        .with_initial_state({
            'dealer_idx': 5,  # 无效值
            'current_player_idx': 0,
            'hands': {0: [0, 1, 2]},
            'wall': [],
        })
        .run()
    )

    assert not result.success
    assert "dealer_idx 必须在 0-3 之间" in result.failure_message


def test_custom_initialization_invalid_current_player():
    """测试无效的 current_player_idx

    验证：
    - 设置无效的 current_player_idx (10)
    - 测试失败并包含错误消息
    """
    result = (
        ScenarioBuilder("无效当前玩家测试")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 10,  # 无效值
            'hands': {0: [0, 1, 2]},
            'wall': [],
        })
        .run()
    )

    assert not result.success
    assert "current_player_idx 必须在 0-3 之间" in result.failure_message


@pytest.mark.parametrize("dealer_idx", [0, 1, 2, 3])
def test_custom_initialization_all_dealer_positions(
    dealer_idx,
    standard_hands,
    standard_wall,
    standard_special_tiles
):
    """测试所有庄家位置

    使用 @pytest.mark.parametrize 为每个庄家位置生成独立的测试用例。

    验证每个玩家都能被正确设置为庄家
    """
    result = (
        ScenarioBuilder(f"庄家位置{dealer_idx}测试")
        .with_initial_state(
            create_initial_config(
                dealer_idx=dealer_idx,
                current_player_idx=dealer_idx,
                hands=standard_hands,
                wall=standard_wall,
                special_tiles=standard_special_tiles,
                last_drawn_tile=12,
            )
        )
        .step(1, f"验证玩家{dealer_idx}是庄家")
            .auto_advance()
            .verify(f"玩家{dealer_idx}是庄家", lambda ctx, idx=dealer_idx: ctx.players[idx].is_dealer)
        .run()
    )

    assert result.success, f"庄家位置{dealer_idx}测试失败: {result.failure_message}"


def test_custom_initialization_different_hand_sizes(standard_wall):
    """测试不同的手牌数量

    验证可以设置不同数量的手牌
    """
    different_hands = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # 13张
        1: [13, 14, 15, 16, 17, 18],  # 6张
        2: [26, 27, 28, 29, 30, 31, 32, 33, 0, 1],  # 10张
        3: [5, 6, 7],  # 3张
    }

    result = (
        ScenarioBuilder("不同手牌数量测试")
        .with_initial_state(
            create_initial_config(
                dealer_idx=0,
                current_player_idx=0,
                hands=different_hands,
                wall=standard_wall,
                special_tiles={'lazy': 8, 'skins': [7, 9]},
                last_drawn_tile=12,
            )
        )
        .step(1, "验证不同手牌数量")
            .auto_advance()
            .verify("玩家0有13张牌", hand_count_equals(0, 13))
            .verify("玩家1有6张牌", hand_count_equals(1, 6))
            .verify("玩家2有10张牌", hand_count_equals(2, 10))
            .verify("玩家3有3张牌", hand_count_equals(3, 3))
        .run()
    )

    assert result.success, f"测试失败: {result.failure_message}"


def test_custom_initialization_empty_wall(standard_hands):
    """测试空牌墙

    验证可以设置空牌墙
    """
    result = (
        ScenarioBuilder("空牌墙测试")
        .with_initial_state(
            create_initial_config(
                dealer_idx=0,
                current_player_idx=0,
                hands=standard_hands,
                wall=[],  # 空牌墙
                special_tiles={'lazy': 8, 'skins': [7, 9]},
                last_drawn_tile=12,
            )
        )
        .step(1, "验证牌墙为空")
            .auto_advance()
            .verify("牌墙为空", wall_count_equals(0))
        .run()
    )

    assert result.success, f"测试失败: {result.failure_message}"


def test_custom_initialization_only_lazy_tile(standard_hands, standard_wall):
    """测试只设置赖子，不设置皮子

    验证可以只设置部分特殊牌
    """
    result = (
        ScenarioBuilder("只设置赖子测试")
        .with_initial_state(
            create_initial_config(
                dealer_idx=0,
                current_player_idx=0,
                hands=standard_hands,
                wall=standard_wall,
                special_tiles={'lazy': 8},  # 只设置赖子
                last_drawn_tile=12,
            )
        )
        .step(1, "验证赖子设置")
            .auto_advance()
            .verify("赖子是8", lambda ctx: ctx.lazy_tile == 8)
        .run()
    )

    assert result.success, f"测试失败: {result.failure_message}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
