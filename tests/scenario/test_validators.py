"""测试验证器函数"""

import pytest
from tests.scenario.validators import (
    hand_count_equals,
    hand_contains,
    wall_count_equals,
    state_is,
    discard_pile_contains,
    meld_count_equals,
)
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType


def test_hand_count_equals():
    """测试手牌数量验证器"""
    context = GameContext()
    context.players[0].hand_tiles = [0, 1, 2, 3, 4]

    validator = hand_count_equals(0, 5)
    assert validator(context) is True

    validator_fail = hand_count_equals(0, 3)
    assert validator_fail(context) is False


def test_hand_contains():
    """测试手牌包含验证器"""
    context = GameContext()
    context.players[0].hand_tiles = [0, 1, 2, 2, 3]

    validator = hand_contains(0, [0, 2, 2])
    assert validator(context) is True

    validator_fail = hand_contains(0, [0, 5])
    assert validator_fail(context) is False


def test_wall_count_equals():
    """测试牌墙数量验证器"""
    context = GameContext()
    context.wall = [0] * 100

    validator = wall_count_equals(100)
    assert validator(context) is True

    validator_fail = wall_count_equals(50)
    assert validator_fail(context) is False


def test_state_is():
    """测试状态验证器"""
    context = GameContext()
    context.current_state = GameStateType.PLAYER_DECISION

    validator = state_is(GameStateType.PLAYER_DECISION)
    assert validator(context) is True

    validator_fail = state_is(GameStateType.WAITING_RESPONSE)
    assert validator_fail(context) is False


def test_discard_pile_contains():
    """测试弃牌堆包含验证器"""
    context = GameContext()
    context.discard_pile = [0, 1, 2, 3, 4]

    validator = discard_pile_contains(2)
    assert validator(context) is True

    validator_fail = discard_pile_contains(99)
    assert validator_fail(context) is False


def test_meld_count_equals():
    """测试副露数量验证器"""
    from src.mahjong_rl.core.PlayerData import Meld
    from src.mahjong_rl.core.mahjong_action import MahjongAction
    from src.mahjong_rl.core.constants import ActionType

    context = GameContext()
    # 创建两个模拟副露
    context.players[0].melds = [
        Meld(action_type=MahjongAction(ActionType.PONG, 4), tiles=[4, 4, 4], from_player=1),
        Meld(action_type=MahjongAction(ActionType.PONG, 5), tiles=[5, 5, 5], from_player=2),
    ]

    validator = meld_count_equals(0, 2)
    assert validator(context) is True

    validator_fail = meld_count_equals(0, 1)
    assert validator_fail(context) is False
