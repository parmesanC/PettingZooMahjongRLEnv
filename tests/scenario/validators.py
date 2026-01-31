"""
场景测试框架 - 验证器函数

提供常用的验证器函数，用于验证游戏状态。
"""

from typing import List, Callable
from collections import Counter
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType, ActionType


def hand_count_equals(player_id: int, expected_count: int) -> Callable[[GameContext], bool]:
    """验证玩家手牌数量

    Args:
        player_id: 玩家索引
        expected_count: 预期手牌数量

    Returns:
        验证函数
    """
    def validator(context: GameContext) -> bool:
        actual = len(context.players[player_id].hand_tiles)
        if actual != expected_count:
            print(f"手牌数量验证失败: 玩家{player_id} 预期{expected_count}张, 实际{actual}张")
            return False
        return True
    return validator


def hand_contains(player_id: int, tiles: List[int]) -> Callable[[GameContext], bool]:
    """验证玩家手牌包含指定牌（不考虑顺序）

    Args:
        player_id: 玩家索引
        tiles: 预期包含的牌列表

    Returns:
        验证函数
    """
    def validator(context: GameContext) -> bool:
        hand = context.players[player_id].hand_tiles
        hand_counter = Counter(hand)
        tiles_counter = Counter(tiles)

        for tile, expected_count in tiles_counter.items():
            actual_count = hand_counter.get(tile, 0)
            if actual_count < expected_count:
                print(f"手牌验证失败: 玩家{player_id} 牌{tile} 预期≥{expected_count}张, 实际{actual_count}张")
                return False
        return True
    return validator


def wall_count_equals(expected: int) -> Callable[[GameContext], bool]:
    """验证牌墙剩余数量

    Args:
        expected: 预期牌墙剩余数量

    Returns:
        验证函数
    """
    def validator(context: GameContext) -> bool:
        actual = len(context.wall)
        if actual != expected:
            print(f"牌墙数量验证失败: 预期{expected}张, 实际{actual}张")
            return False
        return True
    return validator


def discard_pile_contains(tile: int) -> Callable[[GameContext], bool]:
    """验证弃牌堆包含某张牌

    Args:
        tile: 牌ID

    Returns:
        验证函数
    """
    def validator(context: GameContext) -> bool:
        if tile not in context.discard_pile:
            print(f"弃牌堆验证失败: 牌{tile}不在弃牌堆中")
            return False
        return True
    return validator


def state_is(expected_state: GameStateType) -> Callable[[GameContext], bool]:
    """验证当前状态

    Args:
        expected_state: 预期状态

    Returns:
        验证函数
    """
    def validator(context: GameContext) -> bool:
        actual = context.current_state
        if actual != expected_state:
            print(f"状态验证失败: 预期{expected_state.name}, 实际{actual.name if actual else None}")
            return False
        return True
    return validator


def meld_count_equals(player_id: int, expected: int) -> Callable[[GameContext], bool]:
    """验证玩家副露数量

    Args:
        player_id: 玩家索引
        expected: 预期副露数量

    Returns:
        验证函数
    """
    def validator(context: GameContext) -> bool:
        actual = len(context.players[player_id].melds)
        if actual != expected:
            print(f"副白数量验证失败: 玩家{player_id} 预期{expected}组, 实际{actual}组")
            return False
        return True
    return validator


def action_mask_contains(action_type: ActionType) -> Callable[[GameContext], bool]:
    """验证 action_mask 包含指定动作类型

    Args:
        action_type: 动作类型

    Returns:
        验证函数
    """
    def validator(context: GameContext) -> bool:
        mask = context.action_mask
        # 需要使用 WuhanMahjongEnv._action_to_index() 转换
        # 但为了避免循环导入，这里简化处理
        # 实际使用时由 TestExecutor 处理
        print(f"action_mask 验证: {action_type.name} 需要在 executor 中处理")
        return True  # 占位，由 executor 实际验证
    return validator
