"""验证 Wuhan7P4LObservationBuilder 实现是否正确"""

import numpy as np
from collections import deque

from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder
from src.mahjong_rl.core.GameData import GameContext, ActionRecord
from src.mahjong_rl.core.PlayerData import PlayerData, Meld
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.core.constants import ActionType, GameStateType


def create_mock_context():
    """创建一个模拟的游戏上下文"""
    context = GameContext()

    context.current_state = GameStateType.PLAYER_DECISION
    context.current_player_idx = 0
    context.dealer_idx = 0
    context.last_discarded_tile = 5
    context.discard_player = 1
    context.discard_pile = [0, 1, 2, 3, 4]
    context.wall = deque([i % 34 for i in range(100)])
    context.lazy_tile = 1
    context.skin_tile = [2, 3]
    context.red_dragon = 31
    context.training_phase = 3
    context.training_progress = 1.0

    context.players[0].hand_tiles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    context.players[1].hand_tiles = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    context.players[2].hand_tiles = [26, 27, 28, 29, 30, 31, 32, 33, 0, 1, 2, 3, 4]
    context.players[3].hand_tiles = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    meld1 = Meld(action_type=MahjongAction(ActionType.PONG, 5), tiles=[5, 5, 5], from_player=1)
    context.players[0].melds = [meld1]

    action1 = ActionRecord(action_type=MahjongAction(ActionType.DISCARD, 5), tile=5, player_id=0)
    action2 = ActionRecord(action_type=MahjongAction(ActionType.PONG, 6), tile=6, player_id=1)
    context.action_history = [action1, action2]

    context.players[0].special_gangs = [1, 2, 0]
    context.players[1].special_gangs = [0, 1, 1]
    context.players[2].special_gangs = [2, 0, 0]
    context.players[3].special_gangs = [0, 0, 1]

    context.players[0].fan_count = 16
    context.players[1].fan_count = 24
    context.players[2].fan_count = 8
    context.players[3].fan_count = 12

    return context


def test_observation_dimensions():
    """测试观测维度是否正确"""
    print("=" * 60)
    print("测试观测维度")
    print("=" * 60)

    builder = Wuhan7P4LObservationBuilder()
    context = create_mock_context()

    for player_id in range(4):
        observation = builder.build(player_id, context)

        print(f"\n玩家 {player_id} 的观测维度:")
        print(f"  global_hand: {observation['global_hand'].shape} (期望: (136,))")
        assert observation['global_hand'].shape == (136,), f"global_hand 维度错误"

        print(f"  private_hand: {observation['private_hand'].shape} (期望: (34,))")
        assert observation['private_hand'].shape == (34,), f"private_hand 维度错误"

        print(f"  discard_pool_total: {observation['discard_pool_total'].shape} (期望: (34,))")
        assert observation['discard_pool_total'].shape == (34,), f"discard_pool_total 维度错误"

        print(f"  wall: {observation['wall'].shape} (期望: (82,))")
        assert observation['wall'].shape == (82,), f"wall 维度错误"

        print(f"  melds.action_types: {observation['melds']['action_types'].shape} (期望: (16,))")
        assert observation['melds']['action_types'].shape == (16,), f"melds.action_types 维度错误"

        print(f"  melds.tiles: {observation['melds']['tiles'].shape} (期望: (256,))")
        assert observation['melds']['tiles'].shape == (256,), f"melds.tiles 维度错误"

        print(f"  melds.group_indices: {observation['melds']['group_indices'].shape} (期望: (32,))")
        assert observation['melds']['group_indices'].shape == (32,), f"melds.group_indices 维度错误"

        print(f"  action_history.types: {observation['action_history']['types'].shape} (期望: (80,))")
        assert observation['action_history']['types'].shape == (80,), f"action_history.types 维度错误"

        print(f"  action_history.params: {observation['action_history']['params'].shape} (期望: (80,))")
        assert observation['action_history']['params'].shape == (80,), f"action_history.params 维度错误"

        print(f"  action_history.players: {observation['action_history']['players'].shape} (期望: (80,))")
        assert observation['action_history']['players'].shape == (80,), f"action_history.players 维度错误"

        print(f"  special_gangs: {observation['special_gangs'].shape} (期望: (12,))")
        assert observation['special_gangs'].shape == (12,), f"special_gangs 维度错误"

        print(f"  current_player: {observation['current_player'].shape} (期望: (1,))")
        assert observation['current_player'].shape == (1,), f"current_player 维度错误"

        print(f"  fan_counts: {observation['fan_counts'].shape} (期望: (4,))")
        assert observation['fan_counts'].shape == (4,), f"fan_counts 维度错误"

        print(f"  special_indicators: {observation['special_indicators'].shape} (期望: (2,))")
        assert observation['special_indicators'].shape == (2,), f"special_indicators 维度错误"

        print(f"  dealer: {observation['dealer'].shape} (期望: (1,))")
        assert observation['dealer'].shape == (1,), f"dealer 维度错误"

        print(f"  current_phase: {type(observation['current_phase'])} (期望: int)")
        assert isinstance(observation['current_phase'], int), f"current_phase 类型错误"

    print("\n✓ 所有维度测试通过!")


def test_action_mask():
    """测试动作掩码"""
    print("\n" + "=" * 60)
    print("测试动作掩码")
    print("=" * 60)

    builder = Wuhan7P4LObservationBuilder()
    context = create_mock_context()

    for player_id in range(4):
        action_mask = builder.build_action_mask(player_id, context)

        print(f"\n玩家 {player_id} 的动作掩码:")
        print(f"  types: {action_mask['types'].shape} (期望: (11,))")
        assert action_mask['types'].shape == (11,), f"types 维度错误"

        print(f"  params: {action_mask['params'].shape} (期望: (35,))")
        assert action_mask['params'].shape == (35,), f"params 维度错误"

        print(f"  types 中可执行动作数量: {np.sum(action_mask['types'])}")

    print("\n✓ 动作掩码测试通过!")


def test_visibility_masking():
    """测试可见性掩码"""
    print("\n" + "=" * 60)
    print("测试可见性掩码")
    print("=" * 60)

    builder = Wuhan7P4LObservationBuilder()
    context = create_mock_context()

    print("\n训练阶段 1 (完全可见):")
    context.training_phase = 1
    obs1 = builder.build(0, context)
    opponent_hand_visible = np.sum(obs1['global_hand'][34:68])
    print(f"  对手手牌可见度: {opponent_hand_visible} (期望: > 0)")
    wall_visible = np.sum(obs1['wall'] != 34)
    print(f"  牌墙可见度: {wall_visible} (期望: > 0)")

    print("\n训练阶段 3 (隐藏信息):")
    context.training_phase = 3
    obs3 = builder.build(0, context)
    opponent_hand_visible = np.sum(obs3['global_hand'][34:68])
    print(f"  对手手牌可见度: {opponent_hand_visible} (期望: 0)")
    assert opponent_hand_visible == 0, f"对手手牌应该被隐藏"
    wall_visible = np.sum(obs3['wall'] != 34)
    print(f"  牌墙可见度: {wall_visible} (期望: 0)")
    assert wall_visible == 0, f"牌墙应该被隐藏"

    print("\n✓ 可见性掩码测试通过!")


def run_all_tests():
    """运行所有测试"""
    try:
        test_observation_dimensions()
        test_action_mask()
        test_visibility_masking()

        print("\n" + "=" * 60)
        print("所有测试通过! ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
