"""
测试自动跳过状态模式

验证当所有玩家都只能 PASS 时，WAITING_RESPONSE 状态能够自动跳过
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import GameStateType, Tiles
from src.mahjong_rl.state_machine.machine import MahjongStateMachine
from src.mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder


def test_wait_response_auto_skip_when_all_pass():
    """
    测试场景：所有玩家都只能 PASS

    预期行为：
    1. 进入 WAITING_RESPONSE 状态
    2. should_auto_skip() 返回 True
    3. 状态自动转换到 DRAWING
    4. agent_selection 正确更新
    """
    # 创建测试环境
    context = GameContext()
    context.players = [PlayerData(player_id=i) for i in range(4)]
    context.current_player_idx = 0

    # 设置弃牌（一张特殊牌，使得无人能吃碰杠胡）
    context.last_discarded_tile = Tiles.RED_DRAGON.value  # 红中
    context.discard_player = 0
    context.setup_response_order(0)

    # 创建状态机
    rule_engine = Wuhan7P4LRuleEngine(context)
    observation_builder = Wuhan7P4LObservationBuilder(context)
    state_machine = MahjongStateMachine(
        rule_engine=rule_engine,
        observation_builder=observation_builder,
        logger=None,
        enable_logging=False
    )
    state_machine.set_context(context)

    # 转换到 WAITING_RESPONSE 状态
    state_machine.transition_to(GameStateType.WAITING_RESPONSE, context)

    # 验证状态已自动跳过
    # 由于所有玩家都只能 PASS（红中不能吃碰杠），
    # 状态应该自动转换到 DRAWING
    assert state_machine.current_state_type == GameStateType.DRAWING, \
        f"Expected DRAWING state, got {state_machine.current_state_type}"

    # 验证 agent_selection 已更新
    assert context.current_player_idx == 1, \
        f"Expected player_1, got player_{context.current_player_idx}"


def test_wait_response_no_skip_when_has_responders():
    """
    测试场景：有玩家可以响应（非 PASS）

    预期行为：
    1. 进入 WAITING_RESPONSE 状态
    2. should_auto_skip() 返回 False
    3. 状态保持在 WAITING_RESPONSE
    4. active_responders 不为空
    """
    # 创建测试环境
    context = GameContext()
    context.players = [PlayerData(player_id=i) for i in range(4)]

    # 给玩家1一张可以碰的牌
    context.players[1].hand_tiles = [Tiles.ONE_CHAR.value, Tiles.ONE_CHAR.value, Tiles.ONE_CHAR.value]
    context.current_player_idx = 0

    # 设置弃牌（一张万子牌）
    context.last_discarded_tile = Tiles.ONE_CHAR.value
    context.discard_player = 0
    context.setup_response_order(0)

    # 创建状态机
    rule_engine = Wuhan7P4LRuleEngine(context)
    observation_builder = Wuhan7P4LObservationBuilder(context)
    state_machine = MahjongStateMachine(
        rule_engine=rule_engine,
        observation_builder=observation_builder,
        logger=None,
        enable_logging=False
    )
    state_machine.set_context(context)

    # 转换到 WAITING_RESPONSE 状态
    state_machine.transition_to(GameStateType.WAITING_RESPONSE, context)

    # 验证状态没有自动跳过
    assert state_machine.current_state_type == GameStateType.WAITING_RESPONSE, \
        f"Expected WAITING_RESPONSE state, got {state_machine.current_state_type}"

    # 验证 active_responders 不为空
    assert len(context.active_responders) > 0, \
        "Expected active_responders to be non-empty"

    # 验证当前玩家是第一个响应者
    assert context.current_player_idx in context.active_responders, \
        f"Expected current_player to be in active_responders"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
