"""
测试 WaitRobKongState 自动跳过功能

验证当没有玩家可以抢杠和时，状态能够自动跳过。
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData, Meld
from src.mahjong_rl.core.constants import GameStateType, ActionType, Tiles
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.state_machine.machine import MahjongStateMachine
from src.mahjong_rl.state_machine.states.wait_rob_kong_state import WaitRobKongState
from src.mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder


class TestWaitRobKongAutoSkip:
    """测试 WaitRobKongState 自动跳过功能"""

    def test_should_auto_skip_when_no_responders(self):
        """测试没有玩家能抢杠时，should_auto_skip 返回 True"""
        # 创建测试环境
        context = GameContext()
        context.players = [PlayerData(player_id=i) for i in range(4)]
        context.current_player_idx = 0

        # 设置 active_responders 为空（没有人能抢杠和）
        context.active_responders = []
        context.response_order = []
        context.active_responder_idx = 0

        # 创建 WaitRobKongState 实例
        rule_engine = Wuhan7P4LRuleEngine(context)
        observation_builder = Wuhan7P4LObservationBuilder(context)
        state = WaitRobKongState(rule_engine, observation_builder)

        # 验证 should_auto_skip 返回 True
        result = state.should_auto_skip(context)
        assert result is True, \
            f"Expected should_auto_skip() to return True when no responders, got {result}"

    def test_should_not_auto_skip_when_has_responders(self):
        """测试有玩家能抢杠时，should_auto_skip 返回 False"""
        # 创建测试环境
        context = GameContext()
        context.players = [PlayerData(player_id=i) for i in range(4)]
        context.current_player_idx = 0

        # 设置 active_responders 有玩家（有人能抢杠和）
        context.active_responders = [1, 2]
        context.response_order = [1, 2]
        context.active_responder_idx = 0

        # 创建 WaitRobKongState 实例
        rule_engine = Wuhan7P4LRuleEngine(context)
        observation_builder = Wuhan7P4LObservationBuilder(context)
        state = WaitRobKongState(rule_engine, observation_builder)

        # 验证 should_auto_skip 返回 False
        result = state.should_auto_skip(context)
        assert result is False, \
            f"Expected should_auto_skip() to return False when has responders, got {result}"

    def test_step_with_auto_action_when_no_responders(self):
        """测试 step() 方法正确处理 'auto' 动作（没有响应者时）"""
        # 创建测试环境
        context = GameContext()
        context.players = [PlayerData(player_id=i) for i in range(4)]
        context.current_player_idx = 0
        context.kong_player_idx = 0
        context.rob_kong_tile = Tiles.ONE_CHAR.value

        # 设置 active_responders 为空
        context.active_responders = []
        context.response_order = []
        context.active_responder_idx = 0
        context.pending_responses = {}

        # 设置玩家0有一个碰牌（用于补杠）
        meld = Meld(
            action_type=MahjongAction(ActionType.PONG, Tiles.ONE_CHAR.value),
            tiles=[Tiles.ONE_CHAR.value] * 3,
            from_player=1
        )
        context.players[0].melds = [meld]
        context.players[0].hand_tiles = [Tiles.ONE_CHAR.value, Tiles.TWO_CHAR.value]

        # 创建 WaitRobKongState 实例
        rule_engine = Wuhan7P4LRuleEngine(context)
        observation_builder = Wuhan7P4LObservationBuilder(context)
        state = WaitRobKongState(rule_engine, observation_builder)

        # 执行 step(context, 'auto')
        next_state = state.step(context, 'auto')

        # 验证返回 DRAWING_AFTER_GONG
        assert next_state == GameStateType.DRAWING_AFTER_GONG, \
            f"Expected DRAWING_AFTER_GONG, got {next_state}"

        # 验证补杠已执行
        assert len(context.players[0].melds) == 1, \
            "Expected player_0 to have 1 meld after supplement kong"
        assert context.players[0].melds[0].action_type.action_type == ActionType.KONG_SUPPLEMENT, \
            "Expected meld to be KONG_SUPPLEMENT"
        assert len(context.players[0].melds[0].tiles) == 4, \
            "Expected supplement kong to have 4 tiles"

    def test_step_with_auto_action_raises_when_has_responders(self):
        """测试 step() 方法在有响应者时用 'auto' 调用会抛出异常"""
        # 创建测试环境
        context = GameContext()
        context.players = [PlayerData(player_id=i) for i in range(4)]
        context.current_player_idx = 1
        context.kong_player_idx = 0
        context.rob_kong_tile = Tiles.ONE_CHAR.value

        # 设置 active_responders 有玩家
        context.active_responders = [1, 2]
        context.response_order = [1, 2]
        context.active_responder_idx = 0
        context.pending_responses = {}

        # 创建 WaitRobKongState 实例
        rule_engine = Wuhan7P4LRuleEngine(context)
        observation_builder = Wuhan7P4LObservationBuilder(context)
        state = WaitRobKongState(rule_engine, observation_builder)

        # 验证抛出 ValueError
        with pytest.raises(ValueError) as exc_info:
            state.step(context, 'auto')

        # 验证异常消息
        assert "Unexpected 'auto' action with active responders" in str(exc_info.value), \
            f"Expected error message about active responders, got: {exc_info.value}"

    def test_state_machine_auto_skips_when_no_responders(self):
        """测试状态机在进入 WAIT_ROB_KONG 后自动跳过（集成测试）"""
        # 创建测试环境
        context = GameContext()
        context.players = [PlayerData(player_id=i) for i in range(4)]
        context.current_player_idx = 0
        context.kong_player_idx = 0
        context.rob_kong_tile = Tiles.ONE_CHAR.value

        # 设置玩家0有一个碰牌（用于补杠）
        meld = Meld(
            action_type=MahjongAction(ActionType.PONG, Tiles.ONE_CHAR.value),
            tiles=[Tiles.ONE_CHAR.value] * 3,
            from_player=1
        )
        context.players[0].melds = [meld]
        context.players[0].hand_tiles = [Tiles.ONE_CHAR.value, Tiles.TWO_CHAR.value]

        # 设置其他玩家手牌（不能胡牌的牌）
        context.players[1].hand_tiles = [Tiles.TWO_CHAR.value, Tiles.THREE_CHAR.value]
        context.players[2].hand_tiles = [Tiles.FOUR_CHAR.value, Tiles.FIVE_CHAR.value]
        context.players[3].hand_tiles = [Tiles.SIX_CHAR.value, Tiles.SEVEN_CHAR.value]

        # 设置 active_responders 为空（模拟 enter() 后的结果）
        context.active_responders = []
        context.response_order = []
        context.active_responder_idx = 0
        context.pending_responses = {}

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

        # 转换到 WAIT_ROB_KONG 状态
        state_machine.transition_to(GameStateType.WAIT_ROB_KONG, context)

        # 验证状态已自动跳过到 DRAWING_AFTER_GONG
        assert state_machine.current_state_type == GameStateType.DRAWING_AFTER_GONG, \
            f"Expected DRAWING_AFTER_GONG state, got {state_machine.current_state_type}"

        # 验证补杠已执行
        assert len(context.players[0].melds) == 1, \
            "Expected player_0 to have 1 meld after supplement kong"
        assert context.players[0].melds[0].action_type.action_type == ActionType.KONG_SUPPLEMENT, \
            "Expected meld to be KONG_SUPPLEMENT"

        # 验证 is_kong_draw 标记已设置
        assert context.is_kong_draw is True, \
            "Expected is_kong_draw to be True"

    def test_state_machine_no_skip_when_has_responders(self):
        """测试状态机在有响应者时不自动跳过（集成测试）"""
        # 创建测试环境
        context = GameContext()
        context.players = [PlayerData(player_id=i) for i in range(4)]
        context.current_player_idx = 0
        context.kong_player_idx = 0
        context.rob_kong_tile = Tiles.ONE_CHAR.value

        # 设置玩家0有一个碰牌（用于补杠）
        meld = Meld(
            action_type=MahjongAction(ActionType.PONG, Tiles.ONE_CHAR.value),
            tiles=[Tiles.ONE_CHAR.value] * 3,
            from_player=1
        )
        context.players[0].melds = [meld]
        context.players[0].hand_tiles = [Tiles.ONE_CHAR.value, Tiles.TWO_CHAR.value]

        # 设置玩家1有胡牌牌型（可以抢杠）
        # 给玩家1一个能胡牌的手牌（简单的将牌+面子）
        context.players[1].hand_tiles = [
            Tiles.ONE_CHAR.value, Tiles.ONE_CHAR.value,  # 将牌
            Tiles.TWO_CHAR.value, Tiles.THREE_CHAR.value, Tiles.FOUR_CHAR.value,  # 顺子
            Tiles.FIVE_CHAR.value, Tiles.SIX_CHAR.value, Tiles.SEVEN_CHAR.value,  # 顺子
            Tiles.EIGHT_CHAR.value, Tiles.NINE_CHAR.value, Tiles.ONE_CHAR.value,  # 顺子（缺一张）
        ]

        context.players[2].hand_tiles = [Tiles.FOUR_CHAR.value, Tiles.FIVE_CHAR.value]
        context.players[3].hand_tiles = [Tiles.SIX_CHAR.value, Tiles.SEVEN_CHAR.value]

        # 设置 active_responders 有玩家（模拟 enter() 后的结果）
        context.active_responders = [1]
        context.response_order = [1]
        context.active_responder_idx = 0
        context.pending_responses = {}

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

        # 转换到 WAIT_ROB_KONG 状态
        state_machine.transition_to(GameStateType.WAIT_ROB_KONG, context)

        # 验证状态没有自动跳过
        assert state_machine.current_state_type == GameStateType.WAIT_ROB_KONG, \
            f"Expected WAIT_ROB_KONG state, got {state_machine.current_state_type}"

        # 验证当前玩家是第一个响应者
        assert context.current_player_idx == 1, \
            f"Expected current_player_idx to be 1 (first responder), got {context.current_player_idx}"

        # 验证 active_responders 不为空
        assert len(context.active_responders) > 0, \
            "Expected active_responders to be non-empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
