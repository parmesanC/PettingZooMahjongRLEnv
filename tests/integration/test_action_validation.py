"""
测试状态机动作验证机制

验证状态机在各状态下能正确拒绝非法动作，保证游戏流程的健壮性。
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import numpy as np

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import GameStateType, ActionType, Tiles
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.state_machine.machine import MahjongStateMachine
from src.mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder


class TestActionValidation:
    """测试动作验证机制"""

    @pytest.fixture
    def state_machine_env(self):
        """创建测试环境和状态机"""
        # 创建游戏上下文
        context = GameContext()
        context.players = [PlayerData(player_id=i) for i in range(4)]

        # 创建规则引擎和观测构建器
        rule_engine = Wuhan7P4LRuleEngine(context)
        observation_builder = Wuhan7P4LObservationBuilder(context)

        # 创建状态机
        state_machine = MahjongStateMachine(
            rule_engine=rule_engine,
            observation_builder=observation_builder,
            logger=None,
            enable_logging=False
        )
        state_machine.set_context(context)

        return context, state_machine

    def test_player_decision_discard_invalid_tile(self, state_machine_env):
        """测试 PLAYER_DECISION 状态打出不存在的牌"""
        context, state_machine = state_machine_env

        # 设置手牌（不包含 33-白板）- 需要合法的手牌数量（11张，因为摸牌后是11张）
        context.players[0].hand_tiles = [
            Tiles.ONE_CHAR.value,  # 0
            Tiles.TWO_CHAR.value,  # 1
            Tiles.THREE_CHAR.value,  # 2
            Tiles.FOUR_CHAR.value,  # 3
            Tiles.FIVE_CHAR.value,  # 4
            Tiles.SIX_CHAR.value,  # 5
            Tiles.SEVEN_CHAR.value,  # 6
            Tiles.EIGHT_CHAR.value,  # 7
            Tiles.NINE_CHAR.value,  # 8
            Tiles.ONE_BAM.value,  # 9
            Tiles.TWO_BAM.value,  # 10
        ]
        context.current_player_idx = 0
        context.current_state = GameStateType.PLAYER_DECISION

        # 尝试打出 33-白板（不在手牌中）
        action = MahjongAction(ActionType.DISCARD, Tiles.WHITE_DRAGON.value)

        # 应该抛出异常（validate_action 会拒绝不在可用动作列表中的动作）
        with pytest.raises(ValueError, match="Invalid action DISCARD|not in player's hand"):
            state_machine.states[GameStateType.PLAYER_DECISION].step(context, action)

    def test_player_decision_concealed_kong_insufficient_tiles(self, state_machine_env):
        """测试 PLAYER_DECISION 状态暗杠时牌数不足"""
        context, state_machine = state_machine_env

        # 设置手牌（只有2张1万，不足以暗杠）- 需要合法的手牌数量（11张）
        context.players[0].hand_tiles = [
            Tiles.ONE_CHAR.value,
            Tiles.ONE_CHAR.value,
            Tiles.TWO_CHAR.value,
            Tiles.THREE_CHAR.value,
            Tiles.FOUR_CHAR.value,
            Tiles.FIVE_CHAR.value,
            Tiles.SIX_CHAR.value,
            Tiles.SEVEN_CHAR.value,
            Tiles.EIGHT_CHAR.value,
            Tiles.NINE_CHAR.value,
            Tiles.ONE_BAM.value,
        ]
        context.current_player_idx = 0
        context.current_state = GameStateType.PLAYER_DECISION

        # 尝试暗杠 1万（只有2张）
        action = MahjongAction(ActionType.KONG_CONCEALED, Tiles.ONE_CHAR.value)

        # 应该抛出异常（validation会先检查可用动作，失败是因为牌数不足）
        # 实际上这里会先在 validate_action 阶段失败，因为 KONG_CONCEALED 不在可用动作列表中
        with pytest.raises(ValueError, match="Invalid action|cannot concealed kong"):
            state_machine.states[GameStateType.PLAYER_DECISION].step(context, action)

    def test_player_decision_pass_not_allowed(self, state_machine_env):
        """测试 PLAYER_DECISION 状态不允许 PASS 动作"""
        context, state_machine = state_machine_env

        # 设置手牌
        context.players[0].hand_tiles = [
            Tiles.ONE_CHAR.value,
            Tiles.TWO_CHAR.value,
            Tiles.THREE_CHAR.value,
            Tiles.FOUR_CHAR.value,
            Tiles.FIVE_CHAR.value,
        ]
        context.current_player_idx = 0
        context.current_state = GameStateType.PLAYER_DECISION

        # 尝试 PASS（在 PLAYER_DECISION 状态下不允许）
        action = MahjongAction(ActionType.PASS, -1)

        # 应该抛出异常
        with pytest.raises(ValueError, match="PASS action not allowed"):
            state_machine.states[GameStateType.PLAYER_DECISION].step(context, action)

    def test_meld_decision_discard_invalid_tile(self, state_machine_env):
        """测试 MELD_DECISION 状态打出不存在的牌"""
        context, state_machine = state_machine_env

        # 设置手牌（不包含白板）
        context.players[0].hand_tiles = [
            Tiles.ONE_CHAR.value,
            Tiles.TWO_CHAR.value,
            Tiles.THREE_CHAR.value,
            Tiles.FOUR_CHAR.value,
            Tiles.FIVE_CHAR.value,
            Tiles.SIX_CHAR.value,
            Tiles.SEVEN_CHAR.value,
            Tiles.EIGHT_CHAR.value,
        ]
        context.current_player_idx = 0
        context.current_state = GameStateType.MELD_DECISION

        # 尝试打出白板（不在手牌中）
        action = MahjongAction(ActionType.DISCARD, Tiles.WHITE_DRAGON.value)

        # 应该抛出异常
        with pytest.raises(ValueError, match="not in player's hand|Invalid action"):
            state_machine.states[GameStateType.MELD_DECISION].step(context, action)

    def test_meld_decision_win_not_allowed(self, state_machine_env):
        """测试 MELD_DECISION 状态不允许 WIN 动作"""
        context, state_machine = state_machine_env

        # 设置手牌
        context.players[0].hand_tiles = [
            Tiles.ONE_CHAR.value,
            Tiles.TWO_CHAR.value,
            Tiles.THREE_CHAR.value,
            Tiles.FOUR_CHAR.value,
            Tiles.FIVE_CHAR.value,
        ]
        context.current_player_idx = 0
        context.current_state = GameStateType.MELD_DECISION

        # 尝试 WIN（在 MELD_DECISION 状态下不允许）
        action = MahjongAction(ActionType.WIN, -1)

        # 应该抛出异常（validate_action 会拒绝 WIN 动作，因为它不在可用动作列表中）
        with pytest.raises(ValueError, match="Invalid action WIN|not in rob_kong_players"):
            state_machine.states[GameStateType.MELD_DECISION].step(context, action)

    def test_meld_decision_pass_not_allowed(self, state_machine_env):
        """测试 MELD_DECISION 状态不允许 PASS 动作"""
        context, state_machine = state_machine_env

        # 设置手牌
        context.players[0].hand_tiles = [
            Tiles.ONE_CHAR.value,
            Tiles.TWO_CHAR.value,
            Tiles.THREE_CHAR.value,
            Tiles.FOUR_CHAR.value,
            Tiles.FIVE_CHAR.value,
        ]
        context.current_player_idx = 0
        context.current_state = GameStateType.MELD_DECISION

        # 尝试 PASS（在 MELD_DECISION 状态下不允许）
        action = MahjongAction(ActionType.PASS, -1)

        # 应该抛出异常
        with pytest.raises(ValueError, match="PASS action is not allowed"):
            state_machine.states[GameStateType.MELD_DECISION].step(context, action)

    def test_wait_rob_kong_invalid_action_type(self, state_machine_env):
        """测试 WAIT_ROB_KONG 状态非法动作类型"""
        context, state_machine = state_machine_env

        # 设置上下文进入 WAIT_ROB_KONG 状态
        context.current_player_idx = 0
        context.current_state = GameStateType.WAIT_ROB_KONG
        context.rob_kong_tile = Tiles.ONE_CHAR.value
        context.kong_player_idx = 1
        context.rob_kong_responses = {}
        context.current_responder_idx = 0

        # 设置响应顺序
        wait_rob_kong_state = state_machine.states[GameStateType.WAIT_ROB_KONG]
        wait_rob_kong_state.response_order = [2, 3]

        # 尝试 PONG 动作（只应允许 WIN/PASS）
        action = MahjongAction(ActionType.PONG, Tiles.ONE_CHAR.value)

        # 应该抛出异常
        with pytest.raises(ValueError, match="Only WIN or PASS actions allowed"):
            wait_rob_kong_state.step(context, action)

    def test_wait_rob_kong_fake_win(self, state_machine_env):
        """测试 WAIT_ROB_KONG 状态虚假抢杠"""
        context, state_machine = state_machine_env

        # 设置上下文
        context.current_player_idx = 2
        context.current_state = GameStateType.WAIT_ROB_KONG
        context.rob_kong_tile = Tiles.ONE_CHAR.value
        context.kong_player_idx = 0
        context.rob_kong_responses = {}
        context.current_responder_idx = 0

        # 设置玩家2手牌（无法胡牌）
        context.players[2].hand_tiles = [
            Tiles.TWO_CHAR.value,
            Tiles.THREE_CHAR.value,
            Tiles.FOUR_CHAR.value,
            Tiles.FIVE_CHAR.value,
            Tiles.SIX_CHAR.value,
        ]

        # 设置响应顺序（玩家2不在可抢杠列表中）
        wait_rob_kong_state = state_machine.states[GameStateType.WAIT_ROB_KONG]
        wait_rob_kong_state.response_order = [2]
        wait_rob_kong_state.rob_kong_players = []  # 玩家2不能抢杠

        # 尝试抢杠（但实际不能胡）
        action = MahjongAction(ActionType.WIN, -1)

        # 应该抛出异常
        with pytest.raises(ValueError, match="cannot rob kong"):
            wait_rob_kong_state.step(context, action)

    def test_valid_discard_action_passes(self, state_machine_env):
        """测试合法的打牌动作能正常执行"""
        context, state_machine = state_machine_env

        # 设置手牌
        tile_to_discard = Tiles.ONE_CHAR.value
        context.players[0].hand_tiles = [
            tile_to_discard,
            Tiles.TWO_CHAR.value,
            Tiles.THREE_CHAR.value,
            Tiles.FOUR_CHAR.value,
            Tiles.FIVE_CHAR.value,
        ]
        context.current_player_idx = 0
        context.current_state = GameStateType.PLAYER_DECISION

        # 打出手中的牌
        action = MahjongAction(ActionType.DISCARD, tile_to_discard)

        # 应该正常执行，不抛异常
        try:
            next_state = state_machine.states[GameStateType.PLAYER_DECISION].step(context, action)
            assert next_state == GameStateType.DISCARDING
        except ValueError as e:
            pytest.fail(f"Valid action was rejected: {e}")

    def test_valid_concealed_kong_passes(self, state_machine_env):
        """测试合法的暗杠动作能正常执行"""
        context, state_machine = state_machine_env

        # 设置手牌（有4张1万）
        kong_tile = Tiles.ONE_CHAR.value
        context.players[0].hand_tiles = [
            kong_tile, kong_tile, kong_tile, kong_tile,
            Tiles.TWO_CHAR.value,
            Tiles.THREE_CHAR.value,
            Tiles.FOUR_CHAR.value,
            Tiles.FIVE_CHAR.value,
        ]
        context.current_player_idx = 0
        context.current_state = GameStateType.PLAYER_DECISION

        # 暗杠
        action = MahjongAction(ActionType.KONG_CONCEALED, kong_tile)

        # 应该正常执行，不抛异常
        try:
            next_state = state_machine.states[GameStateType.PLAYER_DECISION].step(context, action)
            assert next_state == GameStateType.GONG
        except ValueError as e:
            pytest.fail(f"Valid action was rejected: {e}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
