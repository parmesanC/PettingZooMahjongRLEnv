"""
全局状态观测构建器测试
"""

import pytest
import numpy as np
from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import (
    Wuhan7P4LObservationBuilder,
)
from src.mahjong_rl.core.GameData import GameContext, ActionRecord
from src.mahjong_rl.core.constants import GameStateType
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.core.PlayerData import PlayerData, Meld


class TestGlobalObservationBuilder:
    """测试全局状态观测构建器"""

    def test_build_global_observation_method_exists(self):
        """测试 build_global_observation 方法存在"""
        builder = Wuhan7P4LObservationBuilder()
        assert hasattr(builder, "build_global_observation")

    def test_build_global_observation_output_format(self):
        """测试输出格式正确"""
        builder = Wuhan7P4LObservationBuilder()

        # 创建测试 GameContext
        context = self._create_test_context()

        # 构建全局观测
        global_obs = builder.build_global_observation(context, training_phase=1)

        # 检查必需字段存在
        assert "player_0_hand" in global_obs
        assert "player_1_hand" in global_obs
        assert "player_2_hand" in global_obs
        assert "player_3_hand" in global_obs
        assert "wall_tiles" in global_obs
        assert "discard_piles" in global_obs
        assert "melds" in global_obs
        assert "current_player" in global_obs
        assert "remaining_wall_count" in global_obs
        assert "game_progress" in global_obs

    def test_build_global_observation_shapes(self):
        """测试各字段形状正确"""
        builder = Wuhan7P4LObservationBuilder()
        context = self._create_test_context()

        global_obs = builder.build_global_observation(context, training_phase=1)

        # 检查形状
        assert global_obs["player_0_hand"].shape == (14, 34)
        assert global_obs["player_1_hand"].shape == (14, 34)
        assert global_obs["player_2_hand"].shape == (14, 34)
        assert global_obs["player_3_hand"].shape == (14, 34)
        assert global_obs["wall_tiles"].shape == (34, 34)
        assert global_obs["discard_piles"].shape == (4, 34)
        assert global_obs["melds"].shape == (4, 16, 34)

    def test_phase_1_full_visibility(self):
        """测试 Phase 1 返回完整信息"""
        builder = Wuhan7P4LObservationBuilder()
        context = self._create_test_context()

        # Phase 1: 完全可见
        global_obs_p1 = builder.build_global_observation(context, training_phase=1)

        # 检查手牌不为全零（即没有完全遮蔽）
        for i in range(4):
            hand = global_obs_p1[f"player_{i}_hand"]
            # 应该有非零值（因为每个玩家都有手牌）
            assert hand.sum() > 0

    def test_phase_2_progressive_masking(self):
        """测试 Phase 2 渐进遮蔽"""
        builder = Wuhan7P4LObservationBuilder()
        context = self._create_test_context()

        # Phase 2: 渐进遮蔽
        global_obs_p2 = builder.build_global_observation(context, training_phase=2)

        # 检查有一定数量的遮蔽
        masked_count = 0
        for i in range(4):
            hand = global_obs_p2[f"player_{i}_hand"]
            masked_count += (hand.sum(axis=1) == 0).sum()

        # 应该有部分遮蔽（至少2个位置被遮蔽）
        assert masked_count > 0

    def test_phase_3_belief_sampling(self):
        """测试 Phase 3 信念采样（暂时返回零掩码）"""
        builder = Wuhan7P4LObservationBuilder()
        context = self._create_test_context()

        # Phase 3: 信念采样
        global_obs_p3 = builder.build_global_observation(context, training_phase=3)

        # 检查所有手牌为全零（当前实现返回零掩码）
        for i in range(4):
            hand = global_obs_p3[f"player_{i}_hand"]
            assert hand.sum() == 0

    def test_game_progress_calculation(self):
        """测试游戏进度计算"""
        builder = Wuhan7P4LObservationBuilder()
        context = self._create_test_context()

        global_obs = builder.build_global_observation(context, training_phase=1)

        # 检查游戏进度在 [0, 1] 范围内
        progress = global_obs["game_progress"]
        assert 0.0 <= progress <= 1.0

    def _create_test_context(self) -> GameContext:
        """创建测试用的 GameContext"""
        context = GameContext()
        context.current_state = GameStateType.PLAYER_DECISION
        context.current_player_idx = 0
        context.dealer_idx = 0

        # 添加手牌
        for player in context.players:
            player.hand_tiles = [i % 34 for i in range(13)]

        # 添加一些牌到牌墙
        for i in range(50):
            context.wall.append(i % 34)

        return context
