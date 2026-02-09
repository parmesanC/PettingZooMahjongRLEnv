"""
蒙特卡洛采样器（Monte Carlo Sampler）

用于从信念状态中采样可能的手牌组合
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
import numpy as np
import copy

try:
    from src.mahjong_rl.core.GameData import GameContext, PlayerData
    from src.mahjong_rl.core.mahjong_action import MahjongAction
except ImportError:
    # 如果无法导入，创建简化的测试版本
    GameContext = None
    PlayerData = None
    MahjongAction = None


class MonteCarloSampler:
    """蒙特卡洛采样器：从信念分布中采样可能手牌"""

    def __init__(
        self,
        n_samples: int = 5,
        confidence_threshold: float = 0.7,
        max_retries: int = 10,
    ):
        """
        Args:
            n_samples: 采样数量（默认5）
            confidence_threshold: 置信度阈值（默认0.7）
            max_retries: 最大重试次数（默认10）
        """
        self.n_samples = n_samples
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries

    def sample(
        self,
        beliefs: torch.Tensor,
        known_tiles: Optional[torch.Tensor] = None,
        context: Optional["GameContext"] = None,
    ) -> List["GameContext"]:
        """
        Gumbel-Softmax 采样：从信念分布中采样N个可能手牌

        Args:
            beliefs: [batch, 3, 34] - 3个对手的概率分布
            known_tiles: [batch, 34] - 已知的牌（弃牌堆+副露）
            context: GameContext 用于验证采样有效性

        Returns:
            samples: N个采样的GameContext，每个包含采样的对手手牌
        """
        if context is None:
            raise ValueError("context 不能为 None，需要GameContext用于验证")

        batch_size = beliefs.size(0)
        num_opponents = beliefs.size(1)

        samples = []
        retry_count = 0

        while len(samples) < self.n_samples and retry_count < self.max_retries:
            # 1. Gumbel-Softmax 采样
            gumbel = -torch.log(-torch.log(torch.rand_like(beliefs)))
            sampled_indices = torch.argmax(beliefs + gumbel, dim=-1)  # [batch, 3, 34]

            # 2. 掩码已知的牌
            if known_tiles is not None:
                for opp in range(num_opponents):
                    opp_known = known_tiles[0, opp, :]  # [34]
                    # 将已知的牌的概率设为极小值
                    sampled_indices[0, opp, :] = sampled_indices[0, opp, :] * (
                        1 - opp_known
                    )

            # 3. 构建采样的手牌
            for i in range(batch_size):
                sample_context = self._build_sampled_context(
                    context, sampled_indices[i], batch_idx=i
                )

                # 4. 验证采样有效性
                if self._validate_sample(sample_context, context):
                    samples.append(sample_context)
                else:
                    retry_count += 1
                    if retry_count >= self.max_retries:
                        # 达到最大重试次数，使用零填充
                        samples.append(sample_context)
                        break

        return samples

    def _build_sampled_context(
        self,
        context: "GameContext",
        sampled_indices: torch.Tensor,
        batch_idx: int = 0,
    ) -> "GameContext":
        """
        创建采样的GameContext副本

        Args:
            context: 原始GameContext
            sampled_indices: [3, 34] - 3个对手的采样牌索引
            batch_idx: 批次索引（用于唯一ID）

        Returns:
            sampled_context: 采样的GameContext副本
        """
        # 深拷贝GameContext
        sampled_context = copy.deepcopy(context)

        # 更新对手手牌（仅更新索引为0,1,2的玩家）
        for opp in range(1, 3):
            opp_idx = opp + 1  # 对手索引为1,2,3对应opponents 0,1,2
            opp_player = sampled_context.players[opp_idx]

            # 从采样索引中提取手牌
            opp_indices = sampled_indices[batch_idx, opp, :]  # [34]

            # 统计每张牌的数量
            from collections import Counter

            tile_counts = Counter(opp_indices.tolist())

            # 构建手牌列表（重复tile_indices指定次数）
            sampled_hand = []
            for tile_id, count in tile_counts.items():
                sampled_hand.extend([tile_id] * count)

            # 确保手牌数量正确（通常是13张）
            expected_count = 13  # 标准手牌数
            if len(sampled_hand) < expected_count:
                # 不足时随机填充（不完美但有效）
                remaining = expected_count - len(sampled_hand)
                # 从未知区域随机选择
                unknown_tiles = [t for t in range(34) if t not in tile_counts]
                if unknown_tiles:
                    sampled_hand.extend(
                        np.random.choice(unknown_tiles, remaining, replace=False)
                    )

            opp_player.hand_tiles = sampled_hand[:13]  # 截断到13张

        return sampled_context

    def _validate_sample(
        self, sample_context: "GameContext", original_context: "GameContext"
    ) -> bool:
        """
        验证采样有效性

        Args:
            sample_context: 采样的GameContext
            original_context: 原始GameContext

        Returns:
            valid: 是否有效
        """
        # 验证1: 手牌数量合理（每玩家13张左右）
        for player in sample_context.players:
            if len(player.hand_tiles) < 10 or len(player.hand_tiles) > 14:
                return False

        # 验证2: 不与弃牌堆重复（简化检查）
        # 实际实现中应该避免这种情况
        # 这里简化处理

        # 验证3: 牌墙数量合理
        if len(sample_context.wall) > 136:
            return False

        # 验证4: 累计手牌数不超过总牌数
        total_tiles = sum(len(p.hand_tiles) for p in sample_context.players)
        if total_tiles > 136:
            return False

        return True

    def sample_with_confidence_adjustment(
        self,
        beliefs: torch.Tensor,
        confidence_scores: torch.Tensor,
        known_tiles: Optional[torch.Tensor] = None,
        context: Optional["GameContext"] = None,
    ) -> List["GameContext"]:
        """
        根据置信度调整采样权重

        Args:
            beliefs: [batch, 3, 34] - 3个对手的概率分布
            confidence_scores: [batch, 3] - 置信度（0-1）
            known_tiles: [batch, 34] - 已知的牌

        Returns:
            samples: N个采样的GameContext
        """
        if context is None:
            raise ValueError("context 不能为 None，需要GameContext用于验证")

        batch_size = beliefs.size(0)
        num_opponents = beliefs.size(1)

        samples = []

        for i in range(batch_size):
            # 根据置信度选择采样策略
            for opp in range(num_opponents):
                conf = confidence_scores[i, opp].item()

                if conf < self.confidence_threshold:
                    # 低置信度：多次采样
                    opp_samples = []
                    for _ in range(self.n_samples):
                        # Gumbel-Softmax采样
                        gumbel = -torch.log(-torch.log(torch.rand(1, 34)))
                        sampled = torch.argmax(beliefs[i, opp, :] + gumbel, dim=-1)

                        # 掩码已知牌
                        if known_tiles is not None:
                            opp_known = known_tiles[i, opp, :]
                            sampled = sampled * (1 - opp_known)

                        opp_samples.append(sampled)
                else:
                    # 高置信度：使用最可能的配置
                    gumbel = -torch.log(-torch.log(torch.rand(1, 34)))
                    sampled = torch.argmax(beliefs[i, opp, :] + gumbel, dim=-1)

                    if known_tiles is not None:
                        opp_known = known_tiles[i, opp, :]
                        sampled = sampled * (1 - opp_known)

                    opp_samples = [sampled]  # 只采样最可能的

            # 为每个采样构建上下文
            for j in range(self.n_samples):
                sample_context = self._build_sampled_context(
                    context,
                    torch.stack(
                        [
                            opp_samples[0][j],
                            opp_samples[1][j],
                            opp_samples[2][j],
                        ]
                    ).unsqueeze(0),
                    batch_idx=i,
                )

                if self._validate_sample(sample_context, context):
                    samples.append(sample_context)

        return samples
