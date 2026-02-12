"""
NFSP + MAPPO + Transformer 网络架构

包含：
1. ObservationEncoder: 观测编码器
2. TransformerHistoryEncoder: Transformer编码动作历史
3. ActorCriticNetwork: 最佳响应网络（Actor + Critic）
4. AveragePolicyNetwork: 平均策略网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class HandEncoder(nn.Module):
    """手牌编码器"""

    def __init__(self, output_dim: int = 64):
        super().__init__()
        # 输入：34维（每种牌的数量）
        self.net = nn.Sequential(
            nn.Linear(34, 64), nn.ReLU(), nn.Linear(64, output_dim), nn.ReLU()
        )

    def forward(self, hand: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hand: [batch, 34] 或 [batch, 4, 34]（全局手牌）
        Returns:
            [batch, output_dim] 或 [batch, 4*output_dim]
        """
        if hand.dim() == 3:
            # 全局手牌 [batch, 4, 34]
            batch_size = hand.size(0)
            hand = hand.view(batch_size * 4, 34)
            features = self.net(hand)
            features = features.view(batch_size, 4 * features.size(-1))
        else:
            # 私有手牌 [batch, 34]
            features = self.net(hand)
        return features


class DiscardEncoder(nn.Module):
    """弃牌池编码器"""

    def __init__(self, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(34, 48), nn.ReLU(), nn.Linear(48, output_dim), nn.ReLU()
        )

    def forward(self, discard: torch.Tensor) -> torch.Tensor:
        """
        Args:
            discard: [batch, 34]
        Returns:
            [batch, output_dim]
        """
        return self.net(discard)


class MeldEncoder(nn.Module):
    """副露编码器"""

    def __init__(self, output_dim: int = 64):
        super().__init__()
        # 编码每组副露
        self.meld_net = nn.Sequential(
            nn.Linear(4 + 4 + 1, 32),  # 4张牌 + 4个位置 + 1个动作类型
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        # 聚合所有副露
        self.aggregate = nn.Sequential(
            nn.Linear(16 * 16, 128),  # 16组副露
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )

    def forward(self, melds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            melds: [batch, 16, 9] (action_types[16] + tiles[256] + group_indices[32] 简化)
        Returns:
            [batch, output_dim]
        """
        batch_size = melds.size(0)
        # 编码每组副露
        meld_features = self.meld_net(melds)  # [batch, 16, 16]
        # 展平并聚合
        meld_features = meld_features.view(batch_size, -1)
        return self.aggregate(meld_features)


class TransformerHistoryEncoder(nn.Module):
    """Transformer编码动作历史"""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_history: int = 80,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_history = max_history

        # 动作嵌入（类型 + 参数 + 玩家）
        self.action_type_embed = nn.Embedding(11, 32)  # 11种动作类型
        self.action_param_embed = nn.Embedding(35, 32)  # 34种牌 + 1个通配符
        self.player_embed = nn.Embedding(4, 16)  # 4个玩家

        # 输入投影
        self.input_projection = nn.Linear(32 + 32 + 16, hidden_dim)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, max_history, hidden_dim))

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出投影（取最后一步的表示）
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

    def forward(
        self,
        action_types: torch.Tensor,
        action_params: torch.Tensor,
        action_players: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            action_types: [batch, seq_len] 动作类型索引
            action_params: [batch, seq_len] 动作参数索引
            action_players: [batch, seq_len] 玩家索引
            mask: [batch, seq_len] 有效位置掩码（可选）
        Returns:
            [batch, hidden_dim] 历史编码
        """
        batch_size, seq_len = action_types.size()

        # 嵌入
        type_emb = self.action_type_embed(action_types)  # [batch, seq_len, 32]
        param_emb = self.action_param_embed(action_params)  # [batch, seq_len, 32]
        player_emb = self.player_embed(action_players)  # [batch, seq_len, 16]

        # 拼接
        x = torch.cat([type_emb, param_emb, player_emb], dim=-1)  # [batch, seq_len, 80]
        x = self.input_projection(x)  # [batch, seq_len, hidden_dim]

        # 添加位置编码
        if seq_len <= self.max_history:
            x = x + self.pos_encoding[:, :seq_len, :]
        else:
            x = x + self.pos_encoding[:, : self.max_history, :]

        # Transformer编码
        if mask is not None:
            # 创建 key padding mask
            key_mask = ~mask.bool()  # [batch, seq_len]
            x = self.transformer(x, src_key_padding_mask=key_mask)
        else:
            x = self.transformer(x)

        # 取最后一步的表示
        last_step = x[:, -1, :]  # [batch, hidden_dim]

        return self.output_projection(last_step)


class StateEncoder(nn.Module):
    """全局状态编码器"""

    def __init__(self, output_dim: int = 32):
        super().__init__()
        # 编码各种全局信息
        # 输入维度：current_player(1) + remaining_tiles(1) + fan_counts(4) +
        #           current_phase(1) + special_indicators(2) + dealer(1) = 10
        self.net = nn.Sequential(
            nn.Linear(10, 48),  # 修复：10维不是13维 (current_player是标量不是one-hot)
            nn.ReLU(),
            nn.Linear(48, output_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        current_player: torch.Tensor,
        remaining_tiles: torch.Tensor,
        fan_counts: torch.Tensor,
        current_phase: torch.Tensor,
        special_indicators: torch.Tensor,
        dealer: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            current_player: [batch, 1] 当前玩家索引
            remaining_tiles: [batch, 1] 剩余牌数
            fan_counts: [batch, 4] 各玩家番数
            current_phase: [batch, 1] 当前阶段
            special_indicators: [batch, 2] 赖子和皮子
            dealer: [batch, 1] 庄家
        Returns:
            [batch, output_dim]
        """
        # 拼接所有状态信息
        state = torch.cat(
            [
                current_player.float(),
                remaining_tiles.float(),
                fan_counts.float(),
                current_phase.float(),
                special_indicators.float(),
                dealer.float(),
            ],
            dim=-1,
        )

        return self.net(state)


class WallEncoder(nn.Module):
    """牌墙编码器"""

    def __init__(self, output_dim: int = 32):
        super().__init__()
        # 输入：82维（牌墙中的牌ID或34表示空）
        self.net = nn.Sequential(
            nn.Linear(82, 64), nn.ReLU(), nn.Linear(64, output_dim), nn.ReLU()
        )

    def forward(self, wall: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wall: [batch, 82]
        Returns:
            [batch, output_dim]
        """
        return self.net(wall)


class ObservationEncoder(nn.Module):
    """完整观测编码器 - 完全匹配 Wuhan7P4LObservationBuilder 的输出"""

    def __init__(
        self,
        hidden_dim: int = 256,
        transformer_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 各个编码器
        self.global_hand_encoder = HandEncoder(
            output_dim=64
        )  # global_hand: [4, 34] -> 256
        self.private_hand_encoder = HandEncoder(
            output_dim=32
        )  # private_hand: [34] -> 32
        self.discard_encoder = DiscardEncoder(
            output_dim=32
        )  # discard_pool_total: [34] -> 32
        self.wall_encoder = WallEncoder(output_dim=32)  # wall: [82] -> 32
        self.meld_encoder = MeldEncoder(output_dim=64)  # melds -> 64
        self.history_encoder = (
            TransformerHistoryEncoder(  # action_history -> hidden_dim
                hidden_dim=hidden_dim,
                num_layers=transformer_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        )
        self.state_encoder = StateEncoder(output_dim=32)  # 全局状态 -> 32

        # 融合层
        # global_hand(256) + private_hand(32) + discard(32) + wall(32) +
        # melds(64) + history(hidden_dim) + state(32)
        total_dim = 256 + 32 + 32 + 32 + 64 + hidden_dim + 32
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs: 观测字典，完全匹配 Wuhan7P4LObservationBuilder 的输出
                - global_hand: [batch, 136] (4*34)
                - private_hand: [batch, 34]
                - discard_pool_total: [batch, 34]
                - wall: [batch, 82]
                - melds: dict with action_types[16], tiles[256], group_indices[32]
                - action_history: dict with types[80], params[80], players[80]
                - special_gangs: [batch, 12]
                - current_player: [batch, 1]
                - fan_counts: [batch, 4]
                - special_indicators: [batch, 2]
                - remaining_tiles: [batch, 1] 或 [batch]
                - dealer: [batch, 1]
                - current_phase: [batch, 1]
        Returns:
            [batch, hidden_dim] 编码后的特征
        """
        features = []

        # 1. 编码全局手牌 [batch, 136] -> [batch, 4, 34] -> [batch, 256]
        global_hand = obs["global_hand"]  # [batch, 136]
        batch_size = global_hand.size(0)
        global_hand = global_hand.view(batch_size, 4, 34)
        global_hand_feat = self.global_hand_encoder(global_hand)  # [batch, 256]
        features.append(global_hand_feat)

        # 2. 编码私有手牌 [batch, 34] -> [batch, 32]
        private_hand = obs["private_hand"]  # [batch, 34]
        private_hand_feat = self.private_hand_encoder(private_hand)  # [batch, 32]
        features.append(private_hand_feat)

        # 3. 编码弃牌池 [batch, 34] -> [batch, 32]
        discard_feat = self.discard_encoder(obs["discard_pool_total"])
        features.append(discard_feat)

        # 4. 编码牌墙 [batch, 82] -> [batch, 32]
        wall_feat = self.wall_encoder(obs["wall"])
        features.append(wall_feat)

        # 5. 编码副露 -> [batch, 64]
        melds_tensor = self._process_melds(obs["melds"])
        meld_feat = self.meld_encoder(melds_tensor)
        features.append(meld_feat)

        # 6. 编码动作历史 -> [batch, hidden_dim]
        # 注意：action_history 中的字段是 long 类型，用于 Embedding
        history_feat = self.history_encoder(
            obs["action_history"]["types"].long(),
            obs["action_history"]["params"].long(),
            obs["action_history"]["players"].long(),
        )
        features.append(history_feat)

        # 7. 编码全局状态 [batch, 32]
        # 确保 remaining_tiles 是 [batch, 1] 形状
        remaining_tiles = obs["remaining_tiles"]
        if remaining_tiles.dim() == 1:
            remaining_tiles = remaining_tiles.unsqueeze(-1)
        elif remaining_tiles.dim() == 0:
            remaining_tiles = remaining_tiles.unsqueeze(0).unsqueeze(0)

        state_feat = self.state_encoder(
            obs["current_player"],
            remaining_tiles,
            obs["fan_counts"],
            obs["current_phase"],
            obs["special_indicators"],
            obs["dealer"],
        )
        features.append(state_feat)

        # 融合所有特征
        combined = torch.cat(features, dim=-1)

        return self.fusion(combined)

    def _process_melds(self, melds: Dict) -> torch.Tensor:
        """将melds字典转换为张量 [batch, 16, 9]"""
        batch_size = melds["action_types"].size(0)
        # 简化处理：将 action_types, tiles, group_indices 拼接
        # action_types: [batch, 16]
        # tiles: [batch, 256] -> 需要 reshape
        # group_indices: [batch, 32] -> 需要 reshape

        action_types = melds["action_types"].float()  # [batch, 16]
        tiles = melds["tiles"].float().view(batch_size, 16, 16)  # [batch, 16, 16] 简化
        group_indices = (
            melds["group_indices"].float().view(batch_size, 16, 2)
        )  # [batch, 16, 2]

        # 拼接: [batch, 16, 1+16+2] = [batch, 16, 19]，然后截断到9维
        melds_tensor = torch.cat(
            [
                action_types.unsqueeze(-1),
                tiles[:, :, :4],  # 只取前4维
                group_indices,
            ],
            dim=-1,
        )  # [batch, 16, 7]

        # 填充到9维
        padding = torch.zeros(batch_size, 16, 2, device=melds_tensor.device)
        melds_tensor = torch.cat([melds_tensor, padding], dim=-1)  # [batch, 16, 9]

        return melds_tensor


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic 网络（最佳响应网络）
    用于 NFSP 的强化学习部分（MAPPO）
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        transformer_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        action_mask_size: int = 145,
        use_belief: bool = False,
        n_belief_samples: int = 5,
    ):
        super().__init__()

        # 观测编码器
        self.encoder = ObservationEncoder(
            hidden_dim=hidden_dim,
            transformer_layers=transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Actor 头部（策略）
        self.actor_type = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 11),  # 11种动作类型
        )

        self.actor_param = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 34),  # 34种牌参数
        )

        # Critic 头部（价值）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        # 信念集成（可选）
        self.use_belief = use_belief
        self.n_belief_samples = n_belief_samples

        if self.use_belief:
            # 信念编码器
            self.belief_encoder = nn.Sequential(
                nn.Linear(34 * n_belief_samples, hidden_dim),
                nn.ReLU(),
            )
        else:
            self.belief_encoder = None

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        belief_samples: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: 观测字典
            action_mask: [batch, 145] 动作掩码
            belief_samples: 采样的对手观测列表（可选）

        Returns:
            action_type_logits: [batch, 11]
            action_param_logits: [batch, 34]
            value: [batch, 1]
        """
        # 编码观测
        features = self.encoder(obs)  # [batch, hidden_dim]

        # 信念集成（可选）
        if self.use_belief and belief_samples is not None:
            # 编码信念采样
            belief_features = []
            for sample in belief_samples:
                sample_feat = self.encoder(sample)
                belief_features.append(sample_feat)

            # 平均所有采样的特征
            avg_belief_feat = torch.stack(belief_features, dim=0) / len(belief_features)

            # 拼接到主特征
            combined = torch.cat([features, avg_belief_feat], dim=-1)
            features = combined

        # Actor 输出
        action_type_logits = self.actor_type(features)  # [batch, 11]
        action_param_logits = self.actor_param(features)  # [batch, 34]

        # 应用动作掩码（在 softmax 前）
        # 将 action_mask 转换为动作类型掩码和参数掩码
        type_mask, param_mask = self._split_action_mask(action_mask)

        # 对无效动作设置极小的 logits
        action_type_logits = action_type_logits.masked_fill(~type_mask.bool(), -1e9)
        action_param_logits = action_param_logits.masked_fill(~param_mask.bool(), -1e9)

        # Critic 输出
        value = self.critic(features)

        return action_type_logits, action_param_logits, value

    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """只获取价值估计（用于评估）"""
        features = self.encoder(obs)
        return self.critic(features)

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        action: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作、对数概率、熵和价值
        用于 PPO 训练
        """
        action_type_logits, action_param_logits, value = self.forward(obs, action_mask)

        # 创建分布
        type_probs = F.softmax(action_type_logits, dim=-1)
        param_probs = F.softmax(action_param_logits, dim=-1)

        type_dist = torch.distributions.Categorical(type_probs)
        param_dist = torch.distributions.Categorical(param_probs)

        if action is None:
            # 采样动作
            action_type = type_dist.sample()
            action_param = param_dist.sample()
        else:
            action_type, action_param = action

        # 计算对数概率
        log_prob_type = type_dist.log_prob(action_type)
        log_prob_param = param_dist.log_prob(action_param)
        log_prob = log_prob_type + log_prob_param

        # 计算熵
        entropy_type = type_dist.entropy()
        entropy_param = param_dist.entropy()
        entropy = entropy_type + entropy_param

        return action_type, action_param, log_prob, entropy, value

    def _split_action_mask(
        self, action_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将145维的action_mask分割为动作类型掩码和参数掩码

        action_mask 索引定义：
        - DISCARD: 0-33 (34位)
        - CHOW: 34-36 (3位)
        - PONG: 37 (1位)
        - KONG_EXPOSED: 38 (1位)
        - KONG_SUPPLEMENT: 39-72 (34位)
        - KONG_CONCEALED: 73-106 (34位)
        - KONG_RED: 107 (1位)
        - KONG_SKIN: 108-141 (34位)
        - KONG_LAZY: 142 (1位)
        - WIN: 143 (1位)
        - PASS: 144 (1位)
        """
        batch_size = action_mask.size(0)

        # 动作类型掩码（11种）
        type_mask = torch.zeros(batch_size, 11, device=action_mask.device)
        # 参数掩码（34种牌）
        param_mask = torch.zeros(batch_size, 34, device=action_mask.device)

        # 填充动作类型掩码
        type_mask[:, 0] = action_mask[:, 0:34].any(dim=-1)  # DISCARD
        type_mask[:, 1] = action_mask[:, 34:37].any(dim=-1)  # CHOW
        type_mask[:, 2] = action_mask[:, 37]  # PONG
        type_mask[:, 3] = action_mask[:, 38]  # KONG_EXPOSED
        type_mask[:, 4] = action_mask[:, 39:73].any(dim=-1)  # KONG_SUPPLEMENT
        type_mask[:, 5] = action_mask[:, 73:107].any(dim=-1)  # KONG_CONCEALED
        type_mask[:, 6] = action_mask[:, 107]  # KONG_RED
        type_mask[:, 7] = action_mask[:, 108:142].any(dim=-1)  # KONG_SKIN
        type_mask[:, 8] = action_mask[:, 142]  # KONG_LAZY
        type_mask[:, 9] = action_mask[:, 143]  # WIN
        type_mask[:, 10] = action_mask[:, 144]  # PASS

        # 填充参数掩码（取所有涉及牌ID的位的并集）
        # 使用 torch.maximum 替代按位或运算（支持 Float 类型）
        param_mask = action_mask[:, 0:34].clone()  # DISCARD
        param_mask = torch.maximum(param_mask, action_mask[:, 39:73])  # KONG_SUPPLEMENT
        param_mask = torch.maximum(param_mask, action_mask[:, 73:107])  # KONG_CONCEALED
        param_mask = torch.maximum(param_mask, action_mask[:, 108:142])  # KONG_LAZY

        return type_mask, param_mask


class AveragePolicyNetwork(nn.Module):
    """
    平均策略网络
    用于 NFSP 的监督学习部分（模仿最佳响应网络）
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        transformer_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 观测编码器（与 Actor-Critic 共享架构）
        self.encoder = ObservationEncoder(
            hidden_dim=hidden_dim,
            transformer_layers=transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 策略头部
        self.policy_type = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 11),  # 11种动作类型
        )

        self.policy_param = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 34),  # 34种牌参数
        )

    def forward(
        self, obs: Dict[str, torch.Tensor], action_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: 观测字典
            action_mask: [batch, 145] 动作掩码
        Returns:
            action_type_logits: [batch, 11]
            action_param_logits: [batch, 34]
        """
        # 编码观测
        features = self.encoder(obs)

        # 策略输出
        action_type_logits = self.policy_type(features)
        action_param_logits = self.policy_param(features)

        # 应用动作掩码
        type_mask, param_mask = self._split_action_mask(action_mask)

        action_type_logits = action_type_logits.masked_fill(~type_mask.bool(), -1e9)
        action_param_logits = action_param_logits.masked_fill(~param_mask.bool(), -1e9)

        return action_type_logits, action_param_logits

    def get_action_probs(
        self, obs: Dict[str, torch.Tensor], action_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作概率分布"""
        action_type_logits, action_param_logits = self.forward(obs, action_mask)

        type_probs = F.softmax(action_type_logits, dim=-1)
        param_probs = F.softmax(action_param_logits, dim=-1)

        return type_probs, param_probs

    def _split_action_mask(
        self, action_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """分割动作掩码（与 ActorCriticNetwork 相同）"""
        batch_size = action_mask.size(0)

        type_mask = torch.zeros(batch_size, 11, device=action_mask.device)
        param_mask = torch.zeros(batch_size, 34, device=action_mask.device)

        type_mask[:, 0] = action_mask[:, 0:34].any(dim=-1)
        type_mask[:, 1] = action_mask[:, 34:37].any(dim=-1)
        type_mask[:, 2] = action_mask[:, 37]
        type_mask[:, 3] = action_mask[:, 38]
        type_mask[:, 4] = action_mask[:, 39:73].any(dim=-1)
        type_mask[:, 5] = action_mask[:, 73:107].any(dim=-1)
        type_mask[:, 6] = action_mask[:, 107]
        type_mask[:, 7] = action_mask[:, 108:142].any(dim=-1)
        type_mask[:, 8] = action_mask[:, 142]
        type_mask[:, 9] = action_mask[:, 143]
        type_mask[:, 10] = action_mask[:, 144]

        # 使用 torch.maximum 替代按位或运算（支持 Float 类型）
        param_mask = action_mask[:, 0:34].clone()
        param_mask = torch.maximum(param_mask, action_mask[:, 39:73])
        param_mask = torch.maximum(param_mask, action_mask[:, 73:107])
        param_mask = torch.maximum(param_mask, action_mask[:, 108:142])

        return type_mask, param_mask


class CentralizedCriticNetwork(nn.Module):
    """
    Centralized Critic 网络用于 MAPPO
    接收所有4个智能体的观测，输出每个智能体的价值估计
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        transformer_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 4个独立的观测编码器
        self.agent_encoders = nn.ModuleList(
            [
                ObservationEncoder(
                    hidden_dim=hidden_dim // 2,  # 256
                    transformer_layers=transformer_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(4)
            ]
        )

        # 融合层：concatenate所有agents的特征
        total_feat_dim = 4 * (hidden_dim // 2)  # 4 * 256 = 1024
        self.fusion = nn.Sequential(
            nn.Linear(total_feat_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # 4个独立的价值头（每个agent一个）
        self.critic_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(4)
            ]
        )

    def forward(self, all_observations: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        计算所有agents的价值估计

        Args:
            all_observations: List of 4 observation dicts (每个格式与ObservationEncoder兼容）

        Returns:
            values: [batch, 4] - 每个agent的价值估计
        """
        # 对每个agent编码观测
        agent_features = []
        for i, obs in enumerate(all_observations):
            feat = self.agent_encoders[i](obs)
            agent_features.append(feat)

        # 融合所有agents的特征
        # agent_features: List[torch.Tensor] where each is [batch, hidden_dim//2] = [batch, 256]
        # Stack along agent dim: [batch, 4, hidden_dim//2] = [batch, 4, 256]
        stacked_features = torch.stack(agent_features, dim=1)
        batch_size = stacked_features.size(0)

        # Reshape for fusion: flatten agent dimension
        # [batch, 4, 256] -> [batch, 4*256] = [batch, 1024]
        fused_features = stacked_features.view(batch_size, -1)

        # 融合
        fused_features = self.fusion(fused_features)  # [batch, hidden_dim]

        # 计算每个agent的价值
        values = []
        for head in self.critic_heads:
            val = head(fused_features)  # [batch, 1]
            values.append(val)

        # Stack along value dimension to get [batch, 4, 1], then squeeze
        stacked = torch.stack(values, dim=1)
        return stacked.squeeze(-1)


def init_weights(module):
    """初始化网络权重"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.01)


def create_networks(config, device="cuda"):
    """
    创建网络实例

    Args:
        config: 配置对象
        device: 设备类型

    Returns:
        best_response_net: ActorCriticNetwork
        average_policy_net: AveragePolicyNetwork
    """
    best_response_net = ActorCriticNetwork(
        hidden_dim=config.network.hidden_dim,
        transformer_layers=config.network.transformer_layers,
        num_heads=config.network.num_heads,
        dropout=config.network.dropout,
        action_mask_size=config.network.action_mask_size,
    ).to(device)

    average_policy_net = AveragePolicyNetwork(
        hidden_dim=config.network.hidden_dim,
        transformer_layers=config.network.transformer_layers,
        num_heads=config.network.num_heads,
        dropout=config.network.dropout,
    ).to(device)

    # 应用权重初始化
    best_response_net.apply(init_weights)
    average_policy_net.apply(init_weights)

    return best_response_net, average_policy_net
