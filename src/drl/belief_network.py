"""
信念状态网络（Belief Network）用于估计对手手牌概率分布

功能：
1. 估计3个对手的34维概率分布
2. 使用公共信息：弃牌池、副露、出牌历史
3. 支持贝叶斯更新动态调整概率
4. 辅助损失训练：预测对手行为
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class BeliefNetwork(nn.Module):
    """信念网络：估计对手手牌概率分布"""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_opponents: int = 3,
        transformer_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_opponents = num_opponents

        # 复用现有编码器（需要从 network.py 导入）
        try:
            from src.drl.network import (
                DiscardEncoder,
                MeldEncoder,
                TransformerHistoryEncoder,
            )

            self.discard_encoder = DiscardEncoder(output_dim=32)
            self.meld_encoder = MeldEncoder(output_dim=64)
            self.history_encoder = TransformerHistoryEncoder(
                hidden_dim=hidden_dim,
                num_layers=transformer_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_history=80,
            )
        except ImportError:
            # 如果无法导入，创建简化版本
            print(
                "Warning: Could not import encoders from network.py, using simplified version"
            )
            self.discard_encoder = nn.Sequential(
                nn.Linear(34, 48), nn.ReLU(), nn.Linear(48, 32), nn.ReLU()
            )
            self.meld_encoder = nn.Sequential(
                nn.Linear(144, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
            )  # 16*9 = 144
            self.history_encoder = nn.Sequential(
                nn.Linear(240, 256), nn.ReLU(), nn.Linear(256, hidden_dim), nn.ReLU()
            )  # 80*3 = 240

        # 融合层
        fusion_dim = 32 + 64 + hidden_dim  # discard + meld + history
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # 对手信念头（每个对手一个）
        self.opponent_beliefs = nn.ModuleList(
            [nn.Linear(hidden_dim, 34) for _ in range(num_opponents)]
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播

        Args:
            obs: 公共观测字典
                - discard_pool: [batch, 34] 弃牌池
                - melds: dict 或 [batch, 16, 9] 副露
                - action_history: dict 包含 types[80], params[80], players[80]

        Returns:
            beliefs: [batch, num_opponents, 34] 每个对手的概率分布
        """
        # 1. 编码弃牌池
        discard_pool = obs["discard_pool"]  # [batch, 34]
        discard_feat = self.discard_encoder(discard_pool)  # [batch, 32]

        # 2. 编码副露
        melds = obs["melds"]
        if isinstance(melds, dict):
            # 转换为张量格式
            batch_size = discard_pool.size(0)
            action_types = melds["action_types"].float()  # [batch, 16]
            tiles = melds["tiles"].float().view(batch_size, 16, 16)
            group_indices = melds["group_indices"].float().view(batch_size, 16, 2)
            melds_tensor = torch.cat(
                [action_types.unsqueeze(-1), tiles, group_indices], dim=-1
            )[:, :, :9]  # [batch, 16, 9]
        else:
            melds_tensor = melds  # [batch, 16, 9]

        meld_feat = self.meld_encoder(melds_tensor)  # [batch, 64]

        # 3. 编码出牌历史
        action_history = obs["action_history"]
        if isinstance(action_history, dict):
            history_feat = self.history_encoder(
                action_history["types"].long(),  # [batch, 80]
                action_history["params"].long(),  # [batch, 80]
                action_history["players"].long(),  # [batch, 80]
            )
        else:
            # 简化：直接使用展平的历史
            batch_size = discard_pool.size(0)
            history_flat = action_history.view(batch_size, -1)
            history_feat = self.history_encoder(history_flat)

        # 4. 融合所有特征
        fused = torch.cat([discard_feat, meld_feat, history_feat], dim=-1)
        features = self.fusion(fused)  # [batch, hidden_dim]

        # 5. 计算每个对手的信念
        beliefs = []
        for head in self.opponent_beliefs:
            belief = head(features)  # [batch, 34]
            beliefs.append(belief)

        # Stack: [batch, num_opponents, 34]
        beliefs = torch.stack(beliefs, dim=1)

        # 6. Softmax 归一化（每个对手独立归一化）
        beliefs = F.softmax(beliefs, dim=-1)

        return beliefs

    def get_opponent_beliefs(self, agent_id: int, context: Dict) -> torch.Tensor:
        """
        获取特定agent看到的对手信念

        Args:
            agent_id: 当前agent索引（0-3）
            context: GameContext或观测字典

        Returns:
            beliefs: [3, 34] 或 [batch, 3, 34] 3个对手的信念
        """
        # 构建公共观测
        obs = self._build_public_observation(agent_id, context)
        return self.forward(obs)

    def update_beliefs(
        self,
        beliefs: torch.Tensor,
        action_history: torch.Tensor,
        discard_pool: torch.Tensor,
        melds: torch.Tensor,
    ) -> torch.Tensor:
        """
        贝叶斯更新：根据新的出牌历史调整信念

        Args:
            beliefs: [batch, 3, 34] 当前信念分布
            action_history: [batch, seq_len, 3] 新的出牌历史
            discard_pool: [batch, 34] 弃牌池
            melds: [batch, 16, 9] 副露

        Returns:
            updated_beliefs: [batch, 3, 34] 更新后的信念
        """
        # 简化实现：降低已打出牌的概率
        batch_size, num_opponents, _ = beliefs.shape

        # 检测最近打出的牌（从历史中提取）
        if action_history.dim() == 3:
            # [batch, seq_len, 3] -> 取最后一步
            last_actions = action_history[:, -1, :]  # [batch, 3]
            # 假设格式：[player_id, action_type, tile_id]
            tile_ids = last_actions[:, 2].long()  # [batch]
        else:
            tile_ids = None

        if tile_ids is not None:
            for i in range(batch_size):
                tile_id = tile_ids[i].item()
                # 对于每个对手，降低打出牌的概率
                for opp in range(num_opponents):
                    if 0 <= tile_id < 34:
                        # 降低该牌的概率（乘以 0.5）
                        beliefs[i, opp, tile_id] *= 0.5

                        # 重新归一化
                        total = beliefs[i, opp, :].sum()
                        if total > 0:
                            beliefs[i, opp, :] /= total

        # 考虑弃牌池信息：降低弃牌池中的牌的概率
        for i in range(batch_size):
            discard_mask = discard_pool[i] > 0  # [34]
            for opp in range(num_opponents):
                # 降低在弃牌堆中的牌的概率
                beliefs[i, opp, :] *= 1.0 - 0.3 * discard_mask.float()
                # 重新归一化
                total = beliefs[i, opp, :].sum()
                if total > 0:
                    beliefs[i, opp, :] /= total

        return beliefs

    def _build_public_observation(
        self, agent_id: int, context: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        构建公共观测（不包含私有信息）

        Args:
            agent_id: 当前agent索引
            context: GameContext或字典

        Returns:
            obs: 公共观测字典
        """
        # 简化实现：假设context是字典格式
        obs = {
            "discard_pool": context.get("discard_pool", torch.zeros(1, 34)),
            "melds": context.get(
                "melds",
                {
                    "action_types": torch.zeros(1, 16),
                    "tiles": torch.zeros(1, 256),
                    "group_indices": torch.zeros(1, 32),
                },
            ),
            "action_history": context.get(
                "action_history",
                {
                    "types": torch.zeros(1, 80).long(),
                    "params": torch.zeros(1, 80).long(),
                    "players": torch.zeros(1, 80).long(),
                },
            ),
        }
        return obs

    def compute_auxiliary_loss(
        self,
        beliefs: torch.Tensor,
        next_actions: torch.Tensor,
        next_melds: torch.Tensor,
        next_tile_counts: torch.Tensor,
        use_auxiliary: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算辅助损失

        Args:
            beliefs: [batch, 3, 34] 当前对手信念
            next_actions: [batch, 3, 1] 对手下一轮动作类型（0=出牌, 1=吃, 2=碰, 3=杠）
            next_melds: [batch, 3, 1] 对手下一轮是否副露（0=否, 1=是）
            next_tile_counts: [batch, 3] 对手下一轮手牌数量
            use_auxiliary: 是否使用辅助损失

         Returns:
            total_loss: 总损失
            losses: 各项损失字典
        """
        if not use_auxiliary:
            return torch.tensor(0.0), {}

        batch_size = beliefs.size(0)
        num_opponents = beliefs.size(1)

        # 1. 动作预测损失：预测对手下一轮动作类型（4分类）
        action_logits = self._predict_action_from_belief(
            beliefs
        )  # [batch, num_opponents, 4]
        action_labels = next_actions.squeeze(
            -1
        ).long()  # [batch, num_opponents] - convert to long

        # Reshape 以匹配交叉熵目标形状
        action_logits_flat = action_logits.view(-1, 4)  # [batch*num_opponents, 4]
        action_labels_flat = action_labels.view(-1)  # [batch*num_opponents]
        action_loss = F.cross_entropy(action_logits_flat, action_labels_flat)

        # 2. 副露预测损失：预测对手是否吃/碰/杠（4分类）
        meld_labels = next_melds.squeeze(-1).long()  # [batch, num_opponents]
        meld_labels_flat = meld_labels.view(-1)  # [batch*num_opponents]
        meld_loss = F.cross_entropy(action_logits_flat, meld_labels_flat)

        # 3. 手牌数量预测损失（回归）
        tile_count_logits = self._predict_tile_count_from_belief(
            beliefs
        )  # [batch, num_opponents]
        tile_count_loss = F.mse_loss(tile_count_logits, next_tile_counts)

        # 组合辅助损失
        auxiliary_loss = 0.4 * action_loss + 0.3 * meld_loss + 0.3 * tile_count_loss

        losses = {
            "action_prediction": action_loss,
            "meld_prediction": meld_loss,
            "tile_count": tile_count_loss,
            "auxiliary": auxiliary_loss,
        }

        return auxiliary_loss, losses

    def _predict_action_from_belief(self, beliefs: torch.Tensor) -> torch.Tensor:
        """从信念中预测对手下一轮动作类型"""
        batch_size, num_opponents, _ = beliefs.shape

        # 计算熵（不确定性度量）
        entropy = -torch.sum(beliefs * torch.log(beliefs + 1e-8), dim=-1)  # [batch, 3]

        # 将熵映射到动作类型
        # 低熵表示手牌不确定，更可能出牌
        # 高熵表示手牌接近胡牌，可能杠或胡
        action_logits = torch.zeros(batch_size, num_opponents, 4)

        # 映射逻辑（简化）
        action_logits[:, :, 0] = 5.0 - entropy  # 出牌
        action_logits[:, :, 1] = entropy * 0.5  # 吃
        action_logits[:, :, 2] = entropy * 0.5  # 碰
        action_logits[:, :, 3] = 10.0 - entropy  # 杠

        return action_logits

    def _predict_tile_count_from_belief(self, beliefs: torch.Tensor) -> torch.Tensor:
        """从信念中预测手牌数量"""
        batch_size, num_opponents, _ = beliefs.shape

        # 预测：基于信念的期望手牌数量
        # 使用 softmax 期望
        expected_counts = torch.zeros(batch_size, num_opponents)

        for opp in range(num_opponents):
            # 计算每种牌的期望数量
            # 简化：总概率 * 13（标准手牌数）
            opp_belief = beliefs[:, opp, :]  # [batch, 34]
            total_prob = opp_belief.sum(dim=-1)  # [batch]
            expected_counts[:, opp] = total_prob * 13.0

        return expected_counts
