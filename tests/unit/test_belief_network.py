"""
BeliefNetwork 单元测试
"""

import pytest
import torch
import numpy as np
from src.drl.belief_network import BeliefNetwork


class TestBeliefNetwork:
    """测试 BeliefNetwork 类"""

    def test_network_initialization(self, hidden_dim=256, num_opponents=3):
        """测试网络初始化"""
        network = BeliefNetwork(hidden_dim=hidden_dim, num_opponents=num_opponents)
        assert network is not None
        assert network.hidden_dim == hidden_dim
        assert network.num_opponents == num_opponents
        assert len(network.opponent_beliefs) == num_opponents

    def test_forward_output_shape(self, sample_observation, hidden_dim=256):
        """测试 forward 输出形状"""
        network = BeliefNetwork(hidden_dim=hidden_dim)
        beliefs = network.forward(sample_observation)

        # 检查输出形状：[batch, num_opponents, 34]
        expected_shape = torch.Size([1, 3, 34])
        assert beliefs.shape == expected_shape

    def test_forward_normalization(self, sample_observation):
        """测试概率归一化（每个对手的概率和≈1.0）"""
        network = BeliefNetwork()
        beliefs = network.forward(sample_observation)

        # 检查每个对手的概率和是否接近1.0
        sums = beliefs.sum(dim=-1)  # [batch, num_opponents]
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

        # 检查所有概率在 [0, 1] 范围内
        assert (beliefs >= 0).all()
        assert (beliefs <= 1).all()

    def test_bayesian_update_reduces_discarded_tile_probability(
        self, sample_observation
    ):
        """测试贝叶斯更新降低打出牌的概率"""
        network = BeliefNetwork()

        # 初始信念
        initial_beliefs = network.forward(sample_observation)

        # 模拟打出5万（tile_id=5）
        action_history = torch.zeros(1, 1, 3)
        action_history[0, 0, :] = torch.tensor(
            [0, 0, 5]
        )  # [player, action_type, tile_id]

        discard_pool = sample_observation["discard_pool"].clone()
        discard_pool[0, 5] += 1  # 增加一张5万到弃牌池

        melds = torch.zeros(1, 16, 9)

        # 贝叶斯更新
        updated_beliefs = network.update_beliefs(
            initial_beliefs, action_history, discard_pool, melds
        )

        # 检查5万的概率降低了
        assert updated_beliefs[0, 0, 5] < initial_beliefs[0, 0, 5]

    def test_get_opponent_beliefs(self, sample_observation):
        """测试 get_opponent_beliefs 方法"""
        network = BeliefNetwork()

        # 构建测试上下文
        context = {
            "discard_pool": sample_observation["discard_pool"],
            "melds": sample_observation["melds"],
            "action_history": sample_observation["action_history"],
        }

        # 获取对手信念
        beliefs = network.get_opponent_beliefs(0, context)

        # 检查输出形状：[3, 34] 或 [batch, 3, 34]
        if len(beliefs.shape) == 2:
            assert beliefs.shape == torch.Size([3, 34])
        else:
            assert beliefs.shape[1:] == torch.Size([3, 34])
