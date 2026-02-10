"""
测试配置文件
"""

import pytest
import numpy as np
import torch
from typing import Generator


@pytest.fixture
def sample_observation():
    """创建测试用的观测样本"""
    return {
        "global_hand": np.random.randint(0, 2, size=(4, 34)).astype(np.float32),
        "private_hand": np.random.randint(0, 2, size=(34,)).astype(np.float32),
        "discard_pool_total": np.random.randint(0, 5, size=(34,)).astype(np.float32),
        "wall": np.random.randint(0, 5, size=(82,)).astype(np.float32),
        "melds": {
            "action_types": np.zeros(16, dtype=np.int8),
            "tiles": np.zeros(256, dtype=np.int8),
            "group_indices": np.zeros(32, dtype=np.int8),
        },
        "action_history": {
            "types": np.zeros(80, dtype=np.int8),
            "params": np.zeros(80, dtype=np.int8),
            "players": np.zeros(80, dtype=np.int8),
        },
        "special_gangs": np.zeros(12, dtype=np.int8),
        "current_player": np.array([0], dtype=np.int8),
        "fan_counts": np.zeros(4, dtype=np.int16),
        "special_indicators": np.array([33, 33], dtype=np.int8),
        "remaining_tiles": np.array(100, dtype=np.int8),
        "dealer": np.array([0], dtype=np.int8),
        "current_phase": np.array([1], dtype=np.int8),
        "action_mask": np.ones(145, dtype=np.int8),
    }


@pytest.fixture
def batch_size():
    """测试批次大小"""
    return 4


@pytest.fixture
def hidden_dim():
    """网络隐藏层维度"""
    return 256
