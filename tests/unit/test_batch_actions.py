"""单元测试：批量动作选择"""
import pytest
import numpy as np
from src.drl.config import get_default_config
from src.drl.nfsp import NFSP
from src.drl.agent import NFSPAgentPool


def _create_test_observation():
    """创建测试用观测字典"""
    return {
        'private_hand': np.zeros(34, dtype=np.int64),
        'global_hand': np.zeros(136, dtype=np.int64),
        'discard_pool_total': np.zeros(34, dtype=np.int64),
        'wall': np.zeros(82, dtype=np.int64),
        'melds': {
            'action_types': np.zeros(16, dtype=np.int64),
            'tiles': np.zeros(256, dtype=np.int64),
            'group_indices': np.zeros(32, dtype=np.int64),
        },
        'action_history': {
            'types': np.zeros(80, dtype=np.int64),
            'params': np.zeros(80, dtype=np.int64),
            'players': np.zeros(80, dtype=np.int64),
        },
        'special_gangs': np.zeros(12, dtype=np.int64),
        'current_player': np.zeros(1, dtype=np.int64),
        'fan_counts': np.zeros(4, dtype=np.int64),
        'special_indicators': np.zeros(2, dtype=np.int64),
        'remaining_tiles': np.zeros(1, dtype=np.int64),
        'dealer': np.zeros(1, dtype=np.int64),
        'current_phase': np.zeros(1, dtype=np.int64),
    }


def test_nfsp_select_actions_batch_returns_correct_shapes():
    """测试 NFSP 批量动作选择返回正确的形状"""
    config = get_default_config()
    nfsp = NFSP(config, device="cpu")

    batch_size = 4
    obs_batch = [_create_test_observation() for _ in range(batch_size)]
    mask_batch = [np.ones(145, dtype=np.int64) for _ in range(batch_size)]

    actions_type, actions_param, log_probs, values = nfsp.select_actions_batch(
        obs_batch, mask_batch
    )

    assert actions_type.shape == (batch_size,)
    assert actions_param.shape == (batch_size,)
    assert log_probs.shape == (batch_size,)
    assert values.shape == (batch_size,)


def test_nfsp_select_actions_batch_with_best_response():
    """测试强制使用最佳响应网络的批量选择"""
    config = get_default_config()
    nfsp = NFSP(config, device="cpu")

    obs_batch = [_create_test_observation() for _ in range(2)]
    mask_batch = [np.ones(145, dtype=np.int64) for _ in range(2)]

    actions_type, actions_param, log_probs, values = nfsp.select_actions_batch(
        obs_batch, mask_batch, use_best_response=True
    )

    # 验证所有动作都来自最佳响应网络（log_prob 和 value 不为0）
    assert np.all(log_probs > 0)
    assert np.all(values > 0)


def test_nfsp_select_actions_batch_with_average_policy():
    """测试强制使用平均策略的批量选择"""
    config = get_default_config()
    nfsp = NFSP(config, device="cpu")

    obs_batch = [_create_test_observation() for _ in range(2)]
    mask_batch = [np.ones(145, dtype=np.int64) for _ in range(2)]

    actions_type, actions_param, log_probs, values = nfsp.select_actions_batch(
        obs_batch, mask_batch, use_best_response=False
    )

    # 验证所有动作都来自平均策略网络（log_prob 和 value 应该为0）
    assert np.all(log_probs == 0)
    assert np.all(values == 0)


def test_agent_pool_select_actions_batch():
    """测试 NFSPAgentPool 批量动作选择"""
    config = get_default_config()
    pool = NFSPAgentPool(config=config, device="cpu", num_agents=4, share_parameters=True)

    batch_size = 2
    obs_batch = [_create_test_observation() for _ in range(batch_size)]
    mask_batch = [np.ones(145, dtype=np.int64) for _ in range(batch_size)]

    actions_type, actions_param, log_probs, values = pool.select_actions_batch(
        obs_batch, mask_batch
    )

    assert actions_type.shape == (batch_size,)
    assert actions_param.shape == (batch_size,)
    assert log_probs.shape == (batch_size,)
    assert values.shape == (batch_size,)


def test_agent_pool_select_actions_batch_fallback():
    """测试非共享参数模式的批量动作选择（降级方案）"""
    config = get_default_config()
    pool = NFSPAgentPool(config=config, device="cpu", num_agents=4, share_parameters=False)

    batch_size = 2
    obs_batch = [_create_test_observation() for _ in range(batch_size)]
    mask_batch = [np.ones(145, dtype=np.int64) for _ in range(batch_size)]

    actions_type, actions_param, log_probs, values = pool.select_actions_batch(
        obs_batch, mask_batch
    )

    # 非共享模式应该返回结果（虽然不是最优）
    assert actions_type is not None
    assert actions_type.shape == (batch_size,)
