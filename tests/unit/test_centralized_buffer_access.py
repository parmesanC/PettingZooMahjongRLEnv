"""
Test centralized buffer access patterns
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from src.drl.config import get_default_config
from src.drl.agent import NFSPAgentPool


def test_agent_pool_has_centralized_buffer():
    """Test that NFSPAgentPool has centralized_buffer attribute"""
    config = get_default_config()
    pool = NFSPAgentPool(config=config, device="cpu")

    # Verify centralized_buffer exists
    assert hasattr(pool, "centralized_buffer"), "NFSPAgentPool should have centralized_buffer"
    assert pool.centralized_buffer is not None


def test_shared_nfsp_buffer_is_mixed_buffer():
    """Test that shared_nfsp.buffer is MixedBuffer without centralized_buffer"""
    from src.drl.buffer import MixedBuffer

    config = get_default_config()
    pool = NFSPAgentPool(config=config, device="cpu")

    # Verify shared_nfsp.buffer is MixedBuffer
    assert isinstance(pool.shared_nfsp.buffer, MixedBuffer)

    # Verify MixedBuffer does NOT have centralized_buffer
    assert not hasattr(pool.shared_nfsp.buffer, "centralized_buffer"), \
        "MixedBuffer should not have centralized_buffer attribute"


def test_shared_nfsp_has_own_centralized_buffer():
    """Test that NFSP has its own centralized_buffer"""
    config = get_default_config()
    pool = NFSPAgentPool(config=config, device="cpu")

    # Verify shared_nfsp has centralized_buffer
    assert hasattr(pool.shared_nfsp, "centralized_buffer")
    assert pool.shared_nfsp.centralized_buffer is not None


def test_trainer_uses_correct_buffer_access():
    """Test that trainer uses agent_pool.centralized_buffer not shared_nfsp.buffer.centralized_buffer"""
    config = get_default_config()
    pool = NFSPAgentPool(config=config, device="cpu")

    # This should work (correct path)
    pool.centralized_buffer.add_multi_agent(
        all_observations=[{}, {}, {}, {}],
        action_masks=[np.ones(145) for _ in range(4)],
        actions_type=[0, 0, 0, 0],
        actions_param=[-1, -1, -1, -1],
        log_probs=[0.25, 0.25, 0.25, 0.25],
        rewards=[0.0, 0.0, 0.0, 0.0],
        done=True,
    )

    # Call finish_episode to package data
    pool.centralized_buffer.finish_episode()

    # Verify data was added
    assert len(pool.centralized_buffer.episodes) == 1

    # Verify the WRONG path doesn't work
    try:
        pool.shared_nfsp.buffer.centralized_buffer.add_multi_agent(
            all_observations=[{}, {}, {}, {}],
            action_masks=[np.ones(145) for _ in range(4)],
            actions_type=[0, 0, 0, 0],
            actions_param=[-1, -1, -1, -1],
            log_probs=[0.25, 0.25, 0.25, 0.25],
            rewards=[0.0, 0.0, 0.0, 0.0],
            done=True,
        )
        assert False, "Should have raised AttributeError"
    except AttributeError:
        # Expected - MixedBuffer doesn't have centralized_buffer
        pass
