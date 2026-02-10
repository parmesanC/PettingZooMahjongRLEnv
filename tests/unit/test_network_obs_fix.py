"""
Tests for network and observation preprocessing fixes
"""
import pytest
import numpy as np
import torch
from src.drl.network import StateEncoder


# Test helper function that mimics _prepare_obs logic
def _prepare_obs(obs: dict, device="cpu") -> dict:
    """Mimics MAPPO._prepare_obs for testing"""
    tensor_obs = {}
    for key, value in obs.items():
        if isinstance(value, dict):
            tensor_obs[key] = _prepare_obs(value, device)
        elif isinstance(value, np.ndarray):
            tensor_val = torch.FloatTensor(value).to(device)
            if tensor_val.dim() == 1:
                tensor_val = tensor_val.unsqueeze(0)
            elif tensor_val.dim() == 0:
                tensor_val = tensor_val.unsqueeze(0).unsqueeze(0)
            tensor_obs[key] = tensor_val
        else:
            tensor_val = torch.FloatTensor([value]).unsqueeze(0).to(device)
            tensor_obs[key] = tensor_val
    return tensor_obs


def test_state_encoder_input_dim_is_10():
    """Verify StateEncoder input dimension is 10, not 13"""
    encoder = StateEncoder()
    assert encoder.net[0].in_features == 10, f"Expected 10, got {encoder.net[0].in_features}"
    print("[PASS] StateEncoder input dimension is 10")


def test_prepare_obs_scalar_handling():
    """Verify scalar values are converted to [1, 1]"""
    # Test with scalar values (like remaining_tiles, current_phase)
    obs = {
        "remaining_tiles": 100,  # Python int
        "current_phase": 1,       # Python int
    }

    tensor_obs = _prepare_obs(obs)

    # Check dimensions
    assert tensor_obs["remaining_tiles"].shape == (1, 1), \
        f"remaining_tiles shape is {tensor_obs['remaining_tiles'].shape}, expected (1, 1)"
    assert tensor_obs["current_phase"].shape == (1, 1), \
        f"current_phase shape is {tensor_obs['current_phase'].shape}, expected (1, 1)"

    print("[PASS] Scalar values correctly converted to [1, 1]")


def test_prepare_obs_1d_array_handling():
    """Verify 1D arrays are converted to [1, n]"""
    # Test with 1D arrays (like current_player, dealer)
    obs = {
        "current_player": np.array([0]),  # [1]
        "dealer": np.array([1]),           # [1]
        "fan_counts": np.array([0, 1, 2, 3]),  # [4]
    }

    tensor_obs = _prepare_obs(obs)

    # Check dimensions
    assert tensor_obs["current_player"].shape == (1, 1), \
        f"current_player shape is {tensor_obs['current_player'].shape}, expected (1, 1)"
    assert tensor_obs["dealer"].shape == (1, 1), \
        f"dealer shape is {tensor_obs['dealer'].shape}, expected (1, 1)"
    assert tensor_obs["fan_counts"].shape == (1, 4), \
        f"fan_counts shape is {tensor_obs['fan_counts'].shape}, expected (1, 4)"

    print("[PASS] 1D arrays correctly converted to [1, n]")


def test_state_encoder_forward_with_correct_dims():
    """Verify StateEncoder forward works with 10-dimensional input"""
    encoder = StateEncoder()

    # Create inputs with correct dimensions
    batch_size = 2
    current_player = torch.tensor([[0], [1]])        # [2, 1]
    remaining_tiles = torch.tensor([[100], [50]])    # [2, 1]
    fan_counts = torch.tensor([[0,1,2,3], [1,2,3,4]]) # [2, 4]
    current_phase = torch.tensor([[0], [1]])         # [2, 1]
    special_indicators = torch.tensor([[0,1], [1,0]]) # [2, 2]
    dealer = torch.tensor([[0], [1]])                # [2, 1]

    # Forward should work without errors
    output = encoder(
        current_player,
        remaining_tiles,
        fan_counts,
        current_phase,
        special_indicators,
        dealer
    )

    assert output.shape == (batch_size, 32), \
        f"Output shape is {output.shape}, expected ({batch_size}, 32)"

    print(f"[PASS] StateEncoder forward works with 10-dim input, output shape: {output.shape}")

