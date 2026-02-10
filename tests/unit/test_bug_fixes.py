"""
Test P0 bug fixes
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock

from src.drl.monte_carlo_sampler import MonteCarloSampler


class TestMonteCarloSamplerFix:
    """Test MonteCarloSampler variable name fix"""

    def test_build_sampled_context_processes_all_opponents(self):
        """Test that all 3 opponents are processed"""
        # Create mock context with 4 players
        mock_context = Mock()
        mock_players = []
        for i in range(4):
            player = Mock()
            player.hand_tiles = [0] * 13  # 13 tiles
            mock_players.append(player)
        mock_context.players = mock_players

        # Create sampler
        sampler = MonteCarloSampler(n_samples=1)

        # Create sampled indices tensor [batch=1, 3 opponents, 34 tile types]
        sampled_indices = torch.zeros(1, 3, 34, dtype=torch.long)
        sampled_indices[0, 0, 0] = 1  # Opponent 0 has tile 0
        sampled_indices[0, 1, 1] = 1  # Opponent 1 has tile 1
        sampled_indices[0, 2, 2] = 1  # Opponent 2 has tile 2

        # Build sampled context
        result = sampler._build_sampled_context(mock_context, sampled_indices, batch_idx=0)

        # Verify all 3 opponents were processed
        # (If the bug exists, this will raise NameError: 'opp' is not defined)
        assert result is not None
        assert len(result.players) == 4


class TestMAPPOCentralizedCriticOptimizer:
    """Test MAPPO uses correct optimizer for centralized critic"""

    def test_centralized_critic_optimizer_exists(self):
        """Test that centralized_critic_optimizer is created when centralized_critic is provided"""
        from src.drl.mappo import MAPPO
        from src.drl.network import ActorCriticNetwork, CentralizedCriticNetwork

        device = "cpu"
        actor_net = ActorCriticNetwork(hidden_dim=128).to(device)
        critic_net = CentralizedCriticNetwork(hidden_dim=128).to(device)

        mappo = MAPPO(
            network=actor_net,
            centralized_critic=critic_net,
            device=device,
        )

        # Verify centralized_critic_optimizer exists
        assert mappo.centralized_critic_optimizer is not None
        assert isinstance(mappo.centralized_critic_optimizer, torch.optim.Adam)


class TestObservationEncoderNoBeliefAttribute:
    """Test ObservationEncoder doesn't reference undefined attributes"""

    def test_observation_encoder_no_belief_attributes(self):
        """Test that ObservationEncoder class doesn't have use_belief attribute"""
        from src.drl.network import ObservationEncoder

        encoder = ObservationEncoder(hidden_dim=128)

        # Verify that use_belief attribute is not defined
        # (If the incomplete belief code still exists, it would try to access this)
        assert not hasattr(encoder, "use_belief"), "ObservationEncoder should not have use_belief attribute"

    def test_observation_encoder_forward_signature(self):
        """Test that ObservationEncoder.forward() only accepts obs parameter"""
        from src.drl.network import ObservationEncoder
        import inspect

        encoder = ObservationEncoder(hidden_dim=128)

        # Check forward signature - should only accept 'obs' parameter
        sig = inspect.signature(encoder.forward)
        params = list(sig.parameters.keys())

        # Should only have 'obs' (bound method doesn't include 'self')
        assert params == ["obs"], f"Expected ['obs'], got {params}"
