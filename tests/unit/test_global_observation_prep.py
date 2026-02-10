"""
Test that global_hand is correctly added to observations for centralized critic
"""
import pytest
import numpy as np
from src.drl.buffer import CentralizedRolloutBuffer


def test_add_global_hand_to_observation():
    """Test the function that adds global_hand to individual observations"""
    # Simulate 4 agents' local observations (what env.last() returns)
    agents_obs = [
        {"private_hand": np.zeros(34), "current_player": 0, "fan_counts": np.array([0, 1, 2, 3])},
        {"private_hand": np.ones(34), "current_player": 1, "fan_counts": np.array([1, 0, 2, 3])},
        {"private_hand": np.zeros(34), "current_player": 2, "fan_counts": np.array([2, 1, 0, 3])},
        {"private_hand": np.ones(34), "current_player": 3, "fan_counts": np.array([3, 2, 1, 0])},
    ]

    # Simulate global hand data (from build_global_observation)
    # Each agent's hand as one-hot: [34] per agent
    global_hands = np.array([
        np.eye(34)[0],  # Agent 0: tile 0
        np.eye(34)[1],  # Agent 1: tile 1
        np.eye(34)[2],  # Agent 2: tile 2
        np.eye(34)[3],  # Agent 3: tile 3
    ])  # Shape: [4, 34]

    # Flatten to [136] for global_hand (C-order: row by row)
    global_hand_flat = global_hands.flatten()  # [136] = [agent0[34], agent1[34], agent2[34], agent3[34]]

    # Function to add global_hand to each agent's observation
    def add_global_hand_to_obs(agent_obs, global_hand_flat):
        """Add global_hand field to agent observation"""
        obs_with_global = agent_obs.copy()
        obs_with_global["global_hand"] = global_hand_flat
        return obs_with_global

    # Add global_hand to each agent's observation
    enhanced_obs = [add_global_hand_to_obs(obs, global_hand_flat) for obs in agents_obs]

    # Verify each agent now has global_hand
    for i, obs in enumerate(enhanced_obs):
        assert "global_hand" in obs, f"Agent {i} missing global_hand"
        assert obs["global_hand"].shape == (136,), f"Agent {i} global_hand wrong shape"
        # Verify the global_hand contains all 4 agents' hands
        # flatten() is row-major (C-order): agent0[34], agent1[34], agent2[34], agent3[34]
        # Each agent has a 1 at position i*34 in their own hand
        assert obs["global_hand"][i * 34 + i] == 1.0, f"Agent {i}'s hand should have 1 at position {i * 34 + i}"

    print(f"[PASS] global_hand correctly added to all agents' observations")


def test_global_hand_integration_with_buffer():
    """Test that observations with global_hand can be stored and retrieved"""
    from src.drl.buffer import CentralizedRolloutBuffer

    buffer = CentralizedRolloutBuffer(capacity=1000)

    # Create mock observations WITH global_hand
    global_hand_flat = np.random.randn(136)

    for agent_idx in range(4):
        obs = {
            "global_hand": global_hand_flat,  # [136]
            "private_hand": np.random.randn(34),
            "discard_pool_total": np.random.randn(34),
            "wall": np.random.randn(82),
            "melds": {
                "action_types": np.zeros(16, dtype=np.int64),
                "tiles": np.zeros(256, dtype=np.int64),
                "group_indices": np.zeros(32, dtype=np.int64),
            },
            "action_history": {
                "types": np.zeros(80, dtype=np.int64),
                "params": np.zeros(80, dtype=np.int64),
                "players": np.zeros(80, dtype=np.int64),
            },
            "special_gangs": np.zeros(12),
            "current_player": np.array([[agent_idx]]),  # Fix: [[x]] not [x]
            "fan_counts": np.random.randn(4),
            "special_indicators": np.random.randn(2),
            "remaining_tiles": np.array([[100]]),  # Fix: [[x]] not [x]
            "dealer": np.array([[0]]),  # Fix: [[x]] not [x]
            "current_phase": np.array([[0]]),  # Fix: [[x]] not [x]
        }

        buffer.add(
            obs=obs,
            action_mask=np.ones(145),
            action_type=0,
            action_param=-1,
            log_prob=0.25,
            reward=1.0,
            value=0.5,
            all_observations=[obs for _ in range(4)],  # Each agent gets the full obs
            done=False,
            agent_idx=agent_idx,
        )

    buffer.finish_episode()

    # Retrieve and verify
    batch = buffer.get_centralized_batch(batch_size=1, device="cpu")
    batch_all_obs, _, _, _, _, _ = batch

    # Check that observations have global_hand
    episode_obs = batch_all_obs[0]
    step_obs = episode_obs[0]  # First step
    agent_obs = step_obs[0]  # First agent

    assert "global_hand" in agent_obs, "Retrieved observation missing global_hand"
    assert agent_obs["global_hand"].shape == (136,), "global_hand has wrong shape"

    print(f"[PASS] Observations with global_hand can be stored and retrieved")
