"""
Test that get_centralized_batch works correctly after index fix
"""
import pytest
import numpy as np
from src.drl.buffer import CentralizedRolloutBuffer


def test_returns_numpy_arrays():
    """Verify get_centralized_batch returns numpy arrays (not lists)"""
    buffer = CentralizedRolloutBuffer(capacity=1000)

    # Add 1 episode
    for step in range(3):
        for agent_idx in range(4):
            buffer.add(
                obs={"step": step, "agent": agent_idx},
                action_mask=np.ones(145),
                action_type=step,
                action_param=step,
                log_prob=0.25,
                reward=1.0,
                value=0.5,
                all_observations=[{"step": step} for _ in range(4)],
                done=(step == 2),
                agent_idx=agent_idx,
            )

    buffer.finish_episode()

    # Get batch
    batch_all_obs, batch_actions_type, batch_actions_param, batch_rewards, batch_values, batch_dones = buffer.get_centralized_batch(
        batch_size=1, device="cpu"
    )

    # Verify numpy arrays
    assert isinstance(batch_actions_type, np.ndarray), "actions_type should be numpy array"
    assert isinstance(batch_actions_param, np.ndarray), "actions_param should be numpy array"
    assert isinstance(batch_rewards, np.ndarray), "rewards should be numpy array"
    assert isinstance(batch_dones, np.ndarray), "dones should be numpy array"
    assert isinstance(batch_values, np.ndarray), "values should be numpy array"

    # Verify shape
    assert batch_actions_type.shape == (1, 4, 3), f"Expected shape (1, 4, 3), got {batch_actions_type.shape}"

    print(f"\n[PASS] Returns numpy arrays with correct shapes!")



def test_get_centralized_batch_after_fix():
    """Verify get_centralized_batch works with correct indexing"""
    buffer = CentralizedRolloutBuffer(capacity=1000)

    # Add 2 episodes with 4 agents and 3 steps each
    for episode_idx in range(2):
        # Clear current data for new episode
        buffer.current_obs = [[] for _ in range(4)]
        buffer.current_action_masks = [[] for _ in range(4)]
        buffer.current_actions_type = [[] for _ in range(4)]
        buffer.current_actions_param = [[] for _ in range(4)]
        buffer.current_log_probs = [[] for _ in range(4)]
        buffer.current_rewards = [[] for _ in range(4)]
        buffer.current_values = [[] for _ in range(4)]
        buffer.current_dones = [[] for _ in range(4)]

        for step in range(3):
            for agent_idx in range(4):
                buffer.add(
                    obs={"episode": episode_idx, "step": step, "agent": agent_idx},
                    action_mask=np.ones(145),
                    action_type=step + agent_idx,
                    action_param=step * 10 + agent_idx,
                    log_prob=0.25,
                    reward=episode_idx * 100 + step * 10 + agent_idx,
                    value=episode_idx * 1000 + step * 100 + agent_idx,
                    all_observations=[{"episode": episode_idx, "step": step, "from": i} for i in range(4)],
                    done=(step == 2),
                    agent_idx=agent_idx,
                )

        # Finish episode
        buffer.finish_episode()

    # Now test get_centralized_batch
    print(f"\n=== Testing get_centralized_batch ===")
    print(f"Episodes in buffer: {len(buffer.episodes)}")

    # This should NOT crash anymore
    batch_all_obs, batch_actions_type, batch_actions_param, batch_rewards, batch_values, batch_dones = buffer.get_centralized_batch(
        batch_size=2, device="cpu"
    )

    print(f"Batch size: {len(batch_all_obs)}")
    print(f"batch_all_obs structure: {len(batch_all_obs)} episodes")

    # Verify structure
    assert len(batch_all_obs) == 2, "Should have 2 episodes in batch"
    assert len(batch_actions_type) == 2, "Should have 2 episodes of actions_type"
    assert len(batch_rewards) == 2, "Should have 2 episodes of rewards"

    # Verify each episode has correct structure
    for ep_idx in range(2):
        # batch_actions_type[ep_idx] should be [4, num_steps]
        episode_actions = batch_actions_type[ep_idx]
        print(f"Episode {ep_idx}: actions_type structure is {len(episode_actions)} agents")
        assert len(episode_actions) == 4, f"Episode {ep_idx} should have 4 agents"

        for agent_idx in range(4):
            agent_actions = episode_actions[agent_idx]
            print(f"  Agent {agent_idx}: {len(agent_actions)} steps")
            assert len(agent_actions) == 3, f"Agent {agent_idx} should have 3 steps"

    print(f"\n[PASS] get_centralized_batch works correctly!")


def test_get_centralized_batch_single_episode():
    """Test with single episode"""
    buffer = CentralizedRolloutBuffer(capacity=1000)

    # Add 1 episode
    for step in range(5):
        for agent_idx in range(4):
            buffer.add(
                obs={"step": step, "agent": agent_idx},
                action_mask=np.ones(145),
                action_type=step,
                action_param=step,
                log_prob=0.25,
                reward=1.0,
                value=0.5,
                all_observations=[{"step": step} for _ in range(4)],
                done=(step == 4),
                agent_idx=agent_idx,
            )

    buffer.finish_episode()

    # Get batch
    batch_all_obs, batch_actions_type, batch_actions_param, batch_rewards, batch_values, batch_dones = buffer.get_centralized_batch(
        batch_size=1, device="cpu"
    )

    assert len(batch_all_obs) == 1
    assert len(batch_actions_type[0]) == 4  # 4 agents
    assert len(batch_actions_type[0][0]) == 5  # 5 steps

    print(f"\n[PASS] Single episode test passed!")
