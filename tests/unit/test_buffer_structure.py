"""
Test to verify buffer data structure and identify indexing issues
"""
import pytest
import numpy as np
from src.drl.buffer import CentralizedRolloutBuffer


def test_buffer_data_structure():
    """Verify the actual data structure stored in buffer"""
    buffer = CentralizedRolloutBuffer(capacity=1000)

    # Add data for 4 agents over 3 steps
    for step in range(3):
        for agent_idx in range(4):
            buffer.add(
                obs={"step": step, "agent": agent_idx},
                action_mask=np.ones(145),
                action_type=0,
                action_param=-1,
                log_prob=0.25,
                reward=1.0,
                value=0.5,
                all_observations=[{"step": step, "from": i} for i in range(4)],
                done=(step == 2),
                agent_idx=agent_idx,
            )

    # Finish episode to package data
    episode = buffer.finish_episode()

    # Verify structure: should be [4, num_steps]
    print(f"\n=== Buffer Structure Analysis ===")
    print(f"observations structure: {type(episode['observations'])}")
    print(f"observations length: {len(episode['observations'])}")

    for agent_idx in range(4):
        agent_obs = episode["observations"][agent_idx]
        print(f"  Agent {agent_idx}: {len(agent_obs)} observations")
        if agent_obs:
            print(f"    First obs: {agent_obs[0]}")

    # Verify each agent has correct data
    assert len(episode["observations"]) == 4, "Should have 4 agents"
    assert len(episode["observations"][0]) == 3, "Agent 0 should have 3 steps"
    assert episode["observations"][0][0]["agent"] == 0, "Agent 0's obs should have agent=0"

    # Verify the WRONG indexing would fail
    print(f"\n=== Testing Index Access ===")
    num_steps = 3

    # This should FAIL (current buggy code):
    print(f"Testing WRONG indexing: episode['observations'][step_idx][agent_idx]")
    try:
        for step_idx in range(num_steps):
            for agent_idx in range(4):
                obs = episode["observations"][step_idx][agent_idx]
        print(f"  Wrong indexing PASSED (unexpected!)")
        assert False, "Wrong indexing should have failed!"
    except (IndexError, KeyError) as e:
        print(f"  Wrong indexing FAILED as expected: {e}")

    # This should WORK (correct indexing):
    print(f"Testing CORRECT indexing: episode['observations'][agent_idx][step_idx]")
    for agent_idx in range(4):
        for step_idx in range(num_steps):
            obs = episode["observations"][agent_idx][step_idx]
            print(f"  Agent {agent_idx}, Step {step_idx}: {obs}")
    print(f"  Correct indexing PASSED")


def test_all_fields_structure():
    """Verify all fields have the same [4, num_steps] structure"""
    buffer = CentralizedRolloutBuffer(capacity=1000)

    # Add data
    for step in range(2):
        for agent_idx in range(4):
            buffer.add(
                obs={"test": step},
                action_mask=np.ones(145),
                action_type=step + agent_idx,
                action_param=step * 10 + agent_idx,
                log_prob=0.25,
                reward=step + agent_idx * 0.1,
                value=step * 0.5,
                all_observations=[{"step": step} for _ in range(4)],
                done=(step == 1),
                agent_idx=agent_idx,
            )

    episode = buffer.finish_episode()

    # All fields should have [4, num_steps] structure
    fields = ["observations", "action_masks", "actions_type", "actions_param", "log_probs", "rewards", "values"]

    print(f"\n=== All Fields Structure ===")
    for field in fields:
        if field not in episode or not episode[field]:
            print(f"{field}: SKIPPED (empty or missing)")
            continue

        field_data = episode[field]
        print(f"{field}: length={len(field_data)}, each={len(field_data[0]) if field_data else 0}")

        # Verify structure is [4, num_steps]
        assert len(field_data) == 4, f"{field} should have 4 agents"
        assert len(field_data[0]) == 2, f"{field} agent 0 should have 2 steps"
