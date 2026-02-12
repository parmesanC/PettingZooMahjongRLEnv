"""
Cache optimization verification tests (ASCII-only version)

Verifies cache components are created and working.
"""

import sys
import os

# Set project path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from example_mahjong_env import WuhanMahjongEnv


def test_cache_components():
    """
    Test: Verify cache components are created correctly

    Ensures all cache components are initialized after env.reset().
    """
    print("\n### Test: Cache components creation ###")

    env = WuhanMahjongEnv(render_mode=None, training_phase=1, enable_logging=False)
    obs, _ = env.reset()

    # Verify environment-level cache components
    assert env._cached_validator is not None, "Cached validator should be created"
    assert env._cached_win_checker is not None, "Cached win_checker should be created"
    assert env._cached_mask_cache is not None, "Cached mask_cache should be created"

    # Verify state machine received cached components
    assert env.state_machine._cached_validator is not None, "State machine should have cached validator"
    assert env.state_machine._cached_win_checker is not None, "State machine should have cached win checker"

    # Verify observation builder received cached validator
    assert env.state_machine.observation_builder._cached_validator is not None, \
        "ObservationBuilder should have cached validator"

    print("[PASS] All cache components created correctly")

    # Verify cache stats are accessible
    cache_stats = env._cached_mask_cache.stats()
    assert "hits" in cache_stats, "Cache stats should have 'hits'"
    assert "misses" in cache_stats, "Cache stats should have 'misses'"
    assert "hit_rate" in cache_stats, "Cache stats should have 'hit_rate'"

    print("[PASS] Cache stats functionality normal")


def test_basic_gameplay():
    """
    Test: Run one episode to verify no crashes
    """
    print("\n### Test: Basic gameplay ###")

    env = WuhanMahjongEnv(render_mode=None, training_phase=1, enable_logging=False)
    obs, _ = env.reset()

    steps = 0
    max_steps = 20  # Limit steps for quick test

    for agent_name in env.agent_iter():
        if steps >= max_steps:
            break

        obs, reward, terminated, truncated, info = env.last()
        action_mask = obs["action_mask"]

        # Find first valid action
        valid_actions = [i for i, v in enumerate(action_mask) if v > 0]
        if not valid_actions:
            print("No valid actions available")
            break

        action = valid_actions[0]

        # Map action to (action_type, action_param)
        if 0 <= action < 34:  # DISCARD
            action_type = 0  # ActionType.DISCARD
            action_param = action
        elif 143 <= action <= 143:  # WIN
            action_type = 9  # ActionType.WIN
            action_param = -1
        elif 144 <= action <= 144:  # PASS
            action_type = 10  # ActionType.PASS
            action_param = -1
        else:
            # CRITICAL: Always call env.step() to advance game state
            print(f"[WARN] Unhandled action {action}, using PASS")
            action_type = 10  # ActionType.PASS
            action_param = -1

        # Always call env.step() to avoid infinite loop
        env.step((action_type, action_param))
        steps += 1

        if terminated or truncated:
            break

    print(f"[PASS] Ran {steps} steps without errors")


def run_all_tests():
    """Run all verification tests"""
    print("=" * 60)
    print("Cache Optimization Verification Tests")
    print("=" * 60)

    try:
        test_cache_components()
        test_basic_gameplay()

        print("\n" + "=" * 60)
        print("[SUCCESS] All tests passed")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n[ERROR] Test error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
