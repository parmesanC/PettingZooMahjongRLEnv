"""
Test action_mask validation in WuhanMahjongEnv

This test verifies that the environment correctly validates agent actions
against the action_mask and returns negative rewards for invalid actions.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from example_mahjong_env import WuhanMahjongEnv
from src.mahjong_rl.core.constants import ActionType
from src.mahjong_rl.core.mahjong_action import MahjongAction


def test_action_to_index():
    """Test _action_to_index method maps actions correctly"""
    print("Testing _action_to_index method...")

    env = WuhanMahjongEnv(render_mode=None, enable_logging=False)

    # Test DISCARD (0-33)
    action = MahjongAction(ActionType.DISCARD, 0)
    assert env._action_to_index(action) == 0, "DISCARD(0) should map to index 0"

    action = MahjongAction(ActionType.DISCARD, 33)
    assert env._action_to_index(action) == 33, "DISCARD(33) should map to index 33"

    # Test CHOW (34-36)
    action = MahjongAction(ActionType.CHOW, 0)  # LEFT
    assert env._action_to_index(action) == 34, "CHOW(0) should map to index 34"

    action = MahjongAction(ActionType.CHOW, 2)  # RIGHT
    assert env._action_to_index(action) == 36, "CHOW(2) should map to index 36"

    # Test PONG (37)
    action = MahjongAction(ActionType.PONG, 0)
    assert env._action_to_index(action) == 37, "PONG should map to index 37"

    # Test KONG_EXPOSED (38)
    action = MahjongAction(ActionType.KONG_EXPOSED, 0)
    assert env._action_to_index(action) == 38, "KONG_EXPOSED should map to index 38"

    # Test KONG_SUPPLEMENT (39-72)
    action = MahjongAction(ActionType.KONG_SUPPLEMENT, 0)
    assert env._action_to_index(action) == 39, "KONG_SUPPLEMENT(0) should map to index 39"

    action = MahjongAction(ActionType.KONG_SUPPLEMENT, 33)
    assert env._action_to_index(action) == 72, "KONG_SUPPLEMENT(33) should map to index 72"

    # Test KONG_CONCEALED (73-106)
    action = MahjongAction(ActionType.KONG_CONCEALED, 0)
    assert env._action_to_index(action) == 73, "KONG_CONCEALED(0) should map to index 73"

    action = MahjongAction(ActionType.KONG_CONCEALED, 33)
    assert env._action_to_index(action) == 106, "KONG_CONCEALED(33) should map to index 106"

    # Test KONG_RED (107)
    action = MahjongAction(ActionType.KONG_RED, 0)
    assert env._action_to_index(action) == 107, "KONG_RED should map to index 107"

    # Test KONG_SKIN (108-141)
    action = MahjongAction(ActionType.KONG_SKIN, 0)
    assert env._action_to_index(action) == 108, "KONG_SKIN(0) should map to index 108"

    action = MahjongAction(ActionType.KONG_SKIN, 33)
    assert env._action_to_index(action) == 141, "KONG_SKIN(33) should map to index 141"

    # Test KONG_LAZY (142)
    action = MahjongAction(ActionType.KONG_LAZY, 0)
    assert env._action_to_index(action) == 142, "KONG_LAZY should map to index 142"

    # Test WIN (143)
    action = MahjongAction(ActionType.WIN, 0)
    assert env._action_to_index(action) == 143, "WIN should map to index 143"

    # Test PASS (144)
    action = MahjongAction(ActionType.PASS, 0)
    assert env._action_to_index(action) == 144, "PASS should map to index 144"

    print("[OK] All _action_to_index tests passed!")


def test_is_action_mask_valid():
    """Test _is_action_mask_valid method"""
    print("\nTesting _is_action_mask_valid method...")

    env = WuhanMahjongEnv(render_mode=None, enable_logging=False)

    # Create a test action_mask with only DISCARD(0) and PASS enabled
    action_mask = np.zeros(145, dtype=np.int8)
    action_mask[0] = 1   # DISCARD tile 0
    action_mask[144] = 1  # PASS

    # Test valid action: DISCARD(0)
    action = MahjongAction(ActionType.DISCARD, 0)
    assert env._is_action_mask_valid(action, action_mask), "DISCARD(0) should be valid"

    # Test invalid action: DISCARD(1) - not in mask
    action = MahjongAction(ActionType.DISCARD, 1)
    assert not env._is_action_mask_valid(action, action_mask), "DISCARD(1) should be invalid"

    # Test valid action: PASS
    action = MahjongAction(ActionType.PASS, 0)
    assert env._is_action_mask_valid(action, action_mask), "PASS should be valid"

    # Test invalid action: WIN - not in mask
    action = MahjongAction(ActionType.WIN, 0)
    assert not env._is_action_mask_valid(action, action_mask), "WIN should be invalid"

    # Test with all actions enabled
    full_mask = np.ones(145, dtype=np.int8)
    action = MahjongAction(ActionType.WIN, 0)
    assert env._is_action_mask_valid(action, full_mask), "WIN should be valid with full mask"

    print("[OK] All _is_action_mask_valid tests passed!")


def test_step_rejects_invalid_action():
    """Test that step() rejects actions not in action_mask"""
    print("\nTesting step() validation...")

    env = WuhanMahjongEnv(render_mode=None, enable_logging=False)
    obs, info = env.reset(seed=42)

    # Get a valid action from the mask
    valid_actions = np.where(obs['action_mask'] == 1)[0]
    print(f"  Found {len(valid_actions)} valid actions")

    if len(valid_actions) > 0:
        # Test with an invalid action (choose an action not in mask)
        # Find an action index that's NOT in the mask
        invalid_index = -1
        for i in range(145):
            if i not in valid_actions:
                invalid_index = i
                break

        if invalid_index >= 0:
            # Convert index to action
            # For simplicity, test with DISCARD action
            if invalid_index < 34:
                action = (ActionType.DISCARD.value, invalid_index)
                print(f"  Testing invalid action: DISCARD({invalid_index})")

                obs, reward, terminated, truncated, info = env.step(action)

                # Should return negative reward
                assert reward == -1.0, f"Expected -1.0 reward, got {reward}"
                assert 'error' in info, "Expected error in info"
                assert info['error'] == 'action not in mask', f"Expected 'action not in mask', got '{info['error']}'"
                assert not terminated, "Should not terminate on invalid action"
                print(f"  [OK] Invalid action correctly rejected with reward={reward}")
            else:
                print(f"  [WARNING] Could not find invalid DISCARD action to test")
        else:
            print(f"  [WARNING] All actions are valid (mask is full)")

    print("[OK] step() validation test completed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Action Mask Validation Tests")
    print("=" * 60)

    try:
        test_action_to_index()
        test_is_action_mask_valid()
        test_step_rejects_invalid_action()

        print("\n" + "=" * 60)
        print("All tests passed! [OK]")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
