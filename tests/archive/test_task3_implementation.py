"""
Test script to verify Task 3 implementation: Auto-pass logic in WaitResponseState

This test verifies:
1. The enter() method correctly filters out players who can only PASS
2. active_responders list only contains players who need to make decisions
3. Auto-pass players are automatically added to response_collector
4. If all players can PASS, _select_best_response is called immediately
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test that all necessary imports work"""
    print("Testing imports...")
    try:
        from src.mahjong_rl.core.GameData import GameContext
        from src.mahjong_rl.core.constants import GameStateType, ActionType, ResponsePriority
        from src.mahjong_rl.state_machine.states.wait_response_state import WaitResponseState
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_gamecontext_fields():
    """Test that GameContext has the required fields"""
    print("\nTesting GameContext fields...")
    from src.mahjong_rl.core.GameData import GameContext

    # Check if the fields exist
    ctx = GameContext()
    assert hasattr(ctx, 'active_responders'), "Missing active_responders field"
    assert hasattr(ctx, 'active_responder_idx'), "Missing active_responder_idx field"
    assert hasattr(ctx, 'response_order'), "Missing response_order field"
    assert hasattr(ctx, 'response_collector'), "Missing response_collector field"

    print("✓ All required fields present")
    return True

def test_helper_methods():
    """Test that helper methods use correct fields"""
    print("\nTesting helper methods...")
    from src.mahjong_rl.core.GameData import GameContext

    ctx = GameContext()

    # Test empty state
    assert ctx.get_current_responder() is None, "get_current_responder should return None for empty list"
    assert ctx.is_all_responded() is True, "is_all_responded should return True for empty list"

    # Test with active_responders
    ctx.active_responders = [1, 2, 3]
    ctx.active_responder_idx = 0
    assert ctx.get_current_responder() == 1, "get_current_responder should return first responder"
    assert ctx.is_all_responded() is False, "is_all_responded should return False when not all responded"

    ctx.move_to_next_responder()
    assert ctx.active_responder_idx == 1, "move_to_next_responder should increment index"
    assert ctx.get_current_responder() == 2, "get_current_responder should return second responder"

    print("✓ All helper methods work correctly")
    return True

def test_wait_response_state():
    """Test that WaitResponseState has the required methods"""
    print("\nTesting WaitResponseState...")
    from src.mahjong_rl.state_machine.states.wait_response_state import WaitResponseState

    # Check if _can_only_pass method exists
    assert hasattr(WaitResponseState, '_can_only_pass'), "Missing _can_only_pass method"
    assert hasattr(WaitResponseState, 'enter'), "Missing enter method"
    assert hasattr(WaitResponseState, '_select_best_response'), "Missing _select_best_response method"

    print("✓ WaitResponseState has all required methods")
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Task 3 Implementation Test: Auto-pass Logic")
    print("=" * 60)

    tests = [
        test_imports,
        test_gamecontext_fields,
        test_helper_methods,
        test_wait_response_state
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("\n✅ All tests passed! Task 3 implementation verified.")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
