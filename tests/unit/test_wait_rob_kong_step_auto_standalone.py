"""
Standalone unit tests for WaitRobKongState.step() method's 'auto' action handling

This test file can run without the C++ extension by importing only the necessary modules.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from unittest.mock import Mock, MagicMock, patch
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import GameStateType, ActionType
from src.mahjong_rl.core.mahjong_action import MahjongAction

# Import WaitRobKongState directly without going through __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "wait_rob_kong_state",
    "D:/DATA/Python_Project/Code/PettingZooRLENVMahjong/src/mahjong_rl/state_machine/states/wait_rob_kong_state.py"
)
wait_rob_kong_module = importlib.util.module_from_spec(spec)

# Mock the C++ import
sys.modules['mahjong_win_checker'] = MagicMock()

# Mock the base class
class MockGameState:
    def __init__(self, rule_engine, observation_builder):
        self.rule_engine = rule_engine
        self.observation_builder = observation_builder

    def build_observation(self, context):
        pass

    def validate_action(self, context, action, available_actions):
        return True

# Patch the base class import
wait_rob_kong_module.GameState = MockGameState

# Now load the module
spec.loader.exec_module(wait_rob_kong_module)
WaitRobKongState = wait_rob_kong_module.WaitRobKongState


def create_mock_context():
    """Create a mock GameContext with necessary attributes"""
    context = Mock(spec=GameContext)
    context.active_responders = []
    context.active_responder_idx = 0
    context.pending_responses = {}
    context.rob_kong_tile = 21  # 红中
    context.kong_player_idx = 0
    # Create players with hand_tiles as list (not Mock)
    context.players = []
    for i in range(4):
        player = Mock(spec=PlayerData)
        player.hand_tiles = []  # Real list, not Mock
        context.players.append(player)
    context.current_player_idx = 0
    context.current_state = GameStateType.WAIT_ROB_KONG
    return context


def test_step_with_auto_when_no_responders():
    """
    Test: step(context, 'auto') when active_responders is empty

    Expected: Should directly call _check_rob_kong_result() and return DRAWING_AFTER_GONG
    """
    print("\n[Test] step with 'auto' when no responders...")

    # Create state and context
    rule_engine = Mock()
    observation_builder = Mock()
    state = WaitRobKongState(rule_engine, observation_builder)
    context = create_mock_context()
    context.active_responders = []  # No one can rob kong

    # Mock _check_rob_kong_result to return DRAWING_AFTER_GONG
    state._check_rob_kong_result = Mock(return_value=GameStateType.DRAWING_AFTER_GONG)

    # Call step with 'auto'
    result = state.step(context, 'auto')

    # Verify: _check_rob_kong_result was called
    state._check_rob_kong_result.assert_called_once_with(context)
    # Verify: returned DRAWING_AFTER_GONG
    assert result == GameStateType.DRAWING_AFTER_GONG, \
        f"Expected DRAWING_AFTER_GONG, got {result}"

    print("[PASS] step correctly handles 'auto' when no responders")


def test_step_with_auto_when_has_responders_raises_error():
    """
    Test: step(context, 'auto') when active_responders is NOT empty

    Expected: Should raise ValueError with appropriate message
    """
    print("\n[Test] step with 'auto' when has responders should raise error...")

    # Create state and context
    rule_engine = Mock()
    observation_builder = Mock()
    state = WaitRobKongState(rule_engine, observation_builder)
    context = create_mock_context()
    context.active_responders = [1, 2]  # Some players can rob kong

    # Call step with 'auto' and expect ValueError
    try:
        state.step(context, 'auto')
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        error_msg = str(e)
        assert "Unexpected 'auto' action with active responders" in error_msg, \
            f"Wrong error message: {error_msg}"
        assert "should_auto_skip" in error_msg, \
            f"Error message should mention should_auto_skip: {error_msg}"
        print(f"[PASS] Correctly raised ValueError: {error_msg}")


def test_step_with_normal_pass_action():
    """
    Test: step(context, MahjongAction(PASS))

    Expected: Should process PASS action normally
    """
    print("\n[Test] step with normal PASS action...")

    # Create state and context
    rule_engine = Mock()
    observation_builder = Mock()
    state = WaitRobKongState(rule_engine, observation_builder)
    context = create_mock_context()
    context.active_responders = [1, 2]
    context.response_order = [1, 2]

    # Call step with PASS action
    pass_action = MahjongAction(ActionType.PASS, -1)
    result = state.step(context, pass_action)

    # Verify: PASS was recorded
    assert 1 in context.pending_responses, "Player 1's response should be recorded"
    assert context.pending_responses[1].action_type == ActionType.PASS
    # Verify: moved to next responder
    assert context.active_responder_idx == 1
    # Verify: still in WAIT_ROB_KONG state (more responders to process)
    assert result == GameStateType.WAIT_ROB_KONG

    print("[PASS] step correctly handles PASS action")


def test_step_with_normal_win_action():
    """
    Test: step(context, MahjongAction(WIN))

    Expected: Should process WIN action and return WIN state
    """
    print("\n[Test] step with normal WIN action...")

    # Create state and context
    rule_engine = Mock()
    observation_builder = Mock()
    state = WaitRobKongState(rule_engine, observation_builder)
    context = create_mock_context()
    context.active_responders = [1]
    context.response_order = [1]
    context.action_history = []

    # Call step with WIN action
    win_action = MahjongAction(ActionType.WIN, 21)  # 胡红中
    result = state.step(context, win_action)

    # Verify: WIN was recorded
    assert 1 in context.pending_responses, "Player 1's response should be recorded"
    assert context.pending_responses[1].action_type == ActionType.WIN
    # Verify: rob_kong_tile was added to winner's hand
    assert 21 in context.players[1].hand_tiles, "rob_kong_tile should be in winner's hand"
    # Verify: returned WIN state
    assert result == GameStateType.WIN
    assert context.is_win == True
    assert context.winner_ids == [1]

    print("[PASS] step correctly handles WIN action")


def test_step_with_invalid_action_type():
    """
    Test: step(context, "invalid_string")

    Expected: Should raise ValueError about invalid action type
    """
    print("\n[Test] step with invalid action type...")

    # Create state and context
    rule_engine = Mock()
    observation_builder = Mock()
    state = WaitRobKongState(rule_engine, observation_builder)
    context = create_mock_context()
    context.active_responders = [1]

    # Call step with invalid action
    try:
        state.step(context, "invalid_string")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        error_msg = str(e)
        assert "expects MahjongAction or 'auto'" in error_msg, \
            f"Wrong error message: {error_msg}"
        print(f"[PASS] Correctly raised ValueError: {error_msg}")


def test_step_with_invalid_mahjong_action_type():
    """
    Test: step(context, MahjongAction(DISCARD))

    Expected: Should raise ValueError about only WIN/PASS allowed
    """
    print("\n[Test] step with invalid MahjongAction type...")

    # Create state and context
    rule_engine = Mock()
    observation_builder = Mock()
    state = WaitRobKongState(rule_engine, observation_builder)
    context = create_mock_context()
    context.active_responders = [1]

    # Call step with DISCARD action (not allowed in WAIT_ROB_KONG)
    discard_action = MahjongAction(ActionType.DISCARD, 21)
    try:
        state.step(context, discard_action)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        error_msg = str(e)
        assert "Only WIN or PASS actions allowed" in error_msg, \
            f"Wrong error message: {error_msg}"
        print(f"[PASS] Correctly raised ValueError: {error_msg}")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing WaitRobKongState.step() with 'auto' action handling")
    print("=" * 60)

    test_step_with_auto_when_no_responders()
    test_step_with_auto_when_has_responders_raises_error()
    test_step_with_normal_pass_action()
    test_step_with_normal_win_action()
    test_step_with_invalid_action_type()
    test_step_with_invalid_mahjong_action_type()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
