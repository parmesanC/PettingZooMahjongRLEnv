"""
SimpleGameRunner 单元测试
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.mahjong_rl.web.simple_game_runner import SimpleGameRunner


def test_simple_game_runner_inherits_manual_controller():
    """测试 SimpleGameRunner 继承 ManualController"""
    from src.mahjong_rl.manual_control.base import ManualController

    env = Mock()
    runner = SimpleGameRunner(env, port=8011, strategies=[None, None, None, None])

    assert isinstance(runner, ManualController)


def test_simple_game_runner_initialization():
    """测试 SimpleGameRunner 初始化"""
    env = Mock()
    strategies = [None, None, Mock(), Mock()]

    runner = SimpleGameRunner(
        env=env,
        port=8011,
        max_episodes=1000,
        strategies=strategies,
        ai_delay=0.5
    )

    assert runner.env == env
    assert runner.port == 8011
    assert runner.max_episodes == 1000
    assert runner.strategies == strategies
    assert runner.ai_delay == 0.5
    assert runner.server is None
    assert runner.pending_action is None
    assert runner.action_received is False
    assert runner.action_validator is not None


@patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer')
def test_on_action_received_valid_action(MockServer):
    """测试接收有效动作"""
    env = Mock()
    env.unwrapped.context.current_player_idx = 0
    env.possible_agents = ['player_0', 'player_1', 'player_2', 'player_3']

    # Mock observation with valid action
    mock_obs = {'action_mask': np.zeros(145, dtype=np.int8)}
    mock_obs['action_mask'][5] = 1  # 可以打出5号牌
    env.last = Mock(return_value=(mock_obs, 0, False, False, {}))

    runner = SimpleGameRunner(env, port=8011, strategies=[None, None, None, None])
    runner.server = Mock()
    runner.server.send_json_state = Mock()
    runner._get_action_mask = Mock(return_value=mock_obs['action_mask'])

    action = (0, 5)  # 打出5号牌
    runner.on_action_received(action, player_id=0)

    assert runner.pending_action == action
    assert runner.action_received is True


@patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer')
def test_on_action_received_invalid_player(MockServer):
    """测试非当前玩家发送动作被拒绝"""
    env = Mock()
    env.unwrapped.context.current_player_idx = 0

    runner = SimpleGameRunner(env, port=8011, strategies=[None, None, None, None])
    runner.server = Mock()
    runner.server.websocket_manager = Mock()

    action = (0, 5)
    runner.on_action_received(action, player_id=1)  # 玩家1尝试在玩家0的回合行动

    assert runner.pending_action is None
    assert runner.action_received is False


def test_get_action_mask():
    """测试获取动作掩码"""
    env = Mock()
    env.possible_agents = ['player_0', 'player_1', 'player_2', 'player_3']

    import numpy as np
    mock_obs = {'action_mask': np.zeros(145, dtype=np.int8)}
    mock_obs['action_mask'][0] = 1
    env.last = Mock(return_value=(mock_obs, 0, False, False, {}))

    runner = SimpleGameRunner(env, port=8011, strategies=[None, None, None, None])

    action_mask = runner._get_action_mask(0)

    assert action_mask is not None
    assert action_mask[0] == 1


def test_get_action_mask_when_terminated():
    """测试游戏结束时返回None"""
    env = Mock()
    env.possible_agents = ['player_0', 'player_1', 'player_2', 'player_3']

    import numpy as np
    mock_obs = {'action_mask': np.zeros(145, dtype=np.int8)}
    env.last = Mock(return_value=(mock_obs, 0, True, False, {}))  # terminated=True

    runner = SimpleGameRunner(env, port=8011, strategies=[None, None, None, None])

    action_mask = runner._get_action_mask(0)

    assert action_mask is None
