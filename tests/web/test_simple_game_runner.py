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
    """测试接收有效动作并执行"""
    env = Mock()
    env.unwrapped.context.current_player_idx = 0
    env.possible_agents = ['player_0', 'player_1', 'player_2', 'player_3']

    # Mock observation with valid action
    mock_obs = {'action_mask': np.zeros(145, dtype=np.int8)}
    mock_obs['action_mask'][5] = 1  # 可以打出5号牌
    env.last = Mock(return_value=(mock_obs, 0, False, False, {}))
    env.step = Mock(return_value=(mock_obs, 0, False, False, {}))

    runner = SimpleGameRunner(env, port=8011, strategies=[None, None, None, None])
    runner.server = Mock()
    runner.server.send_json_state = Mock()
    runner._get_action_mask = Mock(return_value=mock_obs['action_mask'])
    runner.render_env = Mock()

    action = (0, 5)  # 打出5号牌
    runner.on_action_received(action, player_id=0)

    # 验证动作被执行
    assert env.step.called
    assert runner.render_env.called


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


@patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer')
def test_on_action_received_rejects_invalid_action(MockServer):
    """测试非法动作被拒绝"""
    env = Mock()
    env.unwrapped.context.current_player_idx = 0
    env.possible_agents = ['player_0', 'player_1', 'player_2', 'player_3']

    # 创建全0的action_mask（没有任何动作可用）
    mock_obs = {'action_mask': np.zeros(145, dtype=np.int8)}
    env.last = Mock(return_value=(mock_obs, 0, False, False, {}))

    runner = SimpleGameRunner(env, port=8011, strategies=[None, None, None, None])
    runner.server = Mock()
    runner.server.websocket_manager = Mock()
    runner.server.websocket_manager.broadcast_sync = Mock()

    action = (0, 5)  # 尝试打出5号牌，但action_mask全0
    runner.on_action_received(action, player_id=0)

    # 验证错误消息被发送
    assert runner.server.websocket_manager.broadcast_sync.called
    call_args = runner.server.websocket_manager.broadcast_sync.call_args[0][0]
    assert call_args['type'] == 'error'
    assert '当前不可用' in call_args['message']

    # 验证动作没有被接受
    assert runner.pending_action is None
    assert runner.action_received is False


@patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer')
def test_process_auto_players_executes_all_ai(MockServer):
    """测试自动处理所有AI玩家"""
    env = Mock()
    env.unwrapped.context.current_player_idx = 1
    env.unwrapped.context.is_win = False
    env.unwrapped.context.is_flush = False
    env.agent_selection = 'player_1'
    env.possible_agents = ['player_0', 'player_1', 'player_2', 'player_3']

    # Mock observations - use numpy array for action_mask
    mock_obs = {'action_mask': np.ones(145, dtype=np.int8)}
    env.last = Mock(return_value=(mock_obs, 0, False, False, {}))
    env.step = Mock(return_value=(mock_obs, 0, False, False, {}))

    # Mock AI strategy
    ai_strategy = Mock()
    ai_strategy.choose_action = Mock(return_value=(10, -1))  # PASS

    runner = SimpleGameRunner(
        env=env,
        port=8011,
        strategies=[None, ai_strategy, ai_strategy, None],
        ai_delay=0
    )
    runner.server = Mock()
    runner.render_env = Mock()

    # 执行自动玩家处理
    runner._process_auto_players(mock_obs)

    # 验证AI动作被执行
    assert ai_strategy.choose_action.called
    assert env.step.called
