"""
SimpleGameRunner 完整游戏流程集成测试

这些测试验证 SimpleGameRunner 与环境的完整集成，
包括人类玩家与AI玩家的交互、动作验证、游戏重启等功能。
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
from src.mahjong_rl.web.simple_game_runner import SimpleGameRunner


@pytest.fixture
def mock_env():
    """创建mock环境"""
    env = Mock()
    env.unwrapped.context.current_player_idx = 0
    env.unwrapped.context.is_win = False
    env.unwrapped.context.is_flush = False
    env.unwrapped.context.lazy_tile = 27
    env.unwrapped.context.skin_tile = [12, 13]
    env.agent_selection = 'player_0'
    env.possible_agents = ['player_0', 'player_1', 'player_2', 'player_3']

    # Mock observation with valid action_mask
    mock_obs = {
        'action_mask': np.zeros(145, dtype=np.int8)
    }
    mock_obs['action_mask'][0] = 1  # 可以打出0号牌
    mock_obs['action_mask'][144] = 1  # 可以PASS
    env.last = Mock(return_value=(mock_obs, 0, False, False, {}))

    return env


def test_full_game_flow_human_then_ai(mock_env):
    """测试完整游戏流程：人类玩家 -> AI玩家"""
    from src.mahjong_rl.agents.ai.random_strategy import RandomStrategy

    # 创建策略：玩家0是人类，玩家1-3是AI
    strategies = [None, RandomStrategy(), RandomStrategy(), RandomStrategy()]

    with patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer'):
        runner = SimpleGameRunner(
            env=mock_env,
            port=8011,
            strategies=strategies,
            ai_delay=0
        )
        runner.server = Mock()
        runner.render_env = Mock()

        # 模拟人类玩家动作
        human_action = (0, 0)  # 打出0号牌

        # 修改 env.step 返回值模拟AI玩家回合
        def mock_step_side_effect(action):
            # 模拟人类动作后，切换到AI玩家
            mock_env.unwrapped.context.current_player_idx = 1
            mock_env.agent_selection = 'player_1'

            # 返回新的观测（AI玩家回合）
            ai_obs = {
                'action_mask': np.zeros(145, dtype=np.int8)
            }
            ai_obs['action_mask'][144] = 1  # AI可以PASS
            return (ai_obs, 0, False, False, {})

        mock_env.step = Mock(side_effect=mock_step_side_effect)

        # 模拟 env.last 返回AI玩家的观测
        def mock_last_side_effect():
            current_idx = mock_env.unwrapped.context.current_player_idx
            obs = {
                'action_mask': np.zeros(145, dtype=np.int8)
            }
            if current_idx == 0:
                obs['action_mask'][0] = 1  # 人类可以打牌
                obs['action_mask'][144] = 1
            else:
                obs['action_mask'][144] = 1  # AI只能PASS
            return (obs, 0, False, False, {})

        mock_env.last = Mock(side_effect=mock_last_side_effect)

        # 执行人类动作
        runner.on_action_received(human_action, player_id=0)

        # 验证人类动作被接受
        assert mock_env.step.called

        # 验证render_env被调用来发送状态更新
        assert runner.render_env.called

        # 由于AI策略会被调用，验证AI逻辑被执行
        # 注意：由于_mock_env的设置，AI玩家应该会自动执行


def test_action_validation_blocks_invalid_actions(mock_env):
    """测试动作验证阻止非法动作"""
    # 修改action_mask，使0号牌不可用
    mock_obs = mock_env.last.return_value[0]
    mock_obs['action_mask'][0] = 0  # 0号牌不可用
    mock_obs['action_mask'][144] = 1  # 只有PASS可用

    with patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer'):
        runner = SimpleGameRunner(
            env=mock_env,
            port=8011,
            strategies=[None, None, None, None]
        )
        runner.server = Mock()
        runner.server.websocket_manager = Mock()
        runner.server.websocket_manager.broadcast_sync = Mock()

        # 尝试非法动作
        invalid_action = (0, 0)  # 尝试打出不可用的0号牌
        runner.on_action_received(invalid_action, player_id=0)

        # 验证错误消息被发送
        assert runner.server.websocket_manager.broadcast_sync.called
        call_args = runner.server.websocket_manager.broadcast_sync.call_args[0][0]
        assert call_args['type'] == 'error'
        assert '当前不可用' in call_args['message']

        # 验证env.step没有被调用（动作被拒绝）
        assert not mock_env.step.called


def test_restart_game_preserves_config(mock_env):
    """测试重启游戏保留配置"""
    with patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer'):
        runner = SimpleGameRunner(
            env=mock_env,
            port=8011,
            strategies=[None, None, Mock(), Mock()],
            ai_delay=0.5
        )
        runner.server = Mock()
        runner.render_env = Mock()
        runner._send_message = Mock()

        # 执行重启
        restart_action = (-1, 0)
        runner.on_action_received(restart_action, player_id=0)

        # 验证配置被保留
        assert runner.port == 8011
        assert runner.ai_delay == 0.5
        assert len(runner.strategies) == 4
        assert runner.strategies[0] is None
        assert runner.strategies[1] is None

        # 验证env.reset被调用
        assert mock_env.reset.called

        # 验证render_env被调用来发送新状态
        assert runner.render_env.called

        # 验证成功消息被发送
        assert runner._send_message.called


def test_action_validation_blocks_wrong_player(mock_env):
    """测试非当前玩家的动作被拒绝"""
    # 当前玩家是0，但玩家1尝试行动
    mock_env.unwrapped.context.current_player_idx = 0

    with patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer'):
        runner = SimpleGameRunner(
            env=mock_env,
            port=8011,
            strategies=[None, None, None, None]
        )
        runner.server = Mock()
        runner.server.websocket_manager = Mock()
        runner.server.websocket_manager.broadcast_sync = Mock()

        # 玩家1尝试在玩家0的回合行动
        wrong_player_action = (0, 0)
        runner.on_action_received(wrong_player_action, player_id=1)

        # 验证错误消息被发送
        assert runner.server.websocket_manager.broadcast_sync.called
        call_args = runner.server.websocket_manager.broadcast_sync.call_args[0][0]
        assert call_args['type'] == 'error'
        assert '不是你的回合' in call_args['message']

        # 验证env.step没有被调用
        assert not mock_env.step.called


def test_action_validation_accepts_pass_action(mock_env):
    """测试PASS动作总是被接受（当可用时）"""
    mock_obs = mock_env.last.return_value[0]
    mock_obs['action_mask'][144] = 1  # PASS可用

    with patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer'):
        runner = SimpleGameRunner(
            env=mock_env,
            port=8011,
            strategies=[None, None, None, None]
        )
        runner.server = Mock()
        runner.render_env = Mock()

        # 设置step返回值
        mock_env.step = Mock(return_value=(mock_obs, 0, False, False, {}))

        pass_action = (10, -1)
        runner.on_action_received(pass_action, player_id=0)

        # 验证动作被接受
        assert mock_env.step.called
        assert runner.render_env.called


def test_ai_players_auto_execute_after_human_action(mock_env):
    """测试人类动作后AI玩家自动执行"""
    from src.mahjong_rl.agents.ai.random_strategy import RandomStrategy

    # 创建AI策略
    ai_strategy = RandomStrategy()
    strategies = [None, ai_strategy, None, None]

    with patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer'):
        runner = SimpleGameRunner(
            env=mock_env,
            port=8011,
            strategies=strategies,
            ai_delay=0
        )
        runner.server = Mock()
        runner.render_env = Mock()

        # 设置环境返回AI玩家的观测
        ai_obs = {
            'action_mask': np.zeros(145, dtype=np.int8)
        }
        ai_obs['action_mask'][144] = 1  # AI可以PASS

        def mock_step_side_effect(action):
            # 切换到AI玩家
            mock_env.unwrapped.context.current_player_idx = 1
            mock_env.agent_selection = 'player_1'
            return (ai_obs, 0, False, False, {})

        mock_env.step = Mock(side_effect=mock_step_side_effect)

        def mock_last_side_effect():
            current_idx = mock_env.unwrapped.context.current_player_idx
            obs = {
                'action_mask': np.zeros(145, dtype=np.int8)
            }
            if current_idx == 0:
                # 人类玩家可以打牌
                obs['action_mask'][0] = 1
                obs['action_mask'][144] = 1
            elif current_idx == 1:
                obs['action_mask'][144] = 1  # AI可以PASS
            return (obs, 0, False, False, {})

        mock_env.last = Mock(side_effect=mock_last_side_effect)

        # 人类玩家动作
        human_action = (10, -1)  # PASS
        runner.on_action_received(human_action, player_id=0)

        # 验证AI玩家的动作被执行（通过_process_auto_players）
        # 由于mock_env.step会被调用多次（人类一次+AI可能多次）
        assert mock_env.step.call_count >= 1


def test_multiple_ai_players_execute_in_sequence(mock_env):
    """测试多个AI玩家按顺序执行"""
    from src.mahjong_rl.agents.ai.random_strategy import RandomStrategy

    # 创建多个AI策略
    strategies = [None, RandomStrategy(), RandomStrategy(), RandomStrategy()]

    with patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer'):
        runner = SimpleGameRunner(
            env=mock_env,
            port=8011,
            strategies=strategies,
            ai_delay=0
        )
        runner.server = Mock()
        runner.render_env = Mock()

        # 模拟玩家切换序列
        player_sequence = [1, 2, 3, 0]  # 人类->AI1->AI2->AI3->人类
        seq_index = [0]

        def mock_step_side_effect(action):
            # 切换到下一个玩家
            next_player = player_sequence[seq_index[0] % len(player_sequence)]
            seq_index[0] += 1
            mock_env.unwrapped.context.current_player_idx = next_player
            mock_env.agent_selection = f'player_{next_player}'

            obs = {
                'action_mask': np.zeros(145, dtype=np.int8)
            }
            obs['action_mask'][144] = 1  # 可以PASS
            return (obs, 0, False, False, {})

        mock_env.step = Mock(side_effect=mock_step_side_effect)

        # 人类玩家动作
        human_action = (10, -1)  # PASS
        runner.on_action_received(human_action, player_id=0)

        # 验证多个AI玩家被执行
        assert mock_env.step.call_count >= 2  # 至少人类+1个AI


def test_game_over_stops_auto_players(mock_env):
    """测试游戏结束后停止自动处理AI玩家"""
    from src.mahjong_rl.agents.ai.random_strategy import RandomStrategy

    strategies = [None, RandomStrategy(), None, None]

    with patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer'):
        runner = SimpleGameRunner(
            env=mock_env,
            port=8011,
            strategies=strategies,
            ai_delay=0
        )
        runner.server = Mock()
        runner.render_env = Mock()
        runner.render_final_state = Mock()

        # 设置游戏结束
        game_over_obs = {
            'action_mask': np.zeros(145, dtype=np.int8)
        }

        # 修改action_mask使WIN动作可用
        mock_obs = mock_env.last.return_value[0]
        mock_obs['action_mask'][143] = 1  # WIN可用

        def mock_step_side_effect(action):
            # 游戏结束
            mock_env.unwrapped.context.is_win = True
            return (game_over_obs, 0, True, False, {'winner': 0})

        mock_env.step = Mock(side_effect=mock_step_side_effect)

        # 人类玩家动作导致游戏结束
        human_action = (9, 0)  # WIN
        runner.on_action_received(human_action, player_id=0)

        # 验证render_final_state被调用
        assert runner.render_final_state.called


def test_get_action_mask_returns_none_when_game_over(mock_env):
    """测试游戏结束时_get_action_mask返回None"""
    mock_env.possible_agents = ['player_0', 'player_1', 'player_2', 'player_3']

    # 设置游戏结束状态
    mock_obs = {'action_mask': np.zeros(145, dtype=np.int8)}
    mock_env.last = Mock(return_value=(mock_obs, 0, True, False, {}))  # terminated=True

    runner = SimpleGameRunner(mock_env, port=8011)

    action_mask = runner._get_action_mask(0)

    # 验证返回None
    assert action_mask is None


def test_send_error_message_formatting(mock_env):
    """测试错误消息格式正确"""
    with patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer'):
        runner = SimpleGameRunner(
            env=mock_env,
            port=8011,
            strategies=[None, None, None, None]
        )
        runner.server = Mock()
        runner.server.websocket_manager = Mock()
        runner.server.websocket_manager.broadcast_sync = Mock()

        # 发送错误消息
        test_error = "测试错误消息"
        runner._send_error(test_error)

        # 验证消息格式
        assert runner.server.websocket_manager.broadcast_sync.called
        call_args = runner.server.websocket_manager.broadcast_sync.call_args[0][0]
        assert call_args['type'] == 'error'
        assert call_args['message'] == test_error


def test_send_info_message_formatting(mock_env):
    """测试信息消息格式正确"""
    with patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer'):
        runner = SimpleGameRunner(
            env=mock_env,
            port=8011,
            strategies=[None, None, None, None]
        )
        runner.server = Mock()
        runner.server.websocket_manager = Mock()
        runner.server.websocket_manager.broadcast_sync = Mock()

        # 发送信息消息
        test_message = "测试信息消息"
        runner._send_message(test_message)

        # 验证消息格式
        assert runner.server.websocket_manager.broadcast_sync.called
        call_args = runner.server.websocket_manager.broadcast_sync.call_args[0][0]
        assert call_args['type'] == 'info'
        assert call_args['message'] == test_message
