"""
简单的游戏运行器
继承 ManualController 基类，复用标准游戏循环逻辑
"""
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.mahjong_rl.manual_control.base import ManualController
from src.mahjong_rl.web.fastapi_server import MahjongFastAPIServer
from src.mahjong_rl.web.utils.action_validator import ActionValidator

logger = logging.getLogger(__name__)


class SimpleGameRunner(ManualController):
    """
    简单的游戏运行器，继承 ManualController

    复用基类的标准游戏循环：
    - 自动处理 AI 玩家
    - 自动状态推进
    - 人类玩家通过 WebSocket 交互
    """

    ACTION_TIMEOUT_SECONDS = 300

    def __init__(self, env, port=8011, max_episodes=1000, strategies=None, ai_delay=0.5):
        """
        初始化游戏运行器

        Args:
            env: PettingZoo 环境实例
            port: WebSocket 服务器端口
            max_episodes: 最大回合数
            strategies: 玩家策略列表
            ai_delay: AI 思考延迟时间（秒）
        """
        super().__init__(env, max_episodes, strategies)
        self.port = port
        self.ai_delay = ai_delay
        self.server = None
        self.pending_action = None
        self.action_received = False
        self.action_validator = ActionValidator()

    def run(self):
        """
        启动服务器并运行游戏循环
        """
        # 创建 FastAPI 服务器
        self.server = MahjongFastAPIServer(
            env=self.env,
            controller=self,
            port=self.port
        )

        # 发送初始状态
        self.render_env()

        # 启动服务器（阻塞）
        self.server.start()

    def render_env(self):
        """
        渲染环境状态到前端
        """
        if self.server and self.env.unwrapped.context:
            context = self.env.unwrapped.context

            # 给每个玩家发送对应视角的状态
            for player_idx in range(4):
                # 获取当前玩家的 action_mask
                action_mask = self._get_action_mask(player_idx)
                self.server.send_json_state(
                    context,
                    player_idx,
                    action_mask
                )

    def get_human_action(self, observation, info):
        """
        获取人类玩家动作（通过 WebSocket）

        阻塞等待前端发送动作。

        Returns:
            (action_type, parameter) 元组
        """
        self.action_received = False
        self.pending_action = None

        # 阻塞等待动作
        start_time = time.time()

        while not self.action_received:
            time.sleep(0.1)
            if time.time() - start_time > self.ACTION_TIMEOUT_SECONDS:
                # 超时，返回 PASS
                return (10, -1)

        action = self.pending_action
        self.action_received = False
        self.pending_action = None

        return action

    def render_final_state(self, info):
        """
        渲染最终状态
        """
        if self.server:
            # 发送游戏结束消息到前端
            message = {
                'type': 'game_over',
                'info': info
            }
            self.server.websocket_manager.broadcast_sync(message)

    def on_action_received(self, action: tuple, player_id: int = None) -> None:
        """
        前端发送动作的回调（由 WebSocket 调用）

        这是主要的游戏驱动方法：
        1. 验证动作
        2. 执行当前玩家动作
        3. 自动推进AI玩家
        4. 渲染新状态

        Args:
            action: (action_type, parameter) 元组
            player_id: 发送动作的玩家ID（用于验证）
        """
        action_type, parameter = action

        # 处理重启请求 (action_type = -1)
        if action_type == -1:
            self._restart_game()
            return

        # 使用 context 的 current_player_idx 进行验证
        current_player_idx = self.env.unwrapped.context.current_player_idx

        if player_id is not None and player_id != current_player_idx:
            self._send_error(f"不是你的回合，当前是玩家{current_player_idx}")
            return

        # 获取当前玩家的 action_mask
        action_mask = self._get_action_mask(current_player_idx)
        if action_mask is None:
            self._send_error("游戏已结束")
            return

        # 验证动作合法性
        is_valid, error_msg = self.action_validator.validate_action_with_error_message(
            action_type, parameter, action_mask
        )

        if not is_valid:
            self._send_error(error_msg)
            return

        # 执行人类玩家的动作
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)

            # 发送新状态到前端
            self.render_env()

            # 检查游戏是否结束
            if terminated or truncated:
                self.render_final_state(info)
                return

            # 自动处理AI玩家
            self._process_auto_players(obs)

        except Exception as e:
            logger.error(f"执行动作失败: {e}")
            import traceback
            traceback.print_exc()

    def _send_error(self, message: str) -> None:
        """
        发送错误消息到前端

        Args:
            message: 错误消息
        """
        if self.server and self.server.websocket_manager:
            error_message = {
                'type': 'error',
                'message': message
            }
            self.server.websocket_manager.broadcast_sync(error_message)

    def _restart_game(self) -> None:
        """
        重启游戏，开始新的一局（保留配置）
        """
        try:
            print("\n" + "=" * 60)
            print("收到重启请求，开始新的一局...")
            print("=" * 60)

            # 重置环境
            self.env.reset()

            print(f"✓ 新的一局开始！")
            print(f"  - 当前玩家: {self.env.unwrapped.context.current_player_idx}")
            print(f"  - 赖子: {self.env.unwrapped.context.lazy_tile}")
            print(f"  - 皮子: {self.env.unwrapped.context.skin_tile}")

            # 发送新状态到前端
            self.render_env()

            # 发送重启成功消息
            self._send_message("游戏已重启，配置已保留")

        except Exception as e:
            print(f"重启游戏失败: {e}")
            import traceback
            traceback.print_exc()

    def _send_message(self, message: str) -> None:
        """
        发送普通消息到前端

        Args:
            message: 消息内容
        """
        if self.server and self.server.websocket_manager:
            msg = {
                'type': 'info',
                'message': message
            }
            self.server.websocket_manager.broadcast_sync(msg)

    def _get_action_mask(self, player_idx: int) -> Optional[np.ndarray]:
        """
        获取指定玩家的 action_mask

        Args:
            player_idx: 玩家索引

        Returns:
            action_mask 数组，如果游戏已结束则返回 None
        """
        try:
            current_agent = self.env.possible_agents[player_idx]
            obs, reward, terminated, truncated, info = self.env.last()
            return obs['action_mask'] if not terminated and not truncated else None
        except (KeyError, IndexError, AttributeError):
            return None

    def _process_auto_players(self, initial_obs) -> None:
        """
        处理 AI 玩家和自动状态推进

        在人类玩家动作后，自动执行所有AI玩家的动作。

        Args:
            initial_obs: 初始观测（人类玩家动作后的状态）
        """
        max_ai_steps = 20  # 防止无限循环
        steps = 0

        while steps < max_ai_steps:
            current_agent = self.env.agent_selection
            current_player_idx = self.env.unwrapped.context.current_player_idx

            # 检查游戏是否结束
            if self.env.unwrapped.context.is_win or self.env.unwrapped.context.is_flush:
                break

            # 如果是人类玩家，停止自动推进
            if self._is_human_player(current_agent):
                break

            # 获取AI策略
            strategy = self.strategies[current_player_idx]
            if strategy is None:
                break

            # 执行AI动作
            logger.info(f"AI玩家{current_player_idx}思考中...")

            # 获取当前观测
            obs, reward, terminated, truncated, info = self.env.last()

            # AI选择动作
            action_mask = obs['action_mask']
            action = strategy.choose_action(obs, action_mask)

            if action is None:
                logger.warning(f"AI玩家{current_player_idx}无可用动作")
                break

            # 验证AI动作
            is_valid, error_msg = self.action_validator.validate_action_with_error_message(
                action[0], action[1], action_mask
            )

            if not is_valid:
                logger.warning(f"AI玩家{current_player_idx}选择了非法动作: {error_msg}")
                # AI使用PASS代替
                action = (10, -1)

            action_type, parameter = action
            logger.info(f"  AI动作: type={action_type}, param={parameter}")

            # 执行AI动作
            try:
                obs, reward, terminated, truncated, info = self.env.step(action)

                # AI延迟（观察用）
                if self.ai_delay > 0:
                    time.sleep(self.ai_delay)

                # 发送状态更新
                self.render_env()

                # 检查游戏是否结束
                if terminated or truncated:
                    self.render_final_state(info)
                    return

            except Exception as e:
                logger.error(f"AI动作执行失败: {e}")
                import traceback
                traceback.print_exc()
                break

            steps += 1

        if steps >= max_ai_steps:
            logger.warning("达到最大AI步数")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='武汉麻将WebSocket服务器')
    parser.add_argument('--port', type=int, default=8011, help='服务器端口')
    parser.add_argument('--human', type=int, default=1, help='人类玩家数量（0-4）')
    parser.add_argument('--ai-delay', type=float, default=0.5, help='AI思考延迟时间（秒）')

    args = parser.parse_args()

    from example_mahjong_env import WuhanMahjongEnv
    from src.mahjong_rl.agents.ai.random_strategy import RandomStrategy

    # 创建环境
    env = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,  # 完全信息
        enable_logging=False
    )

    # 创建策略
    strategies = []
    for i in range(4):
        if i < args.human:
            strategies.append(None)  # 人类玩家
        else:
            strategies.append(RandomStrategy())

    # 创建并启动游戏运行器
    runner = SimpleGameRunner(
        env=env,
        port=args.port,
        strategies=strategies,
        ai_delay=args.ai_delay
    )
    runner.run()
