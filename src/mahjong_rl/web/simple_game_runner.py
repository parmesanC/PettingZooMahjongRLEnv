"""
简单的游戏运行器
继承 ManualController 基类，复用标准游戏循环逻辑
"""
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.mahjong_rl.manual_control.base import ManualController
from src.mahjong_rl.web.fastapi_server import MahjongFastAPIServer
from src.mahjong_rl.web.utils.action_validator import ActionValidator


class SimpleGameRunner(ManualController):
    """
    简单的游戏运行器，继承 ManualController

    复用基类的标准游戏循环：
    - 自动处理 AI 玩家
    - 自动状态推进
    - 人类玩家通过 WebSocket 交互
    """

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
        timeout = 300  # 5分钟超时
        start_time = time.time()

        while not self.action_received:
            time.sleep(0.1)
            if time.time() - start_time > timeout:
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

    def on_action_received(self, action, player_id=None):
        """
        前端发送动作的回调（由 WebSocket 调用）

        Args:
            action: (action_type, parameter) 元组
            player_id: 发送动作的玩家ID（用于验证）
        """
        # 使用 context 的 current_player_idx 进行验证
        if player_id is not None:
            current_player_idx = self.env.unwrapped.context.current_player_idx
            if player_id != current_player_idx:
                print(f"警告: 玩家{player_id}尝试在玩家{current_player_idx}的回合行动")
                return

        # 设置动作，解除阻塞
        self.pending_action = action
        self.action_received = True

    def _get_action_mask(self, player_idx):
        """获取指定玩家的 action_mask"""
        try:
            current_agent = self.env.possible_agents[player_idx]
            obs, reward, terminated, truncated, info = self.env.last()
            return obs['action_mask'] if not terminated and not truncated else None
        except:
            return None


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
