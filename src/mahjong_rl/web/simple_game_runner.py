"""
简单的游戏运行器
用于启动FastAPI服务器并运行游戏循环
支持人类玩家和AI玩家混合
"""
import sys
import time
import asyncio
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.mahjong_rl.web.fastapi_server import MahjongFastAPIServer
from example_mahjong_env import WuhanMahjongEnv
from src.mahjong_rl.agents.ai.random_strategy import RandomStrategy


class SimpleGameRunner:
    """简单的游戏运行器，支持AI玩家"""

    def __init__(self, port=8011, human_players=1, ai_delay=0.5):
        """
        初始化游戏运行器

        Args:
            port: WebSocket服务器端口
            human_players: 人类玩家数量（0-4），其余为AI
            ai_delay: AI思考延迟时间（秒），用于观察游戏过程
        """
        self.port = port
        self.human_players = human_players
        self.ai_delay = ai_delay
        self.env = None
        self.server = None
        self.current_context = None
        self.strategies = []
        self.ai_enabled = human_players < 4
        self.current_action_masks = [None] * 4  # 存储每个玩家的动作掩码

    def setup(self):
        """设置环境和服务器"""
        print("初始化武汉麻将环境...")

        # 创建环境
        self.env = WuhanMahjongEnv(
            render_mode=None,
            training_phase=3,  # 完全信息
            enable_logging=False
        )

        # 重置环境获取初始状态
        obs, info = self.env.reset()
        self.current_context = self.env.unwrapped.context

        # 创建AI策略
        if self.ai_enabled:
            for i in range(4):
                if i >= self.human_players:
                    self.strategies.append(RandomStrategy())
                    print(f"  - 玩家{i}: AI (随机策略)")
                else:
                    self.strategies.append(None)
                    print(f"  - 玩家{i}: 人类")
        else:
            self.strategies = [None] * 4

        print(f"✓ 环境初始化完成")
        print(f"  - 当前玩家: {self.current_context.current_player_idx}")
        print(f"  - agent_selection: {self.env.agent_selection}")
        print(f"  - 赖子: {self.current_context.lazy_tile}")
        print(f"  - 皮子: {self.current_context.skin_tile}")

    def on_action_received(self, action, player_id=None):
        """
        处理接收到的动作

        Args:
            action: (action_type, parameter) 元组
            player_id: 发送动作的玩家ID
        """
        action_type, parameter = action

        # 处理重启请求 (action_type = -1)
        if action_type == -1:
            print(f"\n{'='*60}")
            print("收到重启请求，开始新的一局...")
            print(f"{'='*60}")
            self.restart_game()
            return

        # 使用 context 的 current_player_idx 进行验证
        current_player_idx = self.current_context.current_player_idx

        if player_id is not None and player_id != current_player_idx:
            print(f"警告: 玩家{player_id}尝试在玩家{current_player_idx}的回合行动")
            return

        player_source = f"玩家{current_player_idx} ({'人类' if self.strategies[current_player_idx] is None else 'AI'})"
        print(f"收到动作: type={action_type}, param={parameter}, source={player_source}")

        # 执行动作
        try:
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.current_context = self.env.unwrapped.context

            # 发送新状态到前端
            self.send_state_to_all()

            if terminated or truncated:
                print(f"\n游戏结束! 终止={terminated}, 截断={truncated}")
                if self.current_context.winner_ids:
                    print(f"获胜者: {self.current_context.winner_ids}")
                return

            # 检查是否需要执行AI动作
            self._check_and_execute_ai()

        except Exception as e:
            print(f"执行动作失败: {e}")
            import traceback
            traceback.print_exc()

    def restart_game(self):
        """重启游戏，开始新的一局"""
        try:
            # 重置环境
            print("正在重置环境...")
            obs, info = self.env.reset()
            self.current_context = self.env.unwrapped.context

            print(f"✓ 新的一局开始！")
            print(f"  - 当前玩家: {self.current_context.current_player_idx}")
            print(f"  - 赖子: {self.current_context.lazy_tile}")
            print(f"  - 皮子: {self.current_context.skin_tile}")

            # 发送新状态到前端
            self.send_state_to_all()

            # 如果第一个玩家是AI，立即执行AI动作
            if self.ai_enabled:
                self._check_and_execute_ai()

        except Exception as e:
            print(f"重启游戏失败: {e}")
            import traceback
            traceback.print_exc()

    def _check_and_execute_ai(self):
        """检查当前玩家是否是AI，如果是则执行AI动作"""
        max_ai_steps = 10  # 防止无限循环
        steps = 0

        while steps < max_ai_steps:
            current_player_idx = self.current_context.current_player_idx
            strategy = self.strategies[current_player_idx]

            # 如果当前是人类玩家或游戏已结束，停止
            if strategy is None or self.current_context.is_win or self.current_context.is_flush:
                break

            # 执行AI动作
            print(f"AI玩家{current_player_idx}思考中...")
            ai_action = self._execute_ai_action(current_player_idx, strategy)

            if ai_action is None:
                print(f"AI玩家{current_player_idx}无可用动作")
                break

            steps += 1

        if steps >= max_ai_steps:
            print("警告：达到最大AI步数")

    def _execute_ai_action(self, player_idx, strategy):
        """
        执行AI动作

        Args:
            player_idx: 玩家索引
            strategy: AI策略

        Returns:
            执行的动作，如果无法执行则返回None
        """
        try:
            # AI思考延迟
            if self.ai_delay > 0:
                time.sleep(self.ai_delay)

            # 获取当前玩家的观测
            current_agent = self.env.possible_agents[player_idx]
            obs, reward, terminated, truncated, info = self.env.last()

            if terminated or truncated:
                return None

            # AI选择动作
            action_mask = obs['action_mask']
            action = strategy.choose_action(obs, action_mask)

            if action is None:
                return None

            action_type, parameter = action
            print(f"  AI动作: type={action_type}, param={parameter}")

            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.current_context = self.env.unwrapped.context

            # 发送新状态到前端
            self.send_state_to_all()

            if terminated or truncated:
                print(f"\n游戏结束! 终止={terminated}, 截断={truncated}")
                if self.current_context.winner_ids:
                    print(f"获胜者: {self.current_context.winner_ids}")
                return None

            return action

        except Exception as e:
            print(f"AI动作执行失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_current_context(self):
        """获取当前游戏上下文"""
        return self.current_context

    def send_state_to_all(self):
        """发送状态给所有连接的客户端"""
        if self.server and self.current_context:
            # 给每个玩家发送对应视角的状态
            for player_idx in range(4):
                # 获取当前玩家的 action_mask
                action_mask = self._get_action_mask(player_idx)
                self.server.send_json_state(
                    self.current_context,
                    player_idx,
                    action_mask
                )

    def _get_action_mask(self, player_idx):
        """获取指定玩家的 action_mask"""
        try:
            current_agent = self.env.possible_agents[player_idx]
            obs, reward, terminated, truncated, info = self.env.last()
            return obs['action_mask'] if not terminated and not truncated else None
        except:
            return None

    def start(self):
        """启动服务器"""
        if not self.env:
            self.setup()

        # 创建控制器（将自身作为控制器传入）
        controller = self
        self.server = MahjongFastAPIServer(
            env=self.env,
            controller=controller,
            port=self.port
        )

        # 发送初始状态
        self.send_state_to_all()

        # 如果第一个玩家是AI，立即执行AI动作
        if self.ai_enabled:
            self._check_and_execute_ai()

        # 启动服务器
        self.server.start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='武汉麻将WebSocket服务器')
    parser.add_argument('--port', type=int, default=8011, help='服务器端口')
    parser.add_argument('--human', type=int, default=1, help='人类玩家数量（0-4）')
    parser.add_argument('--ai-delay', type=float, default=0.5, help='AI思考延迟时间（秒）')

    args = parser.parse_args()

    runner = SimpleGameRunner(port=args.port, human_players=args.human, ai_delay=args.ai_delay)
    runner.start()
