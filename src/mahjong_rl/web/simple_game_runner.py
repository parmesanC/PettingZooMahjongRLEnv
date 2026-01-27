"""
简单的游戏运行器
用于启动FastAPI服务器并运行游戏循环
"""
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.mahjong_rl.web.fastapi_server import MahjongFastAPIServer
from example_mahjong_env import WuhanMahjongEnv


class SimpleGameRunner:
    """简单的游戏运行器"""

    def __init__(self, port=8011):
        self.port = port
        self.env = None
        self.server = None
        self.current_context = None

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

        print(f"✓ 环境初始化完成")
        print(f"  - 当前玩家: {self.current_context.current_player_idx}")
        print(f"  - 赖子: {self.current_context.lazy_tile}")
        print(f"  - 皮子: {self.current_context.skin_tile}")

    def on_action_received(self, action, player_id=None):
        """
        处理接收到的动作

        Args:
            action: (action_type, parameter) 元组
            player_id: 发送动作的玩家ID
        """
        current_player = self.env.agent_selection

        if player_id is not None and player_id != self.env.possible_agents.index(current_player):
            print(f"警告: 玩家{player_id}尝试在玩家{current_player}的回合行动")
            return

        action_type, parameter = action
        print(f"收到动作: type={action_type}, param={parameter}, player={current_player}")

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

        except Exception as e:
            print(f"执行动作失败: {e}")
            import traceback
            traceback.print_exc()

    def get_current_context(self):
        """获取当前游戏上下文"""
        return self.current_context

    def send_state_to_all(self):
        """发送状态给所有连接的客户端"""
        if self.server and self.current_context:
            # 给每个玩家发送对应视角的状态
            for player_idx in range(4):
                self.server.send_json_state(self.current_context, player_idx)

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

        # 启动服务器
        self.server.start()


if __name__ == "__main__":
    runner = SimpleGameRunner(port=8011)
    runner.start()
