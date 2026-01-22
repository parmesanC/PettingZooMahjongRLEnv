"""
网页手动控制器
使用WebSocket进行实时交互
"""

import time
import json
from typing import Tuple
from .base import ManualController
from ..visualization.web_renderer import WebRenderer


class WebManualController(ManualController):
    """
    网页手动控制器
    
    特性：
    - WebSocket实时通信
    - 网页渲染游戏状态
    - 接收前端动作
    """
    
    def __init__(self, env, max_episodes=1, port=8000, strategies=None):
        super().__init__(env, max_episodes, strategies)
        self.renderer = WebRenderer()
        self.port = port
        self.pending_action = None
        self.action_received = False
        self.server = None
    
    def run(self):
        """启动FastAPI服务器并运行游戏"""
        try:
            from ..web.fastapi_server import MahjongFastAPIServer
            self.server = MahjongFastAPIServer(
                env=self.env,
                controller=self,
                port=self.port
            )
            
            # 在服务器启动前初始化游戏状态
            self._initialize_game_state()
            
            # 启动服务器（阻塞）
            self.server.start()
            
            # 注意：不要调用super().run()，避免重复初始化
            # 游戏循环在WebSocket消息中驱动
        except ImportError as e:
            print("错误: 需要安装fastapi和uvicorn")
            print("请运行: pip install fastapi uvicorn")
            print(f"详细错误: {e}")
            raise
        except Exception as e:
            print(f"启动服务器失败: {e}")
            raise
    
    def _initialize_game_state(self):
        """初始化游戏状态（在服务器启动前，只执行一次）"""
        print("\n" + "=" * 60)
        print("初始化游戏状态（Web控制器层面）...")
        print("=" * 60)
        
        # 重置环境（只执行一次）
        print("  - 重置环境...")
        self.env.reset()
        print("  ✓ 环境重置成功")

        # 获取action_mask
        print("  - 获取action_mask...")
        # action_mask现在包含在observation中
        action_mask = self.env.context.observation['action_mask']
        print("  ✓ action_mask获取成功")

        # 生成初始HTML
        print("  - 生成初始HTML...")
        initial_html = self.renderer.render(
            self.env.context,
            self.env.agent_selection,
            action_mask
        )
        print("  ✓ 初始HTML生成成功")
        
        # 保存到初始状态管理器
        print("  - 保存初始状态...")
        self.server.set_initial_state(initial_html, action_mask)
        print("  ✓ 初始状态已保存")
        
        print("=" * 60)
        print("✓ 游戏状态初始化完成\n")
    
    def render_env(self):
        """渲染环境到网页"""
        html = self.renderer.render(self.env.context, self.env.agent_selection)
        if self.server:
            self.server.send_state(html)
    
    def get_human_action(self, observation, info) -> Tuple[int, int]:
        """
        获取人类动作（通过WebSocket）
        
        阻塞等待前端发送动作
        """
        self.action_received = False
        
        if self.server:
            self.server.send_action_prompt(observation['action_mask'])
        
        while not self.action_received:
            time.sleep(0.1)
        
        action = self.pending_action
        self.action_received = False
        self.pending_action = None
        
        return action
    
    def render_final_state(self, info):
        """渲染最终状态到网页"""
        html = self.renderer.render_game_over(info)
        if self.server:
            self.server.send_final_state(html)
    
    def on_action_received(self, action: Tuple[int, int]):
        """
        前端发送动作的回调
        
        Args:
            action: (action_type, parameter) 元组
        """
        self.pending_action = action
        self.action_received = True
