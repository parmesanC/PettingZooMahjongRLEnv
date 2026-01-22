"""
FastAPI麻将游戏服务器
提供HTTP服务、WebSocket、静态文件服务
"""
import os
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware


class MahjongFastAPIServer:
    """
    FastAPI麻将游戏服务器
    
    优势：
    - WebSocket原生支持，真正实时通信
    - 自动静态文件服务
    - 异步高性能
    - 完整类型提示
    - Swagger自动文档 (/docs)
    - CORS支持
    """
    
    def __init__(self, env, controller, port=8011):
        self.app = FastAPI(
            title="武汉麻将API",
            description="PettingZoo麻将强化学习环境",
            version="1.0.0"
        )
        self.env = env
        self.controller = controller

        self.port = port
        
        # 导入WebSocket管理器
        from .websocket_manager import WebSocketManager
        from .initial_state_manager import InitialStateManager
        
        # 创建初始状态管理器
        self.initial_state_manager = InitialStateManager()

        # 创建WebSocket管理器（传入初始状态管理器）
        self.websocket_manager = WebSocketManager(self.initial_state_manager)
        
        # 挂载路由
        self._setup_routes()
        
        # 挂载CORS
        self._setup_cors()
        
        # 挂载静态文件
        self._mount_static_files()
    
    def _setup_cors(self):
        """设置CORS"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """设置路由"""
        
        # 主页路由 - 返回游戏HTML
        @self.app.get("/")
        async def read_root():
            html_path = os.path.join(
                os.path.dirname(__file__), 
                "static", 
                "game.html"
            )
            try:
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                return HTMLResponse(content=html_content)
            except FileNotFoundError:
                return HTMLResponse(
                    content="<h1>游戏页面未找到</h1><p>请确保 static/game.html 存在</p>",
                    status_code=404
                )
        
        # WebSocket路由
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket端点"""
            # 使用WebSocket管理器
            await self.websocket_manager.connect(websocket)
            
            try:
                while True:
                    # 接收消息
                    data = await websocket.receive_text()
                    
                    # 解析并处理
                    import json
                    message = json.loads(data)
                    
                    if message['type'] == 'action':
                        self.controller.on_action_received(
                            (message['actionType'], message['parameter'])
                        )
                    
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
                print("WebSocket客户端断开连接")
            
            except Exception as e:
                print(f"WebSocket错误: {e}")
                self.websocket_manager.disconnect(websocket)
    
    def _mount_static_files(self):
        """挂载静态文件目录"""
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        
        # 确保静态文件目录存在
        os.makedirs(static_dir, exist_ok=True)
        
        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    def set_initial_state(self, html: str, action_mask: dict = None):
        """
        设置初始状态（由外部控制器调用）
        
        Args:
            html: 初始游戏HTML
            action_mask: 初始动作掩码（可选）
        """
        self.initial_state_manager.set_initial_state(html, action_mask)
        print("✓ 初始状态已设置（通过控制器）")
    
    def send_state(self, html: str):
        """广播游戏状态"""
        message = {
            'type': 'state',
            'html': html
        }
        self.websocket_manager.broadcast_sync(message)
    
    def send_action_prompt(self, action_mask: Dict):
        """发送动作提示"""
        from ..visualization.web_renderer import WebRenderer
        renderer = WebRenderer()
        html = renderer.render_action_prompt(action_mask)
        
        message = {
            'type': 'action_prompt',
            'html': html
        }
        self.websocket_manager.broadcast_sync(message)
    
    def send_final_state(self, html: str):
        """发送最终状态"""
        message = {
            'type': 'game_over',
            'html': html
        }
        self.websocket_manager.broadcast_sync(message)
    
    def start(self):
        """启动FastAPI服务器"""
        import uvicorn
        
        print("\n" + "=" * 60)
        print("🌐 FastAPI麻将游戏服务器")
        print("=" * 60)
        print(f"📌 游戏地址: http://localhost:{self.port}")
        print(f"📚 API文档: http://localhost:{self.port}/docs")
        print(f"🔌 端点: /ws (WebSocket)")
        print("=" * 60)
        print("请在浏览器中打开游戏地址\n")
        
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )
