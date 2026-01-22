"""
WebSocket连接管理器
"""

from fastapi import WebSocket
from typing import List
import json
import asyncio
from .json_encoder import NumpyJSONEncoder


class WebSocketManager:
    """
    WebSocket连接管理器
    
    职责：
    - 管理所有WebSocket连接
    - 提供广播功能
    - 处理连接/断开
    - 向新连接发送初始状态
    """
    
    def __init__(self, initial_state_manager=None):
        self.active_connections: List[WebSocket] = []
        self.initial_state_manager = initial_state_manager
        
        # 保存对初始状态管理器的引用
        if initial_state_manager is not None:
            initial_state_manager.websocket_manager = self
    
    async def connect(self, websocket: WebSocket):
        """接受新连接"""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✓ 新连接，总连接数: {len(self.active_connections)}")
        
        # 发送初始状态
        if self.initial_state_manager:
            await self._send_initial_state(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """断开连接"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"✓ 连接断开，剩余连接数: {len(self.active_connections)}")
    
    async def _send_initial_state(self, websocket: WebSocket):
        """发送初始状态给新连接的客户端"""
        html, action_mask = self.initial_state_manager.get_initial_state()
        
        if html:
            message = {
                'type': 'initial_state',
                'html': html,
                'action_mask': action_mask
            }
            message_json = json.dumps(message, ensure_ascii=False, cls=NumpyJSONEncoder)
            await websocket.send_text(message_json)
            print("  ✓ 初始状态已发送给新连接")
        else:
            print("  ✗ 警告：初始状态未设置")
    
    async def broadcast(self, message: dict):
        """异步广播消息给所有连接"""
        message_json = json.dumps(message, ensure_ascii=False, cls=NumpyJSONEncoder)
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                print(f"广播失败: {e}")
        
        if len(self.active_connections) > 0:
            print(f"📡 广播消息到 {len(self.active_connections)} 个客户端")
    
    def broadcast_sync(self, message: dict):
        """同步广播（从非异步上下文调用）"""
        import threading
        import asyncio
        
        def run_broadcast():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.broadcast(message))
            loop.close()
        
        thread = threading.Thread(target=run_broadcast, daemon=True)
        thread.start()
