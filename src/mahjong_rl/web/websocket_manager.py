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
        import asyncio

        try:
            # 尝试获取当前运行的事件循环
            loop = asyncio.get_running_loop()
            # 如果已有事件循环在运行，创建任务
            asyncio.create_task(self._broadcast_safe(message))
        except RuntimeError:
            # 没有运行中的事件循环，创建新线程
            import threading

            def run_broadcast():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.broadcast(message))
                finally:
                    loop.close()

            thread = threading.Thread(target=run_broadcast, daemon=True)
            thread.start()

    async def _broadcast_safe(self, message: dict):
        """安全广播，处理连接异常"""
        message_json = json.dumps(message, ensure_ascii=False, cls=NumpyJSONEncoder)

        # 移除已关闭的连接
        to_remove = []
        for connection in self.active_connections:
            try:
                if connection.client_state.value != 1:  # WebSocketState.CONNECTED = 1
                    to_remove.append(connection)
                    continue
                await connection.send_text(message_json)
            except Exception as e:
                print(f"发送消息失败，移除连接: {e}")
                to_remove.append(connection)

        # 清理无效连接
        for conn in to_remove:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
