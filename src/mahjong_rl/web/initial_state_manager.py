"""
初始状态管理器
管理游戏初始化后的初始HTML和action_mask
"""


class InitialStateManager:
    """
    初始状态管理器
    
    职责：
    - 存储初始游戏HTML
    - 存储初始action_mask
    - 提供给新连接的客户端
    """
    
    def __init__(self):
        self.initial_html = None
        self.action_mask = None
        self.is_initialized = False
    
    def set_initial_state(self, html: str, action_mask: dict = None):
        """
        设置初始状态
        
        Args:
            html: 初始游戏HTML
            action_mask: 初始动作掩码（可选）
        """
        self.initial_html = html
        self.action_mask = action_mask
        self.is_initialized = True
        print("✓ 初始状态已设置")
    
    def get_initial_state(self) -> tuple:
        """
        获取初始状态
        
        Returns:
            (html, action_mask) 元组，如果未初始化则返回 (None, None)
        """
        return self.initial_html, self.action_mask
    
    def clear(self):
        """清除初始状态"""
        self.initial_html = None
        self.action_mask = None
        self.is_initialized = False
