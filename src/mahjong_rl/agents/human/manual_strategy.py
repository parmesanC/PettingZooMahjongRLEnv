"""
人类玩家策略
连接Manual Control控制器
"""

from typing import Tuple, Dict
from ..base import PlayerStrategy


class ManualPlayerStrategy(PlayerStrategy):
    """
    人类玩家策略
    
    通过ManualController获取用户输入。
    
    设计原则：
    - SRP: 单一职责 - 只负责从控制器获取动作
    - DIP: 依赖ManualController抽象接口
    """
    
    def __init__(self, controller):
        """
        初始化
        
        Args:
            controller: ManualController实例（提供输入接口）
        """
        self.controller = controller
    
    def choose_action(self, observation: Dict, action_mask: Dict) -> Tuple[int, int]:
        """通过控制器获取用户动作"""
        return self.controller.get_human_action(observation, {})
    
    def reset(self):
        """重置策略"""
        pass
