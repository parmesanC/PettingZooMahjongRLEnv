"""
策略基类（符合RL标准）
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict


class PlayerStrategy(ABC):
    """
    玩家策略基类
    
    符合PettingZoo强化学习标准。
    所有策略（人类、随机、基于规则）都继承此类。
    
    设计原则：
    - SRP: 单一职责 - 只负责选择动作
    - OCP: 开放封闭 - 可扩展新策略
    - LSP: 里氏替换 - 所有策略可互换
    """
    
    @abstractmethod
    def choose_action(self, observation: Dict, action_mask: Dict) -> Tuple[int, int]:
        """
        选择动作
        
        Args:
            observation: 当前观测（包含全局手牌、私有手牌等）
            action_mask: 动作掩码 {'types': np.ndarray, 'params': np.ndarray}
        
        Returns:
            (action_type, parameter) 元组
            - action_type: 0-10 (DISCARD, CHOW, PONG, KONG_*, WIN, PASS)
            - parameter: 牌ID (0-33) 或 吃牌类型 (0-2)
        """
        pass
    
    def reset(self):
        """重置策略状态（新回合开始时调用）"""
        pass
