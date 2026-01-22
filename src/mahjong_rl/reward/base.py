from abc import ABC, abstractmethod
from typing import Dict

from src.mahjong_rl.core.GameData import GameContext


class IRewardSystem(ABC):
    """奖励系统接口"""

    @abstractmethod
    def calculate_reward(self, player_id: int, context: GameContext) -> float:
        """计算玩家奖励"""
        pass

    @abstractmethod
    def calculate_final_rewards(self, context: GameContext) -> Dict[int, float]:
        """计算终局奖励"""
        pass
