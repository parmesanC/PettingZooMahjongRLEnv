from abc import ABC, abstractmethod

from src.mahjong_rl.core.GameData import GameContext


class IVisualizer(ABC):
    """可视化接口"""
    @abstractmethod
    def render(self, context: GameContext):
        """渲染游戏状态"""
        pass
