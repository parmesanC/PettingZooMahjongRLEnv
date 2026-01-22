from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np

from src.mahjong_rl.core.GameData import GameContext


class IObservationBuilder(ABC):
    """观察构建器接口"""

    @abstractmethod
    def build(self, player_id: int, context: GameContext) -> Dict[str, Any]:
        """构建玩家观察"""
        pass

    @abstractmethod
    def build_action_mask(self, player_id: int, context: GameContext) -> np.ndarray:
        """构建动作掩码"""
        pass

