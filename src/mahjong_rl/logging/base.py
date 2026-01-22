from abc import ABC, abstractmethod
from typing import Optional

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType, ActionType


class ILogger(ABC):
    """日志接口"""

    @abstractmethod
    def log_state_transition(self, from_state: GameStateType, to_state: GameStateType, context: GameContext):
        """记录状态转移"""
        pass

    @abstractmethod
    def log_action(self, player_id: int, action_type: ActionType, tile: Optional[int], context: GameContext):
        """记录玩家动作"""
        pass