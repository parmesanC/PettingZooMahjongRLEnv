from abc import ABC, abstractmethod
from typing import List

from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.core.GameData import GameContext


class IRuleEngine(ABC):
    """规则引擎接口 - 麻将游戏逻辑"""

    @abstractmethod
    def detect_available_actions_after_draw(self,current_player: PlayerData, draw_tile: int) -> List[MahjongAction]:
        """检测玩家摸牌后可用的动作"""
        pass

    @abstractmethod
    def detect_available_actions_after_discard(self, player_id: int, context: GameContext, discard_tile: int) -> list[int]:
        """检测玩家碰牌后可用的动作"""
        pass
