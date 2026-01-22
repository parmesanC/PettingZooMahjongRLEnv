from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.core.constants import ActionType, ChowType


@dataclass
class Meld:
    """牌组"""
    action_type: MahjongAction
    tiles: List[int] = field(default_factory=list)
    from_player: int = -1  # 牌的来源玩家

    @property
    def is_kong(self) -> bool:
        """是否为杠"""
        return self.action_type.action_type.value in {ActionType.KONG_EXPOSED.value, ActionType.KONG_SUPPLEMENT.value, ActionType.KONG_CONCEALED.value}

    @property
    def is_pong(self) -> bool:
        """是否为碰"""
        return self.action_type.action_type.value == ActionType.PONG.value

    @property
    def is_chow(self) -> bool:
        """是否为吃"""
        return self.action_type.action_type.value in {ActionType.CHOW.value}

    @property
    def is_opened(self) -> bool:
        """是否为开口"""
        return self.action_type.action_type.value in {ActionType.KONG_EXPOSED.value, ActionType.KONG_SUPPLEMENT.value, ActionType.PONG.value, ActionType.CHOW.value}


@dataclass
class PlayerData:
    """玩家数据"""
    player_id: int
    hand_tiles: List[int] = field(default_factory=list)  # 手牌
    melds: List[Meld] = field(default_factory=list)  # 牌组
    discard_tiles: List[int] = field(default_factory=list)  # 牌河
    special_gangs: List[int] = field(default_factory=lambda: [0, 0, 0])  # [皮子杠数, 赖子杠数, 红中杠数]
    fan_count: int = 0
    is_dealer: bool = False
    is_win: bool = False
    is_ting: bool = False
    action_mask: np.ndarray = None

    def __post_init__(self):
        if self.action_mask is None:
            self.action_mask = np.zeros(46, dtype=np.bool_)

    @property
    def has_opened(self) -> bool:
        """是否有开口"""
        return self.melds is not None and any(m.action_type.action_type.value != ActionType.KONG_CONCEALED.value for m in self.melds)


if __name__ == "__main__":
    melds = [
        Meld(action_type=MahjongAction(ActionType.CHOW, ChowType.LEFT.value), tiles=[1, 2, 3]),
        Meld(action_type=MahjongAction(ActionType.PONG, 4), tiles=[4, 4, 4]),
        Meld(action_type=MahjongAction(ActionType.KONG_EXPOSED, 5), tiles=[5, 5, 5, 5])
    ]

    for meld in melds:
        print(f"{meld.action_type}: {meld.tiles}")
        print(f"is_kong: {meld.is_kong}")
        print(f"is_pong: {meld.is_pong}")
        print(f"is_chow: {meld.is_chow}")
