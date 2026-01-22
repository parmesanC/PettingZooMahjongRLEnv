from typing import List

from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.rules.base import IRuleEngine
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.action_validator import ActionValidator
from src.mahjong_rl.core.GameData import GameContext

"""武汉麻将七皮四赖子玩法规则"""


class Wuhan7P4LRuleEngine(IRuleEngine):
    def __init__(self, game_context: GameContext):
        self.action_validator = ActionValidator(game_context)

    def detect_available_actions_after_draw(self, current_player: PlayerData, draw_tile: int) -> List[MahjongAction]:
        """
        检测玩家摸牌后可用的动作

        Args:
            current_player: 当前玩家数据
            draw_tile: 摸到的牌

        Returns:
            可用动作列表
        """
        return self.action_validator.detect_available_actions_after_draw(current_player, draw_tile)

    def detect_available_actions_after_discard(self, current_player: PlayerData, discard_tile: int, discard_player_idx: int) -> List[MahjongAction]:
        """
        检测玩家对其他玩家弃牌后的可用动作

        Args:
            current_player: 当前玩家数据
            discard_tile: 弃牌
            discard_player_idx: 弃牌玩家索引

        Returns:
            可用动作列表
        """
        return self.action_validator.detect_available_actions_after_discard(current_player, discard_tile, discard_player_idx)
