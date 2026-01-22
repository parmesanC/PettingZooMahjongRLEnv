from collections import Counter
from typing import Any

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import ActionType
from src.mahjong_rl.rules.base import IRuleEngine


class SimpleRuleEngine(IRuleEngine):
    def can_chow(self, player_id: int, tile: int, context: Any) -> bool:
        return False

    def can_kong(self, player_id: int, tile: int, context: Any) -> bool:
        return False

    def can_pong(self, player_id: int, tile: int, context: Any) -> bool:
        return False

    def can_win(self, player_id: int, tile: int, context: Any, is_self_draw: bool) -> bool:
        return False

    def valid_kong_actions_after_draw(self, player_id: int, context: GameContext, draw_tile: int) -> list[int]:
        kong_actions = []
        player_hand = context.players[player_id].hand_tiles
        hand_count = Counter(player_hand)

        # 检查是否有可以暗杠的牌
        valid_kong_tiles = [t for t in hand_count if hand_count[t] == 4]
        if valid_kong_tiles:
            kong_actions.append(ActionType.KONG_EXPOSED.value)

        # 检查是否有可以补杠的牌
        hand_tiles_set = set(player_hand)
        for meld in context.players[player_id].melds:
            if meld.action_type == ActionType.PONG and set(meld.tiles) & hand_tiles_set:
                kong_actions.append(ActionType.KONG_SUPPLEMENT.value)

        # 红中杠
        if context.red_dragon in hand_tiles_set:
            kong_actions.append(ActionType.KONG_RED.value)

        # 赖子杠
        if context.lazy_tile in hand_tiles_set:
            kong_actions.append(ActionType.KONG_LAZY.value)

        # 皮子杠
        if set(context.skin_tile) & hand_tiles_set:
            kong_actions.append(ActionType.KONG_SKIN.value)

        return kong_actions

    def valid_chow_actions(self, player_id: int, context: GameContext, draw_tile: int) -> list[int]:
        chow_actions = []
        player_hand = context.players[player_id].hand_tiles
        hand_tiles_set = set(player_hand)
        # 左吃
        if draw_tile + 1 in hand_tiles_set and draw_tile + 2 in hand_tiles_set:
            chow_actions.append(ActionType.CHOW_LEFT.value)

        # 中吃
        if draw_tile - 1 in hand_tiles_set and draw_tile + 1 in hand_tiles_set:
            chow_actions.append(ActionType.CHOW_MIDDLE.value)

        # 右吃
        if draw_tile - 1 in hand_tiles_set and draw_tile - 2 in hand_tiles_set:
            chow_actions.append(ActionType.CHOW_RIGHT.value)


