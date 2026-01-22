from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np

from src.mahjong_rl.observation.builder import IObservationBuilder
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import ActionType, GameStateType
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.action_validator import ActionValidator


# action_mask 索引范围定义（总长度：145位）
ACTION_MASK_RANGES = {
    'DISCARD': (0, 34),          # 0-33: 可打出的牌（不包含特殊牌）
    'CHOW': (34, 37),            # 34-36: 左吃/中吃/右吃
    'PONG': (37, 38),            # 37: 是否可碰
    'KONG_EXPOSED': (38, 39),    # 38: 是否可明杠（1位）
    'KONG_SUPPLEMENT': (39, 73),  # 39-72: 可补杠的牌
    'KONG_CONCEALED': (73, 107),  # 73-106: 可暗杠的牌
    'KONG_RED': (107, 108),      # 107: 是否可红中杠（1位，ActionType.KONG_RED=6）
    'KONG_SKIN': (108, 142),     # 108-141: 皮子杠（34位，ActionType.KONG_SKIN=7）
    'KONG_LAZY': (142, 143),     # 142: 赖子杠（1位，ActionType.KONG_LAZY=8）
    'WIN': (143, 144),           # 143: 是否可胡
    'PASS': (144, 145),          # 144: 是否可过
}

STATE_TO_PHASE = {
    GameStateType.INITIAL: 0,
    GameStateType.DRAWING: 1,
    GameStateType.PLAYER_DECISION: 2,      # 摸牌后决策（可杠/胡/出牌）
    GameStateType.DISCARDING: 3,
    GameStateType.MELD_DECISION: 4,        # 鸣牌后决策（可杠/出牌，不能胡）
    GameStateType.WAITING_RESPONSE: 5,
    GameStateType.RESPONSES: 6,
    GameStateType.PROCESSING_MELD: 7,
    GameStateType.GONG: 8,
    GameStateType.WAIT_ROB_KONG: 9,
    GameStateType.DRAWING_AFTER_GONG: 10,
    GameStateType.RESPONSES_AFTER_GONG: 11,
    GameStateType.WIN: 12,
    GameStateType.FLOW_DRAW: 13,
    GameStateType.SETTLE: 14
}


class Wuhan7P4LObservationBuilder(IObservationBuilder):
    """武汉麻将七皮四赖子观察构建器"""

    def __init__(self, context: Optional[GameContext] = None):
        self.context = context

    def build(self, player_id: int, context: GameContext) -> Dict[str, Any]:
        """构建玩家观察"""
        observation = {}

        observation['global_hand'] = self._build_global_hand(context.players)
        observation['private_hand'] = self._build_private_hand(context.players[player_id].hand_tiles)
        observation['discard_pool_total'] = self._build_discard_pool_total(context.discard_pile)
        observation['wall'] = self._build_wall(context.wall)
        observation['melds'] = self._build_melds(context.players)
        observation['action_history'] = self._build_action_history(context.action_history)
        observation['special_gangs'] = self._build_special_gangs(context.players)
        observation['current_player'] = np.array([context.current_player_idx], dtype=np.int8)
        observation['fan_counts'] = self._build_fan_counts(context.players)
        observation['special_indicators'] = self._build_special_indicators(context)
        observation['remaining_tiles'] = len(context.wall)
        observation['dealer'] = np.array([context.dealer_idx], dtype=np.int8)
        observation['current_phase'] = self._map_game_state_to_phase(context.current_state)

        # 添加action_mask
        observation['action_mask'] = self.build_action_mask(player_id, context)

        observation = self._apply_visibility_mask(observation, player_id, context)

        return observation

    def build_action_mask(self, player_id: int, context: GameContext) -> np.ndarray:
        """
        构建动作掩码 - 返回扁平化的145位二进制数组

        Returns:
            np.ndarray: 形状为 (145,) 的二进制数组
        """
        mask = np.zeros(145, dtype=np.int8)

        current_state = context.current_state
        player = context.players[player_id]

        # 检查手牌是否已正确初始化（至少有1张牌）
        # 鸣牌后手牌可能少于13张（吃牌11张，碰牌10张）
        if len(player.hand_tiles) < 1:
            return mask

        if current_state == GameStateType.MELD_DECISION:
            # 鸣牌后决策：可以杠、出牌，但不能胡
            mask = self._build_meld_decision_mask(player, context, mask)

        elif current_state in [GameStateType.PLAYER_DECISION, GameStateType.DRAWING]:
            # 摸牌后决策：可以杠、胡、出牌
            mask = self._build_decision_mask(player, context, mask)

        elif current_state in [GameStateType.WAITING_RESPONSE, GameStateType.RESPONSES,
                               GameStateType.RESPONSES_AFTER_GONG]:
            # 响应状态
            mask = self._build_response_mask(player, context, mask)

        return mask

    def _build_decision_mask(self, player, context, mask):
        """
        构建摸牌后决策的动作掩码（PLAYER_DECISION）

        可以：暗杠、补杠、红中杠、赖子杠、皮子杠、胡牌、出牌
        不能：过

        修改：不再依赖 last_drawn_tile，直接基于手牌检测
        """
        # 不再依赖 last_drawn_tile，传入 None 让 ActionValidator 基于手牌检测
        actions = ActionValidator(context).detect_available_actions_after_draw(
            player, None
        )

        for action in actions:
            action_type = action.action_type.value

            if action_type == ActionType.DISCARD.value:
                # DISCARD: 标记手牌中所有可打出的牌 (索引 0-33)
                # 但不包含特殊牌（赖子、皮子、红中），特殊牌只能通过杠动作处理
                hand_counts = self._get_hand_counts(player.hand_tiles)
                special_tiles = [context.lazy_tile, context.red_dragon] + context.skin_tile
                for tile_id in range(34):
                    if tile_id not in special_tiles and hand_counts[tile_id] > 0:
                        mask[tile_id] = 1

            elif action_type == ActionType.CHOW.value:
                # CHOW: 标记具体吃法 (索引 34-36)
                chow_param = action.parameter  # 0=左, 1=中, 2=右
                mask[34 + chow_param] = 1

            elif action_type == ActionType.PONG.value:
                mask[37] = 1  # PONG 位

            elif action_type == ActionType.KONG_EXPOSED.value:
                mask[38] = 1  # KONG_EXPOSED 位（1位）

            elif action_type == ActionType.KONG_SUPPLEMENT.value:
                mask[39 + action.parameter] = 1

            elif action_type == ActionType.KONG_CONCEALED.value:
                mask[73 + action.parameter] = 1

            elif action_type == ActionType.KONG_RED.value:
                # 红中杠：1位（全场只有一张红中）
                mask[107] = 1

            elif action_type == ActionType.KONG_SKIN.value:
                # 皮子杠：34位（两张皮子是独立的）
                mask[108 + action.parameter] = 1

            elif action_type == ActionType.KONG_LAZY.value:
                # 赖子杠：1位（全场只有一张赖子）
                lazy_tile = context.lazy_tile
                if lazy_tile is not None:
                    mask[142] = 1

            elif action_type == ActionType.WIN.value:
                mask[143] = 1  # WIN 位

        # 确保 DISCARD 可用（后备逻辑，同样需要排除特殊牌）
        if not np.any(mask[:34] > 0):
            special_tiles = [context.lazy_tile, context.red_dragon] + context.skin_tile
            for tile in player.hand_tiles:
                if 0 <= tile < 34 and tile not in special_tiles:
                    mask[tile] = 1

        return mask

    def _build_meld_decision_mask(self, player, context, mask):
        """
        构建鸣牌后决策的动作掩码（MELD_DECISION）

        可以：暗杠、补杠、红中杠、赖子杠、皮子杠、出牌
        不能：胡牌、过
        """
        # 不依赖 last_drawn_tile，直接基于手牌检测
        actions = ActionValidator(context).detect_available_actions_after_draw(
            player, None
        )

        for action in actions:
            action_type = action.action_type.value

            # 处理杠动作（所有杠都允许）
            if action_type == ActionType.KONG_SUPPLEMENT.value:
                mask[39 + action.parameter] = 1

            elif action_type == ActionType.KONG_CONCEALED.value:
                mask[73 + action.parameter] = 1

            elif action_type == ActionType.KONG_RED.value:
                # 红中杠：1位（全场只有一张红中）
                mask[107] = 1

            elif action_type == ActionType.KONG_SKIN.value:
                # 皮子杠：34位（两张皮子是独立的）
                mask[108 + action.parameter] = 1

            elif action_type == ActionType.KONG_LAZY.value:
                # 赖子杠：1位（全场只有一张赖子）
                lazy_tile = context.lazy_tile
                if lazy_tile is not None:
                    mask[142] = 1

            # 处理 DISCARD（同样需要排除特殊牌）
            elif action_type == ActionType.DISCARD.value:
                hand_counts = self._get_hand_counts(player.hand_tiles)
                special_tiles = [context.lazy_tile, context.red_dragon] + context.skin_tile
                for tile_id in range(34):
                    if tile_id not in special_tiles and hand_counts[tile_id] > 0:
                        mask[tile_id] = 1

            # 忽略 WIN、CHOW、PONG、KONG_EXPOSED 动作（鸣牌后不允许）

        # 确保 DISCARD 可用（后备逻辑，同样需要排除特殊牌）
        if not np.any(mask[:34] > 0):
            special_tiles = [context.lazy_tile, context.red_dragon] + context.skin_tile
            for tile in player.hand_tiles:
                if 0 <= tile < 34 and tile not in special_tiles:
                    mask[tile] = 1

        return mask

    def _build_response_mask(self, player, context, mask):
        """构建响应状态的动作掩码"""
        discard_tile = context.last_discarded_tile
        discard_player = context.discard_player

        if discard_tile is not None and discard_player is not None:
            actions = ActionValidator(context).detect_available_actions_after_discard(
                player, discard_tile, discard_player
            )

            for action in actions:
                action_type = action.action_type.value

                if action_type == ActionType.CHOW.value:
                    mask[34 + action.parameter] = 1

                elif action_type == ActionType.PONG.value:
                    mask[37] = 1

                elif action_type == ActionType.KONG_EXPOSED.value:
                    mask[38] = 1  # KONG_EXPOSED 位（1位，因为明杠哪张牌是确定的）

                elif action_type == ActionType.WIN.value:
                    mask[143] = 1  # WIN 位

            # PASS 在响应状态总是可用
            mask[144] = 1  # PASS 位

        return mask

    def _get_hand_counts(self, hand_tiles: List[int]) -> np.ndarray:
        """获取手牌中每种牌的数量"""
        counts = np.zeros(34, dtype=np.int8)
        for tile in hand_tiles:
            if 0 <= tile < 34:
                counts[tile] = min(counts[tile] + 1, 4)
        return counts

    def _build_hand_array(self, hand_tiles: List[int]) -> np.ndarray:
        counts = np.zeros(34, dtype=np.int8)
        for tile in hand_tiles:
            counts[tile] = min(counts[tile] + 1, 5)
        return counts

    def _build_global_hand(self, players: List) -> np.ndarray:
        global_hand = []
        for player in players:
            hand_array = self._build_hand_array(player.hand_tiles)
            global_hand.extend(hand_array)
        return np.array(global_hand, dtype=np.int8)

    def _build_private_hand(self, hand_tiles: List[int]) -> np.ndarray:
        return self._build_hand_array(hand_tiles)

    def _build_discard_pool_total(self, discard_pile: List[int]) -> np.ndarray:
        counts = np.zeros(34, dtype=np.int8)
        for tile in discard_pile:
            if 0 <= tile < 34:
                counts[tile] = min(counts[tile] + 1, 5)
        return counts

    def _build_wall(self, wall: deque) -> np.ndarray:
        wall_array = np.full(82, 34, dtype=np.int8)
        for i, tile in enumerate(wall):
            if i < 82:
                wall_array[i] = tile
        return wall_array

    def _build_melds(self, players: List) -> Dict[str, np.ndarray]:
        action_types = np.zeros(16, dtype=np.int8)
        tiles_one_hot = np.zeros(256, dtype=np.int8)
        group_indices = np.zeros(32, dtype=np.int8)

        for player_id, player in enumerate(players):
            for meld_id, meld in enumerate(player.melds[:4]):
                idx = player_id * 4 + meld_id
                action_types[idx] = meld.action_type.action_type.value

                for tile_pos, tile in enumerate(meld.tiles[:4]):
                    one_hot_idx = (player_id * 4 * 4 + meld_id * 4 + tile_pos) * 34 + tile
                    if 0 <= one_hot_idx < 256:
                        tiles_one_hot[one_hot_idx] = 1

                group_indices[idx*2] = player_id
                group_indices[idx*2+1] = meld_id

        return {
            'action_types': action_types,
            'tiles': tiles_one_hot,
            'group_indices': group_indices
        }

    def _build_action_history(self, action_history: List) -> Dict[str, np.ndarray]:
        types = np.zeros(80, dtype=np.int8)
        params = np.zeros(80, dtype=np.int8)
        players = np.zeros(80, dtype=np.int8)

        recent_history = action_history[-80:]
        for i, record in enumerate(recent_history):
            types[i] = record.action_type.action_type.value
            params[i] = record.tile if record.tile is not None and 0 <= record.tile < 34 else 34
            players[i] = record.player_id

        return {
            'types': types,
            'params': params,
            'players': players
        }

    def _build_special_gangs(self, players: List) -> np.ndarray:
        special_gangs = []
        for player in players:
            pi_gang = min(player.special_gangs[0], 7)
            lai_gang = min(player.special_gangs[1], 3)
            zhong_gang = min(player.special_gangs[2], 4)
            special_gangs.extend([pi_gang, lai_gang, zhong_gang])
        return np.array(special_gangs, dtype=np.int8)

    def _build_fan_counts(self, players: List) -> np.ndarray:
        fan_counts = []
        for player in players:
            fan_count = min(player.fan_count, 599)
            fan_counts.append(fan_count)
        return np.array(fan_counts, dtype=np.int16)

    def _build_special_indicators(self, context: GameContext) -> np.ndarray:
        lazy_tile = context.lazy_tile if context.lazy_tile is not None else 33
        skin_tile = context.skin_tile[0] if context.skin_tile and context.skin_tile[0] != -1 else 33
        return np.array([lazy_tile, skin_tile], dtype=np.int8)

    def _map_game_state_to_phase(self, game_state: GameStateType) -> int:
        return STATE_TO_PHASE.get(game_state, 7)

    def _apply_visibility_mask(self, observation: Dict, player_id: int, context: GameContext) -> Dict:
        training_phase = getattr(context, 'training_phase', 3)
        progress = getattr(context, 'training_progress', 1.0)

        if training_phase == 1:
            pass
        elif training_phase == 2:
            visibility = 1.0 - progress
            if np.random.random() > visibility:
                for i in range(4):
                    if i != player_id:
                        observation['global_hand'][i*34:(i+1)*34] = 0
                observation['wall'].fill(34)
        elif training_phase == 3:
            for i in range(4):
                if i != player_id:
                    observation['global_hand'][i*34:(i+1)*34] = 0
            observation['wall'].fill(34)

        return observation
