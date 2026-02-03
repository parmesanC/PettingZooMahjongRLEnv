from typing import List, Optional
from collections import Counter

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import ActionType, ChowType
from src.mahjong_rl.core.mahjong_action import MahjongAction


class ActionValidator:
    """动作检测器 - 基于武汉麻将七皮四赖规则"""

    def __init__(self, game_context: GameContext):
        """
        :param game_context: 游戏上下文
        """
        self.context = game_context
        self.laizi = game_context.lazy_tile
        self.pizi = game_context.skin_tile
        self.red = game_context.red_dragon
        self.special_tiles = self.pizi + [self.red, self.laizi]


    # --------------------------- 对外核心接口 ---------------------------
    def detect_available_actions_after_discard(self, current_player: PlayerData, discard_tile: int, discard_player_idx: int) -> List[MahjongAction]:
        """
        检测当前玩家针对某张弃牌可执行的动作
        :param current_player: 当前响应玩家
        :param discard_tile: 弃牌编码
        :param discard_player_idx: 弃牌玩家索引
        :return: 可执行的动作类型列表
        """

        # 0. 牌墙为空时，不允许吃碰杠胡（直接流局）
        if len(self.context.wall) == 0:
            return []

        # 1. 排除自身弃牌（不能对自己的弃牌执行吃碰杠）
        if current_player.player_id == discard_player_idx:
            return []

        # 2. 弃牌如果是红中、赖子、皮子则不能进行吃碰杠
        if discard_tile in self.special_tiles:
            return []

        available_actions = []

        # 3. 判断吃牌（优先级最低）
        chow_actions = self._can_chow(current_player, discard_tile, discard_player_idx)
        available_actions.extend(chow_actions)

        # 4. 判断碰牌
        if self._can_pong(current_player, discard_tile):
            available_actions.append(MahjongAction(ActionType.PONG, discard_tile))

        # 5. 判断明杠（优先级最高）
        if self._can_kong_exposed(current_player, discard_tile):
            available_actions.append(MahjongAction(ActionType.KONG_EXPOSED, discard_tile))

        # 6. 判断接炮胡牌（优先级最高，高于所有吃碰杠）
        if self._can_win_by_discard(current_player, discard_tile):
            available_actions.append(MahjongAction(ActionType.WIN, -1))

        return available_actions

    # --------------------------- 核心动作判断逻辑 ---------------------------
    def _can_chow(self, current_player: PlayerData, discard_tile: int, discard_player_idx: int) -> List[MahjongAction]:
        """判断是否可以吃牌
        规则：
        1. 只能吃上家弃牌
        2. 弃牌必须是序数牌（万条筒），且非特殊牌
        3. 手牌中存在两张牌，与弃牌组成同花色顺子
        4. 吃的类型分为左/中/右三种
        """
        available_chows = []
        current_idx = current_player.player_id
        upper_idx = self._get_upper_player_idx(current_idx)

        # 条件1：不是上家弃牌 → 不能吃
        if discard_player_idx != upper_idx:
            return available_chows

        # 条件2：弃牌是特殊牌或非序数牌 → 不能吃
        discard_suit = discard_tile // 9
        discard_val = discard_tile % 9 + 1
        if discard_tile in self.special_tiles or discard_suit > 3:
            return available_chows

        # 过滤手牌：移除特殊牌
        valid_hand = self._filter_valid_hand_tiles(current_player.hand_tiles)
        if len(valid_hand) < 2:
            return available_chows

        # 条件3：寻找能组成顺子的两张手牌
        # 生成所有可能的顺子组合：弃牌作为左/中/右的情况
        possible_combinations = [
            # 弃牌为左 (val, val+1, val+2) → 需手牌有 val+1, val+2
            (ChowType.LEFT, [discard_val + 1, discard_val + 2]),
            # 弃牌为中 (val-1, val, val+1) → 需手牌有 val-1, val+1
            (ChowType.MIDDLE, [discard_val - 1, discard_val + 1]),
            # 弃牌为右 (val-2, val-1, val) → 需手牌有 val-2, val-1
            (ChowType.RIGHT, [discard_val - 2, discard_val - 1])
        ]

        # 遍历所有可能的吃型
        for chow_type, required_vals in possible_combinations:
            # 检查数值是否合法（序数牌数值1-9）
            if any(v < 1 or v > 9 for v in required_vals):
                continue
            # 检查手牌是否包含这两个数值，且花色相同
            required_cards = [v + discard_suit * 9 - 1 for v in required_vals]
            if all(t in valid_hand for t in required_cards):
                available_chows.append(MahjongAction(ActionType.CHOW, chow_type.value))

        return available_chows

    def _can_pong(self, current_player: PlayerData, discard_tile: int) -> bool:
        """判断是否可以碰牌
        规则：
        1. 弃牌非特殊牌
        2. 手牌中存在至少两张与弃牌相同的牌
        """
        # 条件1：弃牌是特殊牌 → 不能碰
        if discard_tile in self.special_tiles:
            return False

        # 条件2：过滤手牌后，相同牌数量≥2
        valid_hand = self._filter_valid_hand_tiles(current_player.hand_tiles)
        return valid_hand.count(discard_tile) >= 2

    def _can_kong_exposed(self, current_player: PlayerData, discard_tile: int) -> bool:
        """判断是否可以明杠（冲杠）
        规则：
        1. 弃牌非特殊牌
        2. 手牌中存在至少三张与弃牌相同的牌
        """
        # 条件1：弃牌是特殊牌 → 不能杠
        if discard_tile in self.special_tiles:
            return False

        # 条件2：过滤手牌后，相同牌数量≥3
        valid_hand = self._filter_valid_hand_tiles(current_player.hand_tiles)
        return valid_hand.count(discard_tile) >= 3

    def _get_upper_player_idx(self, current_idx: int) -> int:
        """获取当前玩家的上家索引（4人桌：逆时针）"""
        return (current_idx - 1) % len(self.context.players)

    def _filter_valid_hand_tiles(self, hand_tiles: List[int]) -> List[int]:
        """过滤手牌：移除特殊牌，仅保留可参与吃碰杠的牌"""
        return [t for t in hand_tiles if not t in self.special_tiles]


    def detect_available_actions_after_draw(self, current_player: PlayerData, draw_tile: Optional[int]) -> List[MahjongAction]:
        """
        检测当前玩家针对某张摸牌可执行的动作

        Args:
            current_player: 当前响应玩家
            draw_tile: 摸牌编码（如果为 None，表示不指定特定牌，仅基于手牌检测）

        Returns:
            可执行的动作类型列表
        """
        temp_hand = current_player.hand_tiles.copy()

        temp_hand_counter = Counter(temp_hand)
        melds = current_player.melds.copy()

        available_actions = []
        # 1. 可以打的牌、以及可杠的牌
        for card, count in temp_hand_counter.items():
            if count > 0:
                if card not in self.special_tiles:
                    available_actions.append(MahjongAction(ActionType.DISCARD, card))
                else:
                    if card == self.context.lazy_tile:
                        available_actions.append(MahjongAction(ActionType.KONG_LAZY, 0))
                    elif card == self.context.red_dragon:
                        available_actions.append(MahjongAction(ActionType.KONG_RED, 0))
                    elif card in self.context.skin_tile:
                        available_actions.append(MahjongAction(ActionType.KONG_SKIN, card))
            if count >= 4:
                available_actions.append(MahjongAction(ActionType.KONG_CONCEALED, card))

            if count > 0 and any(meld.tiles[0] == card for meld in melds if meld.is_pong):
                available_actions.append(MahjongAction(ActionType.KONG_SUPPLEMENT, card))

        # 3. 检查是否和牌
        temp_hand = current_player.hand_tiles.copy()

        temp_player = PlayerData(
            player_id=current_player.player_id,
            hand_tiles=temp_hand,
            melds=current_player.melds.copy(),
            special_gangs=current_player.special_gangs.copy()
        )
        from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.win_detector import WuhanMahjongWinChecker
        win_checker = WuhanMahjongWinChecker(self.context)
        result = win_checker.check_win(temp_player)
        if result.can_win:
            # 修复问题3：调用 score_calculator 的方法检查起胡番
            from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.score_calculator import MahjongScoreSettler
            score_calculator = MahjongScoreSettler(False)
            can_win = score_calculator.check_min_fan_requirement(
                current_player.player_id, result.win_type, self.context
            )
            if can_win:
                available_actions.append(MahjongAction(ActionType.WIN, -1))
            return available_actions

        return available_actions

    def detect_available_actions_after_meld(self, current_player: PlayerData) -> List[MahjongAction]:
        """
        检测鸣牌后可用的动作

        鸣牌后不能胡牌，只能：出牌、暗杠、红中杠、皮子杠、赖子杠、补杠

        Args:
            current_player: 当前玩家数据

        Returns:
            可用动作列表
        """
        available_actions = []
        temp_hand = current_player.hand_tiles.copy()

        # 1. 检测所有可打出的牌（排除特殊牌）
        for tile in temp_hand:
            # 不能打特殊牌（赖子、皮、红中）
            if tile in self.special_tiles:
                continue

            available_actions.append(MahjongAction(ActionType.DISCARD, tile))

        # 2. 检测暗杠（手牌中有4张相同的牌）
        tile_counts = Counter(temp_hand)
        for tile, count in tile_counts.items():
            if count >= 4 and tile not in self.special_tiles:
                available_actions.append(MahjongAction(ActionType.KONG_CONCEALED, tile))

        # 3. 检测红中杠
        if current_player.hand_tiles.count(self.context.red_dragon) >= 1:
            available_actions.append(MahjongAction(ActionType.KONG_RED, 0))

        # 4. 检测皮子杠
        for skin_tile in self.context.skin_tile:
            if skin_tile != -1 and current_player.hand_tiles.count(skin_tile) >= 1:
                available_actions.append(MahjongAction(ActionType.KONG_SKIN, skin_tile))

        # 5. 检测赖子杠
        if self.context.lazy_tile is not None and current_player.hand_tiles.count(self.context.lazy_tile) >= 1:
            available_actions.append(MahjongAction(ActionType.KONG_LAZY, 0))

        # 6. 检测补杠（如果有碰牌，手牌有第4张）
        for meld in current_player.melds:
            if meld.action_type.action_type == ActionType.PONG:
                pong_tile = meld.tiles[0]
                if pong_tile in current_player.hand_tiles:
                    available_actions.append(MahjongAction(ActionType.KONG_SUPPLEMENT, pong_tile))

        return available_actions

    def _can_win_by_discard(self, current_player: PlayerData, discard_tile: int) -> bool:
        """
        判断是否可以接炮胡牌

        将弃牌临时加入玩家手牌，检查是否可以胡牌，并检查起胡番要求。

        Args:
            current_player: 当前玩家
            discard_tile: 弃牌编码

        Returns:
            True 如果可以接炮胡牌
        """
        # 创建临时手牌（加入弃牌）
        temp_hand = current_player.hand_tiles.copy()
        temp_hand.append(discard_tile)

        # 创建临时玩家对象
        temp_player = PlayerData(
            player_id=current_player.player_id,
            hand_tiles=temp_hand,
            melds=current_player.melds.copy(),
            special_gangs=current_player.special_gangs.copy()
        )

        # 检查是否可以胡牌
        from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.win_detector import WuhanMahjongWinChecker
        win_checker = WuhanMahjongWinChecker(self.context)
        result = win_checker.check_win(temp_player)

        if not result.can_win:
            return False

        # 检查起胡番要求
        from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.score_calculator import MahjongScoreSettler
        score_calculator = MahjongScoreSettler(False)
        return score_calculator.check_min_fan_requirement(
            current_player.player_id, result.win_type, self.context
        )

