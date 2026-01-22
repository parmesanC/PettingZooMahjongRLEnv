import os
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
from collections import Counter, deque
from multiprocessing import cpu_count, Pool

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData, Meld
from src.mahjong_rl.core.constants import WinType, ActionType, Tiles, WinWay

import mahjong_win_checker as mjc


# checker = mjc.MahjongWinChecker()
# def test_basic_win():
#     # 普通胡牌手牌（清一色筒子）
#     hand = [0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8]
#     success, min_laizi = checker.get_min_laizi_needed(hand, wild_index=0, unlimited_eye=True)
#     print(f"普通胡牌: {success}, 最小赖子: {min_laizi}")  # 应该返回 (True, 0)

@dataclass
class WinCheckResult:
    """胡牌判定结果"""
    can_win: bool = False
    win_type: List[WinType] = field(default_factory=list)
    min_wild_need: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)


class WuhanMahjongWinChecker:
    """胡牌判定引擎"""

    # 花色定义
    CHARACTERS = set(range(0, 9))  # 万: 0-8
    BAMBOOS = set(range(9, 18))  # 条: 9-17
    DOTS = set(range(18, 27))  # 筒: 18-26
    WINDS = set(range(27, 31))  # 东南西北: 27-30
    DRAGONS = set(range(31, 34))  # 中发白: 31-33
    ALL_WINDS_DRAGONS = set(range(27, 34))  # 所有风牌
    ALL_JIANG = {1, 4, 7, 10, 13, 16, 19, 22, 25}

    def __init__(self, game_context: GameContext):
        """初始化引擎组件"""
        self.checker = mjc.MahjongWinChecker()
        self.game_context = game_context

    def check_ting(self, player_data: PlayerData) -> bool:
        """检查玩家是否听牌"""

        # 检查手牌是否合法
        if len(player_data.hand_tiles) not in {1, 4, 7, 10, 13}:
            raise ValueError("Invalid hand tiles length")

        # 检查是否有开口
        if not player_data.has_opened:
            player_data.is_ting = False
            return False

        # 检查是否有红中、皮子在手牌中
        if (self.game_context.skin_tile[0] in player_data.hand_tiles
                or self.game_context.skin_tile[1] in player_data.hand_tiles
                or self.game_context.red_dragon in player_data.hand_tiles):
            player_data.is_ting = False
            return False

        # 检查是否听牌
        for i in range(33):
            hand_copy = player_data.hand_tiles.copy()
            hand_copy.append(i)
            result = self.check_win(player_data)
            if result.can_win:
                player_data.is_ting = True
                return True

        player_data.is_ting = False
        return False

    def check_win(self, player_data: PlayerData) -> WinCheckResult:
        """检查玩家是否胡牌"""

        result = WinCheckResult()

        # 检查手牌是否合法
        if len(player_data.hand_tiles) not in {2, 5, 8, 11, 14}:
            raise ValueError("Invalid hand tiles length")

        # 检查是否有开口
        if not player_data.has_opened:
            result.detail["error"] = "No opened melds"
            return result

        # 胡牌时不能有皮或红中在手牌
        # 检查是否有皮子和红中
        if (self.game_context.skin_tile[0] in player_data.hand_tiles
                or self.game_context.skin_tile[1] in player_data.hand_tiles
                or self.game_context.red_dragon in player_data.hand_tiles):
            result.detail["error"] = "Win hands cannot have skin tile or red dragon"
            return result

        # 检查是否大胡
        hands_bk = player_data.hand_tiles.copy()
        melds_bk = deepcopy(player_data.melds)
        big_wins_dict = self.detect_big_wins(hands_bk, melds_bk)

        # 检查是否有大胡
        big_win_flag = False
        for win_type, (is_win, min_laizi) in big_wins_dict.items():
            if is_win:
                result.can_win = True
                result.win_type.append(win_type)
                result.min_wild_need = max(result.min_wild_need, min_laizi)
                big_win_flag = True
                if result.min_wild_need == 0 and WinType.HARD_WIN not in result.win_type:
                    result.win_type.append(WinType.HARD_WIN)

        # 如果没有大胡，检查是否有普通胡牌
        if not big_win_flag:
            success, min_laizi = self.checker.get_min_laizi_needed(hands_bk, wild_index=self.game_context.lazy_tile, unlimited_eye=False)
            if success and min_laizi == 0:
                result.can_win = True
                result.win_type.append(WinType.HARD_WIN)
                result.min_wild_need = max(result.min_wild_need, min_laizi)
            elif success and min_laizi > 0:
                result.can_win = True
                result.win_type.append(WinType.SOFT_WIN)
                result.min_wild_need = max(result.min_wild_need, min_laizi)

        return result

    def detect_big_wins(self, hand_tiles: List[int], melds: List[Meld]) -> Dict[WinType, Tuple[bool, int]]:
        """
        检测所有满足的大胡牌型

        Args:
            hand_tiles: 手牌列表
            melds: 鸣牌列表

        Returns:
            字典: {WinType: (是否满足, 最小赖子需求数量)}
            赖子需求数量仅统计作为万能牌的数量，不包括自身还原
        """
        big_wins = {}

        all_tiles = hand_tiles.copy()
        for meld in melds:
            all_tiles.extend(meld.tiles)

        # 检测每种大胡
        # 1. 风一色
        big_wins[WinType.ALL_WIND] = self._check_all_winds(all_tiles)

        # 2. 将一色
        big_wins[WinType.ALL_JIANG] = self._check_all_jiang(all_tiles)

        # 3. 清一色 (需要检查是否能胡牌)
        big_wins[WinType.PURE_FLUSH] = self._check_pure_flush(all_tiles, hand_tiles)

        # 4. 碰碰胡 (需要检查是否能胡牌)
        big_wins[WinType.PENG_PENG_HU] = self._check_peng_peng_hu(hand_tiles, melds)

        # 5. 全求人 (需要特殊条件)
        big_wins[WinType.FULLY_MELDED] = self._check_fully_melded(hand_tiles, melds)

        # 6. 杠上开花 (需要特殊条件)
        big_wins[WinType.FLOWER_ON_KONG] = self._check_flower_on_kong(hand_tiles)

        # 7. 海底捞月 (需要特殊条件)
        big_wins[WinType.LAST_TILE_WIN] = self._check_last_tile_win(hand_tiles)

        # 8. 抢杠胡 (需要特殊条件)
        big_wins[WinType.ROB_KONG] = self._check_rob_kong(hand_tiles)

        return big_wins

    # ==================== 大胡检测方法 ====================

    def _check_all_winds(self, all_tiles: List[int]) -> Tuple[bool, int]:
        """
        检测风一色
        条件：所有牌都是风牌（东南西北发白），不包含红中

        正确理解：
        1. 风一色要求所有牌都是风牌（东南西北发白）
        2. 如果有非风牌（万条筒），则不可能成立，因为赖子只能补缺不能改变牌的本质
        3. 赖子可以作为任何风牌（补缺），但不能把非风牌变成风牌

        返回：(是否满足, 最小赖子需求)
        """
        # 统计牌型
        laizi_count = 0

        for tile in all_tiles:
            if tile == self.game_context.lazy_tile:
                laizi_count += 1
            elif tile not in self.ALL_WINDS_DRAGONS:
                return False, -1
        else:
            if self.game_context.lazy_tile in self.ALL_WINDS_DRAGONS:
                return True, 0
            elif laizi_count <= 2:
                return True, laizi_count
            else:
                return False, -1

    def _check_all_jiang(self, all_tiles: List[int]) -> Tuple[bool, int]:
        """
        检测将一色
        条件：所有牌都是将牌（中发白），不包含红中

        返回：(是否满足, 最小赖子需求)
        """
        # 统计牌型
        laizi_count = 0

        for tile in all_tiles:
            if tile == self.game_context.lazy_tile:
                laizi_count += 1
            elif tile not in self.ALL_JIANG:
                return False, -1
        else:
            if self.game_context.lazy_tile in self.ALL_JIANG:
                return True, 0
            elif laizi_count <= 2:
                return True, laizi_count
            else:
                return False, -1

    def _check_pure_flush(self, all_tiles: List[int], hand: List[int]) -> Tuple[bool, int]:
        """
        检测清一色
        条件：所有牌为同一花色（万/条/筒），且满足标准胡牌结构
        规则：赖子可以补缺，不能把风牌变成数字牌

        返回：(是否满足, 最小赖子需求)
        """
        laizi_tile = self.game_context.lazy_tile
        # 1. 检查花色一致性（赖子只能补缺，不能替换）
        tile_suit = all_tiles[0] // 9 if all_tiles[0] != laizi_tile else None
        if tile_suit is None:
            for tile in all_tiles:
                if tile != laizi_tile:
                    tile_suit = tile // 9
                    break
            else:
                # 如果 for 循环结束，说明手牌有问题
                raise ValueError("Invalid hand tiles: all tiles are lazy tiles")

        if tile_suit >= 3:
            return False, -1
        if any(tile // 9 != tile_suit for tile in all_tiles if tile != laizi_tile):
            return False, -1

        laizi_count = hand.count(laizi_tile)
        laizi_suit = laizi_tile // 9
        if tile_suit != laizi_suit:
            if laizi_count >= 3:
                return False, -1
            else:
                success, _ = self.checker.get_min_laizi_needed(hand, wild_index=laizi_tile, unlimited_eye=True)
                if not success:
                    return False, -1
                else:
                    return True, laizi_count
        else:
            success, min_laizi = self.checker.get_min_laizi_needed(hand, wild_index=laizi_tile, unlimited_eye=True)
            if not success or min_laizi > 2:
                return False, -1
            else:
                return True, min_laizi

    def _check_peng_peng_hu(self, hand_tiles: List[int], melds: List[Meld]) -> Tuple[bool, int]:
        """
        检测碰碰胡
        条件：所有面子都是刻子或杠，且满足标准胡牌结构
        返回：(是否满足, 最小赖子需求)
        """
        # 检查鸣牌中是否有吃

        if any(meld.is_chow for meld in melds):
            return False, -1

        # 统计手牌
        hand_counter = Counter(hand_tiles)
        laizi_tile = self.game_context.lazy_tile
        laizi_count = hand_counter.get(laizi_tile, 0)

        # 尝试使用0-min(laizi_count, 2)个赖子检测是否能胡牌
        for temp_laizi_used in range(min(laizi_count, 2) + 1):
            # 尝试用temp_laizi_used个赖子检测是否能胡牌
            temp_hand_counter = hand_counter.copy()
            # 从手牌中移除temp_laizi_used个赖子
            temp_hand_counter[laizi_tile] -= temp_laizi_used
            # 遍历所有将牌
            for tile in temp_hand_counter.keys():
                # 检查需要多少个赖子才能组成将牌
                need_for_pair = max(0, 2 - temp_hand_counter[tile])
                # 如果需要的赖子数大于当前可用的赖子数，直接跳过
                if need_for_pair > temp_laizi_used:
                    continue

                remaining_laizi = temp_laizi_used - need_for_pair

                remaining_hand = temp_hand_counter.copy()
                # 从手牌中移除需要的牌
                remaining_hand[tile] = max(0, remaining_hand.get(tile, 0) - 2)

                # 检查剩余牌是否能全部组成刻子
                total_need = 0
                for t, cnt in remaining_hand.items():
                    if cnt % 3 != 0:
                        total_need += (3 - cnt % 3)
                        if total_need > remaining_laizi:
                            break

                if total_need == remaining_laizi:
                    return True, temp_laizi_used
        else:
            return False, -1


    def _check_fully_melded(self, hand_tiles: List[int], melds: List[Meld]) -> Tuple[bool, int]:
        """
        检测全求人
        条件：
        1. 4面子均鸣牌得来
        2. 手牌仅剩2张
        3. 只能接炮胡不能自摸
        4. 满足标准胡牌结构（将牌必须是2、5、8）
        返回：(是否满足, 最小赖子需求)
        """
        # 1. 必须有4组鸣牌
        if len(melds) != 4:
            return False, -1

        # 2. 所有鸣牌都必须是开口的（不能有暗杠）
        if any(not meld.is_opened for meld in melds):
            return False, -1

        # 3. 手牌应该只有2张
        if len(hand_tiles) != 2:
            return False, -1

        # 4. 胡牌方式必须是接炮胡（点炮）
        if self.game_context.win_way != WinWay.DISCARD.value:
            return False, -1

        # 5. 统计赖子数量
        laizi_tile = self.game_context.lazy_tile
        laizi_count = hand_tiles.count(laizi_tile)

        if laizi_count == 0:
            if hand_tiles[0] == hand_tiles[1] and hand_tiles[0] in self.ALL_JIANG:
                return True, 0
            else:
                return False, -1
        elif laizi_count == 1:
            if ((hand_tiles[0] != laizi_tile and hand_tiles[0] in self.ALL_JIANG)
                    or (hand_tiles[1] != laizi_tile and hand_tiles[1] in self.ALL_JIANG)):
                return True, 1
            else:
                return False, -1
        else:
            if laizi_tile in self.ALL_JIANG:
                return True, 0
            else:
                return False, -1

    def _check_flower_on_kong(self, hand_tiles: List[int]) -> Tuple[bool, int]:
        """检测杠上开花"""
        if not self.game_context.win_way == WinWay.KONG_SELF_DRAW.value:
            return False, -1

        # 杠上开花需要标准胡牌结构，将牌必须是2、5、8
        success, min_laizi = self.checker.get_min_laizi_needed(hand_tiles, wild_index=self.game_context.lazy_tile, unlimited_eye=False)

        if not success:
            return False, -1

        if min_laizi > 2:
            return False, -1

        return True, min_laizi

    def _check_last_tile_win(self, hand_tiles: List[int]) -> Tuple[bool, int]:
        """检测海底捞月"""
        if len(self.game_context.wall) > 3:
            return False, -1

        # 海底捞月需要标准胡牌结构，将牌必须是2、5、8
        success, min_laizi = self.checker.get_min_laizi_needed(hand_tiles, wild_index=self.game_context.lazy_tile, unlimited_eye=False)

        if not success:
            return False, -1

        if min_laizi > 2:
            return False, -1

        return True, min_laizi

    def _check_rob_kong(self, hand_tiles: List[int]) -> Tuple[bool, int]:
        """检测抢杠胡"""
        if not self.game_context.win_way == WinWay.ROB_KONG.value:
            return False, -1

        # 抢杠胡需要标准胡牌结构，将牌必须是2、5、8
        success, min_laizi = self.checker.get_min_laizi_needed(hand_tiles, wild_index=self.game_context.lazy_tile, unlimited_eye=False)

        if not success:
            return False, -1

        if min_laizi > 2:
            return False, -1

        return True, min_laizi

#
# if __name__ == "__main__":
#     print("WuhanMahjongEngine Test")
    # print("清一色 测试")
    # hand = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 29, 29]
    # melds = []
    # all_tiles = hand.copy()
    # game_context = GameContext(
    #     lazy_tile=29,
    # )
    # engine = WuhanMahjongWinChecker(game_context=game_context)
    # print(engine._check_pure_flush(all_tiles, hand))

    # hand = [9, 9, 10, 10, 11, 11, 12, 12]
    # melds = [
    #     Meld(action_type=ActionType.PONG, tiles=[13, 13, 13]),  # 条子刻子
    #     Meld(action_type=ActionType.CHOW_LEFT, tiles=[14, 15, 16])  # 条子顺子
    # ]
    # all_tiles = hand.copy()
    # for meld in melds:
    #     all_tiles.extend(meld.tiles)
    # game_context = GameContext(
    #     lazy_tile=29,
    # )
    # engine = WuhanMahjongWinChecker(game_context=game_context)
    # print(engine._check_pure_flush(all_tiles, hand))

    # hand =  [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    # melds = []
    # all_tiles = hand.copy()
    # game_context = GameContext(
    #     lazy_tile=4,
    # )
    # engine = WuhanMahjongWinChecker(game_context=game_context)
    # print(engine._check_pure_flush(all_tiles, hand))

    # hand =  [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 9, 9, 9, 9]
    # melds = []
    # all_tiles = hand.copy()
    # game_context = GameContext(
    #     lazy_tile=9,
    # )
    # engine = WuhanMahjongWinChecker(game_context=game_context)
    # print(engine._check_pure_flush(all_tiles, hand))
    #
    #
    # hand =  [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 9, 9, 9, 9]
    # melds = []
    # game_context = GameContext(
    #     lazy_tile=9,
    # )
    # engine = WuhanMahjongWinChecker(game_context=game_context)
    # print(engine._check_peng_peng_hu(hand, melds))
    #
    #
    #
    # # 测试用例数据
    # test_cases = [
    #     # A组：正向用例
    #     {
    #         "desc": "Standard - 基础碰碰胡（无副露）",
    #         "hand": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5],  # 4个刻子(1-4) + 1对(5)
    #         "melds": [],
    #         "game_context": {"lazy_tile": 33},
    #         "expected_result": True
    #     },
    #     {
    #         "desc": "With Kongs - 带明杠碰碰胡",
    #         "hand": [2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6],  # 4个刻子(2-5) + 1对(6)
    #         "melds": [Meld(action_type=ActionType.KONG_EXPOSED, tiles=[1, 1, 1, 1])],  # 明杠(1)
    #         "game_context": {"lazy_tile": 33},
    #         "expected_result": True
    #     },
    #     {
    #         "desc": "All Honors - 全字牌碰碰胡",
    #         "hand": [27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31],  # 27=东,28=南,29=西,30=北,31=中
    #         "melds": [],
    #         "game_context": {"lazy_tile": 33},
    #         "expected_result": True
    #     },
    #     {
    #         "desc": "Lazy Tile - 赖子自身还原组成将牌",
    #         "hand": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 33, 33],  # 33=白板(赖子)
    #         "melds": [],
    #         "game_context": {"lazy_tile": 33},
    #         "expected_result": True
    #     },
    #
    #     # B组：反向用例
    #     {
    #         "desc": "Has Chow - 包含顺子（非碰碰胡）",
    #         "hand": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 7, 7, 8, 9],  # 顺子[7,8,9] + 3个刻子
    #         "melds": [],
    #         "game_context": {"lazy_tile": 33},
    #         "expected_result": False
    #     },
    #     {
    #         "desc": "Mixed - 3刻子+1顺子+1将",
    #         "hand": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6, 7, 7],  # 顺子[4,5,6] + 3刻子
    #         "melds": [],
    #         "game_context": {"lazy_tile": 33},
    #         "expected_result": False
    #     },
    #     {
    #         "desc": "Seven Pairs - 七对子（非碰碰胡）",
    #         "hand": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
    #         "melds": [],
    #         "game_context": {"lazy_tile": 33},
    #         "expected_result": False
    #     },
    #     {
    #         "desc": "Seven Pairs - 七对子（非碰碰胡）",
    #         "hand": [3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
    #         "melds": [Meld(action_type=ActionType.CHOW_LEFT, tiles=[4, 5, 6])],
    #         "game_context": {"lazy_tile": 7},
    #         "expected_result": False
    #     },
    #
    #     # C组：异常用例
    #     {
    #         "desc": "Incomplete - 牌数不足（13张）",
    #         "hand": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5],
    #         "melds": [],
    #         "game_context": {"lazy_tile": 33},
    #         "expected_result": False
    #     },
    #     {
    #         "desc": "No Pair - 无将牌（孤张）",
    #         "hand": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6],
    #         "melds": [],
    #         "game_context": {"lazy_tile": 33},
    #         "expected_result": False
    #     },
    #     {
    #         "desc": "Lazy Limit - 赖子使用量超限（需3个但最多2个）",
    #         "hand": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 33, 33],  # 需3个赖子补刻子，但只有2个
    #         "melds": [],
    #         "game_context": {"lazy_tile": 33},
    #         "expected_result": False
    #     }
    # ]
    # for test_case in test_cases:
    #     game_context = GameContext(
    #         lazy_tile=test_case["game_context"]["lazy_tile"],
    #     )
    #     engine = WuhanMahjongWinChecker(game_context=game_context)
    #     result = engine._check_peng_peng_hu(test_case["hand"], test_case["melds"])
    #     print(f"{test_case['desc']}: {result} (Expected: {test_case['expected_result']})")
#
# meld_actions = [
#     ActionType.KONG_EXPOSED,
#     ActionType.KONG_SUPPLEMENT,
#     ActionType.KONG_CONCEALED,
#     ActionType.CHOW_LEFT,
#     ActionType.CHOW_MIDDLE,
#     ActionType.CHOW_RIGHT,
#     ActionType.PONG,
# ]
#
# total_card = [i for i in range(0, 34)] * 4
# total_card_set = set(range(0, 34))

# for i in range(0, 10000000):
#     context = GameContext()
#     context.lazy_tile, context.skin_tile[0], context.skin_tile[1] = Tiles.generate_special_tiles(random.randint(0, 33))
#     card_remain = total_card.copy()
#     for _ in range(4):
#         card_remain.remove(context.lazy_tile)
#         card_remain.remove(context.skin_tile[0])
#         card_remain.remove(context.skin_tile[1])
#         card_remain.remove(context.red_dragon)
#     card_remain_count = Counter(card_remain)
#     hand_num = random.sample([2, 5, 8, 11, 14], 1)[0]
#     hand_tile = random.sample(total_card, hand_num)
#     for tile in hand_tile:
#         if tile not in {context.lazy_tile, context.skin_tile[0], context.skin_tile[1], context.red_dragon}:
#             card_remain.remove(tile)
#             card_remain_count[tile] -= 1
#     hand_tile.sort()
#     meld_num = (14 - hand_num) // 3
#     melds_tile = []
#     for j in range(meld_num):
#         _meld_tile = []
#         meld_action = random.sample(meld_actions, 1)[0]
#         if meld_action.is_kong_action:
#             can_form_kong_list = [k for k, v in card_remain_count.items() if v > 3]
#             kong_tiles = random.sample(can_form_kong_list, 1)[0]
#             _meld_tile.extend([kong_tiles] * 4)
#             card_remain_count[kong_tiles] -= 4
#             for _ in range(4):
#                 card_remain.remove(kong_tiles)
#         elif meld_action.is_chow_action:
#             # 遍历 hand_tile 和 meld_tile 找所有到剩余至少 1 张的牌，如果存在顺子，就随机选一个顺子
#             chow_tiles_list = []
#             for k, v in card_remain_count.items():
#                 if v >= 1 and k % 9 < 7 and k // 9 < 3:
#                     if card_remain_count[k + 1] >= 1 and card_remain_count[k + 2] >= 1:
#                         chow_tiles_list.append([k, k + 1, k + 2])
#             choiced_chow_tiles = random.sample(chow_tiles_list, 1)[0]
#             _meld_tile.extend(choiced_chow_tiles)
#             for tile in choiced_chow_tiles:
#                 card_remain_count[tile] -= 1
#                 card_remain.remove(tile)
#         elif meld_action.is_pon_action:
#             can_form_pong_list = [k for k, v in card_remain_count.items() if v >= 3]
#             pong_tiles = random.sample(can_form_pong_list, 1)[0]
#             _meld_tile.extend([pong_tiles] * 3)
#             card_remain_count[pong_tiles] -= 3
#             for _ in range(3):
#                 card_remain.remove(pong_tiles)
#         melds_tile.append(Meld(action_type=meld_action, tiles=_meld_tile))
#
#     player_data = PlayerData(
#         player_id=1,
#         hand_tiles=hand_tile,
#         melds=melds_tile,
#     )
#
#     context.wall = deque(random.sample(card_remain, random.randint(0, len(card_remain) - 13 * 3)))
#
#     engine_1 = WuhanMahjongWinChecker(game_context=context)
#     result1 = engine_1.check_win(player_data)
#     if result1.can_win and len(player_data.hand_tiles) not in [2, 5] and WinType.SOFT_WIN not in result1.win_type:
#         print(result1)
#
#         print(len(context.wall))
#         print(context.lazy_tile, context.skin_tile[0], context.skin_tile[1])
#         print(hand_tile)
#         print(melds_tile)
#         print(player_data.has_opened)


# player_data = PlayerData(
#         player_id=1,
#         hand_tiles=[1, 1, 1, 2, 2, 2, 4, 4],
#         melds=[
#             Meld(action_type=ActionType.PONG, tiles=[5, 5, 5]),
#             Meld(action_type=ActionType.KONG_EXPOSED, tiles=[12, 12, 12, 12]),
#         ],
# )
# context = GameContext(
#     lazy_tile=28,
#     skin_tile=[27, 33],
#     wall=deque([0, 19, 2])
# )
# engine_1 = WuhanMahjongWinChecker(game_context=context)
# result1 = engine_1.check_win(player_data)
# print(result1)
#
# player_data2 = PlayerData(
#         player_id=1,
#         hand_tiles=[0, 20],
#         melds=[
#             Meld(action_type=ActionType.CHOW_LEFT, tiles=[21, 22, 23]),
#             Meld(action_type=ActionType.KONG_CONCEALED, tiles=[28, 28, 28, 28]),
#             Meld(action_type=ActionType.KONG_EXPOSED, tiles=[19, 19, 19, 19]),
#             Meld(action_type=ActionType.CHOW_MIDDLE, tiles=[12, 13, 14]),
#         ],
# )
# context = GameContext(
#     lazy_tile=11,
#     skin_tile=[10, 9],
#     wall=deque(random.sample(total_card, 43))
# )
# engine_2 = WuhanMahjongWinChecker(game_context=context)
# result2 = engine_2.check_win(player_data2)
# print(result2)
#
#
# player_data3 = PlayerData(
#         player_id=1,
#         hand_tiles=[14, 15, 16, 21, 21],
#         melds=[
#             Meld(action_type=ActionType.CHOW_RIGHT, tiles=[22, 23, 24]),
#             Meld(action_type=ActionType.PONG, tiles=[29, 29, 29]),
#             Meld(action_type=ActionType.KONG_CONCEALED, tiles=[19, 19, 19, 19])
#         ],
# )
# context3 = GameContext(
#     lazy_tile=28,
#     skin_tile=[27, 33],
#     wall=deque(random.sample(total_card, 7))
# )
# engine_3 = WuhanMahjongWinChecker(game_context=context3)
# result3 = engine_3.check_win(player_data3)
# print(result3)

#
# meld_actions = [
#     ActionType.KONG_EXPOSED,
#     ActionType.KONG_SUPPLEMENT,
#     ActionType.KONG_CONCEALED,
#     ActionType.CHOW_LEFT,
#     ActionType.CHOW_MIDDLE,
#     ActionType.CHOW_RIGHT,
#     ActionType.PONG,
# ]
#
# total_card = [i for i in range(0, 34)] * 4
# total_card_set = set(range(0, 34))
#
# def process_batch(batch_size):
#     """处理一批游戏模拟"""
#     results = []
#     # 每个进程使用独立随机种子
#     random.seed(os.getpid() + int(str(os.getpid())[::-1]))
#
#     for _ in range(batch_size):
#         context = GameContext()
#         context.lazy_tile, context.skin_tile[0], context.skin_tile[1] = Tiles.generate_special_tiles(
#             random.randint(0, 33))
#         card_remain = total_card.copy()
#
#         # 移除特殊牌
#         for _ in range(4):
#             card_remain.remove(context.lazy_tile)
#             card_remain.remove(context.skin_tile[0])
#             card_remain.remove(context.skin_tile[1])
#             card_remain.remove(context.red_dragon)
#
#         card_remain_count = Counter(card_remain)
#         hand_num = random.sample([2, 5, 8, 11, 14], 1)[0]
#         hand_tile = random.sample(total_card, hand_num)
#
#         # 更新手牌和剩余牌
#         for tile in hand_tile:
#             if tile not in {context.lazy_tile, context.skin_tile[0], context.skin_tile[1], context.red_dragon}:
#                 card_remain.remove(tile)
#                 card_remain_count[tile] -= 1
#
#         hand_tile.sort()
#         meld_num = (14 - hand_num) // 3
#         melds_tile = []
#
#         # 生成牌组
#         for _ in range(meld_num):
#             _meld_tile = []
#             # meld_action = random.sample(meld_actions, 1)[0]
#             meld_action = random.choices(meld_actions, weights=[16, 11, 14, 5, 4, 6, 44])[0]
#
#             if meld_action.is_kong_action:
#                 # 处理杠牌逻辑
#                 can_form_kong_list = [k for k, v in card_remain_count.items() if v > 3]
#                 kong_tiles = random.sample(can_form_kong_list, 1)[0]
#                 _meld_tile.extend([kong_tiles] * 4)
#                 card_remain_count[kong_tiles] -= 4
#                 for _ in range(4):
#                     card_remain.remove(kong_tiles)
#
#             elif meld_action.is_chow_action:
#                 # 处理吃牌逻辑
#                 chow_tiles_list = []
#                 for k, v in card_remain_count.items():
#                     if v >= 1 and k % 9 < 7 and k // 9 < 3:
#                         if card_remain_count[k + 1] >= 1 and card_remain_count[k + 2] >= 1:
#                             chow_tiles_list.append([k, k + 1, k + 2])
#                 choiced_chow_tiles = random.sample(chow_tiles_list, 1)[0]
#                 _meld_tile.extend(choiced_chow_tiles)
#                 for tile in choiced_chow_tiles:
#                     card_remain_count[tile] -= 1
#                     card_remain.remove(tile)
#
#             elif meld_action.is_pon_action:
#                 # 处理碰牌逻辑
#                 can_form_pong_list = [k for k, v in card_remain_count.items() if v >= 3]
#                 pong_tiles = random.sample(can_form_pong_list, 1)[0]
#                 _meld_tile.extend([pong_tiles] * 3)
#                 card_remain_count[pong_tiles] -= 3
#                 for _ in range(3):
#                     card_remain.remove(pong_tiles)
#
#             melds_tile.append(Meld(action_type=meld_action, tiles=_meld_tile))
#
#         # 创建玩家数据
#         player_data = PlayerData(
#             player_id=1,
#             hand_tiles=hand_tile,
#             melds=melds_tile,
#         )
#
#         context.wall = deque(random.sample(card_remain, random.randint(0, len(card_remain) - 13 * 3)))
#         engine_1 = WuhanMahjongWinChecker(game_context=context)
#         result1 = engine_1.check_win(player_data)
#
#         # 收集有效结果
#         if result1.can_win and len(player_data.hand_tiles) not in [2, 5] and WinType.SOFT_WIN not in result1.win_type:
#             results.append({
#                 "result": result1,
#                 "wall_len": len(context.wall),
#                 "special_tiles": (context.lazy_tile, context.skin_tile[0], context.skin_tile[1]),
#                 "hand_tiles": hand_tile,
#                 "melds": melds_tile,
#                 "has_opened": player_data.has_opened
#             })
#
#     return results
#
#
# if __name__ == '__main__':
#     # 配置并行参数
#     total_iterations = 1_000_000
#     num_processes = cpu_count()  # 使用所有可用CPU核心
#     batch_size = total_iterations // num_processes
#
#     print(f"Starting parallel execution with {num_processes} processes")
#     print(f"Total iterations: {total_iterations:,}")
#     print(f"Batch size per process: {batch_size:,}")
#
#     # 创建进程池
#     with Pool(processes=num_processes) as pool:
#         # 分配任务
#         tasks = [batch_size] * num_processes
#         # 处理剩余迭代次数
#         if total_iterations % num_processes != 0:
#             tasks[-1] += total_iterations % num_processes
#
#         # 执行并行计算
#         batch_results = pool.map(process_batch, tasks)
#
#     # 合并结果
#     all_results = []
#     for result_batch in batch_results:
#         all_results.extend(result_batch)
#
#     # 输出结果
#     for res in all_results:
#         print(res["result"])
#         print(f"Wall length: {res['wall_len']}")
#         print(f"Special tiles: lazy tile={res['special_tiles'][0]}, skin tile=[{res['special_tiles'][1]}, {res['special_tiles'][2]}]")
#         print(f"Hand tiles: {res['hand_tiles']}")
#         print(f"Melds: {res['melds']}")
#         print(f"Has opened: {res['has_opened']}")
#         print("-" * 50)
#
#     print(f"Total valid results found: {len(all_results)}")