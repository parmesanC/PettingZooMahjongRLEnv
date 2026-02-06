from typing import List, Optional
from functools import reduce

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData, Meld
from src.mahjong_rl.core.constants import ActionType, WinType, WinWay
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.win_detector import WinCheckResult


class MahjongScoreSettler:
    """武汉麻将分数结算器"""

    # 大胡类型（除SOFT_WIN和HARD_WIN外的所有WinType）
    BIG_HU_TYPES = {
        WinType.PURE_FLUSH, WinType.PENG_PENG_HU, WinType.ALL_WIND,
        WinType.ALL_JIANG, WinType.FULLY_MELDED, WinType.FLOWER_ON_KONG,
        WinType.LAST_TILE_WIN, WinType.ROB_KONG
    }

    def __init__(self, is_kou_kou_fan: bool = True):
        """
        Args:
            is_kou_kou_fan: 是否口口翻模式（影响金顶分数）
        """
        self.is_kou_kou_fan = is_kou_kou_fan

    def settle(self, win_result: WinCheckResult, ctx: GameContext) -> List[float]:
        """
        核心结算方法

        Returns:
            List[float]: 长度为4的分数变化列表，正数表示赢，负数表示输
        """
        # 1. 验证胡牌结果
        if not win_result or not win_result.can_win:
            return [0.0, 0.0, 0.0, 0.0]

        # 2. 确定赢家
        winner_idx = self._find_winner(ctx)
        if winner_idx is None:
            return [0.0, 0.0, 0.0, 0.0]

        winner = ctx.players[winner_idx]
        loser_indices = [i for i in range(4) if i != winner_idx]

        # 3. 判断大胡/小胡
        is_big_hu = self._is_big_hu(win_result)
        big_hu_cnt = self._count_big_hu(win_result)

        # 4. 计算基础分
        base_score = 10 * big_hu_cnt if is_big_hu else 1

        # 5. 计算赢家番数乘积
        winner_fan_product = reduce(lambda x, y: x * y, self._get_winner_fan_components(winner, win_result, ctx), 1)

        # 6. 计算每个输家的分数
        loser_scores = []  # 输家应支付的分数（未封顶）
        for loser_idx in loser_indices:
            loser = ctx.players[loser_idx]

            # 计算输家番数乘积
            loser_fan_components = self._get_loser_fan_components(loser, win_result, ctx, loser_idx)
            loser_fan_product = reduce(lambda x, y: x * y, loser_fan_components, 1)

            # 计算庄闲系数
            zhuang_xian_factor = self._get_zhuang_xian_factor(
                winner.is_dealer, is_big_hu, loser_idx, ctx.dealer_idx
            )

            # 计算原始分数
            raw_score = base_score * winner_fan_product * loser_fan_product * zhuang_xian_factor

            # 处理放冲因子（已在loser_fan_components中处理）

            loser_scores.append(raw_score)

        # 7. 应用封顶和金顶规则
        capped_scores = self._apply_capping(loser_scores)

        # 8. 检查承包规则
        contractor_idx = self._check_contractor(ctx, win_result, winner_idx)
        if contractor_idx is not None and contractor_idx != winner_idx:
            # 承包者支付所有分数
            final_scores = [0.0, 0.0, 0.0, 0.0]
            total_lost = sum(capped_scores)
            final_scores[contractor_idx] = -total_lost
            final_scores[winner_idx] = total_lost
        else:
            # 正常结算
            final_scores = [0.0, 0.0, 0.0, 0.0]
            for idx, score in zip(loser_indices, capped_scores):
                final_scores[idx] = -score
            final_scores[winner_idx] = sum(capped_scores)

        return final_scores

    def _find_winner(self, ctx: GameContext) -> Optional[int]:
        """找到赢家索引"""
        for i, player in enumerate(ctx.players):
            if player.is_win:
                return i
        return None

    def _is_big_hu(self, win_result: WinCheckResult) -> bool:
        """判断是否为大胡"""
        return any(wt in self.BIG_HU_TYPES for wt in win_result.win_type)

    def _count_big_hu(self, win_result: WinCheckResult) -> int:
        """计算大胡个数"""
        return sum(1 for wt in win_result.win_type if wt in self.BIG_HU_TYPES)

    def _get_winner_fan_components(self, winner: PlayerData,
                                   win_result: WinCheckResult,
                                   ctx: GameContext) -> List[float]:
        """计算赢家的番数乘积因子"""
        # 基础因子（开口、杠等）
        fan_components = self._get_base_fan_components(winner)

        # 硬胡因子（硬胡乘2，软胡乘1）
        if WinType.HARD_WIN in win_result.win_type:
            fan_components.append(2.0)
        # 软胡因子（隐含乘1，无需添加）

        # 自摸因子
        if ctx.win_way == WinWay.SELF_DRAW.value:
            if self._is_big_hu(win_result):
                fan_components.append(1.5)  # 大胡自摸乘1.5
            else:
                fan_components.append(2.0)  # 小胡自摸乘2

        return fan_components

    def _get_loser_fan_components(self, loser: PlayerData,
                                  win_result: WinCheckResult,
                                  ctx: GameContext,
                                  loser_idx: int) -> List[float]:
        """计算输家的番数乘积因子"""
        fan_components = self._get_base_fan_components(loser)

        # 放冲因子
        if ctx.win_way == WinWay.DISCARD.value and ctx.discard_player == loser_idx:
            if self._is_big_hu(win_result):
                fan_components.append(1.5)  # 放冲大胡乘1.5
            else:
                fan_components.append(2.0)  # 放冲小胡乘2

        return fan_components

    def _get_base_fan_components(self, player: PlayerData) -> List[float]:
        """计算基础番数因子（开口、明杠、暗杠、特殊杠）"""
        fan_components = []

        # 统计开口次数（吃、碰、明杠）
        kou_count = sum(1 for meld in player.melds if self._is_kou(meld))
        fan_components.append(2.0 ** kou_count)  # 2的k次方

        # 明杠个数（直杠+补杠）
        ming_gang = sum(1 for meld in player.melds
                        if meld.action_type.action_type in [ActionType.KONG_EXPOSED, ActionType.KONG_SUPPLEMENT])
        fan_components.append(2.0 ** ming_gang)

        # 暗杠个数（每个乘4）
        an_gang = sum(1 for meld in player.melds
                      if meld.action_type.action_type == ActionType.KONG_CONCEALED)
        for _ in range(an_gang):
            fan_components.append(4.0)

        # 特殊杠（皮子、赖子、红中）
        pi_zi, lai_zi, hong_zhong = player.special_gangs

        # 皮子杠：每个乘2
        for _ in range(pi_zi):
            fan_components.append(2.0)

        # 赖子杠：每个乘4
        for _ in range(lai_zi):
            fan_components.append(4.0)

        # 红中杠：每个乘2
        for _ in range(hong_zhong):
            fan_components.append(2.0)

        return fan_components

    def check_min_fan_requirement(self, player_id: int, win_types: List[WinType], ctx: GameContext) -> bool:
        """
        检查起胡番要求

        武汉麻将规则：起胡番需要达到16番
        计算方式：每个玩家的分数 + 赢家分数 >= 16

        Args:
            player_id: 玩家ID
            win_types: 胡牌类型列表
            ctx: 游戏上下文

        Returns:
            True 如果满足起胡番要求
        """
        # 获取玩家
        player = ctx.players[player_id]

        # 创建临时 WinCheckResult 用于计算分数
        temp_result = WinCheckResult(
            can_win=True,
            win_type=win_types,
            min_wild_need=0
        )

        # 保存原始状态
        original_is_win = player.is_win
        original_winner_ids = ctx.winner_ids.copy() if ctx.winner_ids else []
        original_win_way = ctx.win_way
        original_discard_player = ctx.discard_player

        # 临时设置为赢家（使用自摸方式计算，避免承包等复杂逻辑）
        player.is_win = True
        ctx.winner_ids = [player_id]
        ctx.win_way = WinWay.SELF_DRAW.value
        ctx.discard_player = None

        try:
            # 计算分数
            score_list = self.settle(temp_result, ctx)
            winner_score = max(score_list)

            # 检查是否达到起胡番（16番）
            # 每个玩家的分数绝对值 + 赢家分数 >= 16
            if all(abs(score) + winner_score >= 16 for score in score_list):
                return True

            return False
        finally:
            # 恢复原始状态
            player.is_win = original_is_win
            ctx.winner_ids = original_winner_ids
            ctx.win_way = original_win_way
            ctx.discard_player = original_discard_player

    def _is_kou(self, meld: Meld) -> bool:
        """判断是否为开口（吃、碰、明杠）"""
        return meld.is_opened and meld.action_type.action_type in [
            ActionType.CHOW, ActionType.PONG, ActionType.KONG_EXPOSED
        ]

    def _get_zhuang_xian_factor(self, winner_is_dealer: bool, is_big_hu: bool,
                                loser_idx: int, dealer_idx: int) -> int:
        """计算庄闲系数"""
        if is_big_hu:
            return 1  # 大胡无庄闲系数

        # 小胡庄闲系数
        if winner_is_dealer:
            # 赢家是庄家，所有闲家输点×2
            return 2 if loser_idx != dealer_idx else 1
        else:
            # 赢家是闲家，庄家输点×2，其他闲家×1
            return 2 if loser_idx == dealer_idx else 1

    def _apply_capping(self, loser_scores: List[float]) -> List[float]:
        """应用封顶和金顶规则

        规则：
        1. 普通封顶：单家最高300分
        2. 金顶：三家原始分数都≥300时，上限提升到400分（口口翻500分）

        注意：这是上限值，不是固定值。每家分数分别应用上限。
        """
        # 确定上限：检查是否触发金顶（三家原始分数都≥300）
        if all(score >= 300.0 for score in loser_scores):
            # 启用金顶，上限变为400（口口翻模式500）
            cap = 500.0 if self.is_kou_kou_fan else 400.0
        else:
            # 普通封顶，上限300
            cap = 300.0

        # 对每家分数分别应用上限
        return [min(score, cap) for score in loser_scores]

    def _check_contractor(self, ctx: GameContext,
                          win_result: WinCheckResult,
                          winner_idx: int) -> Optional[int]:
        """
        检查承包者
        Returns:
            Optional[int]: 承包者索引，若无承包则返回None
        """
        # 条件1：抢杠胡承包
        if WinType.ROB_KONG in win_result.win_type and ctx.discard_player is not None:
            return ctx.discard_player

        # 条件2：全求人承包
        # A放冲给B作全求人，且A没听牌
        if (WinType.FULLY_MELDED in win_result.win_type and
                ctx.win_way == WinWay.DISCARD.value and
                ctx.discard_player is not None):

            discarder = ctx.players[ctx.discard_player]
            if not discarder.is_ting:
                return ctx.discard_player

        # 条件3：清一色承包
        # B胡清一色，且B的第三次开口对象是A
        if WinType.PURE_FLUSH in win_result.win_type:
            winner = ctx.players[winner_idx]
            # 统计开口次数
            kou_melds = [meld for meld in winner.melds if self._is_kou(meld)]
            if len(kou_melds) >= 3:
                third_kou_from = kou_melds[2].from_player
                if third_kou_from is not None and third_kou_from != winner_idx:
                    return third_kou_from

        return None

    def settle_flow_draw(self, ctx: GameContext) -> List[float]:
        """
        流局（荒庄）查大叫结算

        规则：
        1. 检查所有未胡牌者是否听牌
        2. 已听牌者可向未听牌者收取基础番钱
        3. 计翻项仅计：底翻1 + 开口翻 + 皮/中翻 + 明/暗杠翻
        4. 未听牌者若未开口，支付额 ×2

        Returns:
            List[float]: 长度为4的分数变化列表，正数表示赢，负数表示输
        """
        final_scores = [0.0, 0.0, 0.0, 0.0]

        # 区分听牌者和未听牌者
        ting_players = []
        not_ting_players = []

        for idx, player in enumerate(ctx.players):
            if not player.is_win:
                if player.is_ting:
                    ting_players.append(idx)
                else:
                    not_ting_players.append(idx)

        # 如果没有听牌者或没有未听牌者，则无需结算
        if not ting_players or not not_ting_players:
            return final_scores

        # 计算每个听牌者的基础番数
        ting_players_fan = {}
        for ting_idx in ting_players:
            fan_score = self._get_base_fan_score(ctx.players[ting_idx])
            ting_players_fan[ting_idx] = fan_score

        # 计算每个未听牌者的基础番数
        not_ting_players_fan = {}
        for not_ting_idx in not_ting_players:
            fan_score = self._get_base_fan_score(ctx.players[not_ting_idx])
            not_ting_players_fan[not_ting_idx] = fan_score

        # 结算：每个听牌者向每个未听牌者收取番钱
        for ting_idx in ting_players:
            ting_fan = ting_players_fan[ting_idx]
            for not_ting_idx in not_ting_players:
                not_ting_fan = not_ting_players_fan[not_ting_idx]
                not_ting_player = ctx.players[not_ting_idx]

                # 计算支付额：听牌者番数 × 未听牌者番数
                payment = ting_fan * not_ting_fan

                # 未听牌者若未开口，支付额 ×2
                if not not_ting_player.has_opened:
                    payment *= 2

                # 更新分数
                final_scores[not_ting_idx] -= payment  # 未听牌者支付
                final_scores[ting_idx] += payment  # 听牌者收取

        return final_scores

    def _get_base_fan_score(self, player: PlayerData) -> float:
        """
        计算玩家基础番数（不包括硬胡、自摸、放冲因子）

        计翻项仅计：底翻1 + 开口翻 + 皮/中翻 + 明/暗杠翻

        Returns:
            float: 基础番数
        """
        fan_components = []

        # 底翻1（基础1倍）
        fan_components.append(1.0)

        # 开口次数：每开一次口（吃、碰、明杠）乘2
        kou_count = sum(1 for meld in player.melds if self._is_kou(meld))
        fan_components.append(2.0 ** kou_count)

        # 明杠个数：每有一个明杠（冲杠、补杠）乘2
        ming_gang = sum(1 for meld in player.melds
                        if meld.action_type.action_type in [ActionType.KONG_EXPOSED, ActionType.KONG_SUPPLEMENT])
        fan_components.append(2.0 ** ming_gang)

        # 暗杠个数：每有一个暗杠乘4
        an_gang = sum(1 for meld in player.melds
                      if meld.action_type.action_type == ActionType.KONG_CONCEALED)
        for _ in range(an_gang):
            fan_components.append(4.0)

        # 皮子杠：每个乘2
        pi_zi, lai_zi, hong_zhong = player.special_gangs
        for _ in range(pi_zi):
            fan_components.append(2.0)

        # 赖子杠：每个乘4
        for _ in range(lai_zi):
            fan_components.append(4.0)

        # 红中杠：每个乘2
        for _ in range(hong_zhong):
            fan_components.append(2.0)

        # 计算番数乘积
        return reduce(lambda x, y: x * y, fan_components, 1.0)


if __name__ == "__main__":
    # 创建结算器（口口翻模式）
    settler = MahjongScoreSettler(is_kou_kou_fan=True)

    # 准备游戏上下文
    players = [PlayerData(player_id=i) for i in range(4)]
    players[0].is_dealer = True  # 设玩家0为庄家
    players[1].is_win = True     # 玩家1胡牌
    # players[1].melds = [...]     # 设置牌组（需要根据实际情况添加Meld对象）
    players[1].special_gangs = [0, 1, 0]  # 1个赖子杠

    ctx = GameContext(
        players=players,
        dealer_idx=0,
        win_way=WinWay.SELF_DRAW.value  # 自摸
    )

    # 准备胡牌结果
    win_result = WinCheckResult(
        can_win=True,
        win_type=[WinType.HARD_WIN, WinType.PURE_FLUSH],  # 硬胡+清一色
        min_wild_need=0
    )

    # 执行结算
    scores = settler.settle(win_result, ctx)
    print(f"最终得分: {scores}")  # 例如: [0, +500, -200, -150, -150]