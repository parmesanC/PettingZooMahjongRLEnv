class RoundInfo:
    """
    武汉麻将庄家管理类

    设计原则：
    - 单一职责：仅管理庄家轮换逻辑
    - 接口隔离：只暴露必要方法
    - 最少知识：内部状态最小化，不存储无关信息
    """

    def __init__(self, total_players: int = 4, dealer_position: int = 0):
        """
        初始化

        Args:
            total_players: 玩家数量（4人麻将）
            dealer_position: 初始庄家位置（0-3）
        """
        self.total_players: int = total_players
        self.dealer_position: int = dealer_position  # 当前庄家位置（0-3）
        self.total_rounds_played: int = 0  # 已完成的总局数

    def advance_round(self, win_position: int | None, is_dealer_win: bool = False) -> None:
        """
        推进到下一局

        Args:
            win_position: 胡牌玩家位置（0-3），None表示荒庄
            is_dealer_win: 是否庄家自摸（仅在win_position不为None时有效）
        """
        self.total_rounds_played += 1

        # 连庄情况：庄家胡牌 或 荒庄 → 庄家位置不变
        if is_dealer_win or win_position is None:
            return

        # 下庄情况：其他玩家胡牌 → 胡牌者成为新庄家
        self.dealer_position = win_position

    def get_dealer(self) -> int:
        """获取当前庄家位置"""
        return self.dealer_position

    def is_dealer(self, position: int) -> bool:
        """
        判断指定位置是否为庄家

        Args:
            position: 玩家位置（0-3）
        Returns:
            是否为庄家
        """
        return position == self.dealer_position

    def reset(self) -> None:
        """重置游戏（新牌局开始）"""
        self.__init__(self.total_players)

    def __str__(self) -> str:
        """简洁的字符串表示"""
        return f"第{self.total_rounds_played + 1}局 - 玩家{self.dealer_position}庄"


# ========================================
# 使用示例
# ========================================
if __name__ == "__main__":
    round_info = RoundInfo()

    print("=== 初始状态 ===")
    print(round_info)  # 输出: 第1局 - 玩家0庄

    print("\n=== 场景1：庄家胡牌 ===")
    round_info.advance_round(win_position=0, is_dealer_win=True)
    print(round_info)  # 输出: 第2局 - 玩家0庄

    print("\n=== 场景2：荒庄 ===")
    round_info.advance_round(win_position=None)
    print(round_info)  # 输出: 第3局 - 玩家0庄

    print("\n=== 场景3：玩家2胡牌 ===")
    round_info.advance_round(win_position=2, is_dealer_win=False)
    print(round_info)  # 输出: 第4局 - 玩家2庄

    print(f"\n验证：玩家2是庄家吗？ {round_info.is_dealer(2)}")  # True
    print(f"验证：玩家0是庄家吗？ {round_info.is_dealer(0)}")  # False
