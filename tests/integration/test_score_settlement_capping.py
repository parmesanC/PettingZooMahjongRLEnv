import pytest
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import WinWay
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.score_calculator import MahjongScoreSettler
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.win_detector import WinCheckResult, WinType


def test_settlement_with_normal_capping():
    """测试普通封顶场景的完整结算"""
    # 创建玩家
    players = [PlayerData(player_id=i) for i in range(4)]
    players[0].is_win = True  # 玩家0胡牌
    players[0].is_dealer = True

    # 设置一些开口和杠牌以产生高分
    players[1].melds = []  # 简化测试
    players[2].melds = []
    players[3].melds = []

    ctx = GameContext(players=players, dealer_idx=0)
    ctx.win_way = WinWay.SELF_DRAW.value

    win_result = WinCheckResult(
        can_win=True,
        win_type=[WinType.HARD_WIN],
        min_wild_need=0
    )

    settler = MahjongScoreSettler(is_kou_kou_fan=False)
    scores = settler.settle(win_result, ctx)

    # 验证分数
    assert scores[0] > 0  # 赢家得分
    assert all(s <= 0 for i, s in enumerate(scores) if i != 0)  # 输家失分


def test_settlement_with_golden_cap():
    """测试金顶场景的完整结算"""
    # 创建高分场景（多次开口和杠牌）
    players = [PlayerData(player_id=i) for i in range(4)]
    players[0].is_win = True
    players[0].is_dealer = True
    players[0].special_gangs = [0, 2, 0]  # 2个赖子杠，高分

    # 输家也设置高分
    for i in range(1, 4):
        players[i].special_gangs = [0, 2, 0]  # 每个输家2个赖子杠

    ctx = GameContext(players=players, dealer_idx=0)
    ctx.win_way = WinWay.SELF_DRAW.value

    win_result = WinCheckResult(
        can_win=True,
        win_type=[WinType.HARD_WIN, WinType.PURE_FLUSH],  # 大胡
        min_wild_need=0
    )

    settler = MahjongScoreSettler(is_kou_kou_fan=False)
    scores = settler.settle(win_result, ctx)

    # 验证输家分数不超过400（金顶上限）
    for i, score in enumerate(scores):
        if i != 0 and score < 0:  # 输家
            assert abs(score) <= 400, f"输家{i}分数{score}超过金顶400"


def test_settlement_koukou_mode_golden_cap():
    """测试口口翻模式金顶"""
    players = [PlayerData(player_id=i) for i in range(4)]
    players[0].is_win = True
    players[0].is_dealer = True
    players[0].special_gangs = [0, 2, 0]

    for i in range(1, 4):
        players[i].special_gangs = [0, 2, 0]

    ctx = GameContext(players=players, dealer_idx=0)
    ctx.win_way = WinWay.SELF_DRAW.value

    win_result = WinCheckResult(
        can_win=True,
        win_type=[WinType.HARD_WIN, WinType.PURE_FLUSH],
        min_wild_need=0
    )

    settler = MahjongScoreSettler(is_kou_kou_fan=True)
    scores = settler.settle(win_result, ctx)

    # 验证输家分数不超过500（口口翻金顶上限）
    for i, score in enumerate(scores):
        if i != 0 and score < 0:
            assert abs(score) <= 500, f"输家{i}分数{score}超过口口翻金顶500"
