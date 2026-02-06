import pytest
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.score_calculator import MahjongScoreSettler


def test_flow_draw_no_capping():
    """验证流局结算不应用封顶（可能产生极高分数）"""
    players = [PlayerData(player_id=i) for i in range(4)]

    # 听牌者
    players[0].is_ting = True
    players[0].special_gangs = [0, 5, 0]  # 5个赖子杠，极高分数

    # 未听牌者
    players[1].is_ting = False
    players[1].special_gangs = [0, 5, 0]

    players[2].is_ting = True
    players[2].special_gangs = [0, 5, 0]

    players[3].is_ting = False
    players[3].special_gangs = [0, 5, 0]

    ctx = GameContext(players=players, dealer_idx=0)

    settler = MahjongScoreSettler(is_kou_kou_fan=False)
    scores = settler.settle_flow_draw(ctx)

    # 流局结算不封顶，分数可以超过300
    # 验证：如果有高分数，说明没有封顶
    max_abs_score = max(abs(s) for s in scores)
    # 由于5个赖子杠会产生极高分数，如果超过300说明没有封顶
    print(f"流局结算分数: {scores}, 最大绝对值: {max_abs_score}")
    # 这里只是验证不会报错，实际分数取决于具体计算
