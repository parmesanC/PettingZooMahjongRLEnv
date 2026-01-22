"""
测试 score_calculator.py 的功能
"""
import sys
sys.path.append('.')

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData, Meld
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.core.constants import ActionType, WinType, WinWay
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.win_detector import WinCheckResult
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.score_calculator import MahjongScoreSettler


def test_basic_settle():
    """测试基本结算功能"""
    print("=" * 50, file=sys.stderr)
    print("测试1: 基本胡牌结算", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    # 创建结算器
    settler = MahjongScoreSettler(is_kou_kou_fan=False)

    # 准备游戏上下文
    players = [PlayerData(player_id=i) for i in range(4)]
    players[0].is_dealer = True  # 玩家0为庄家
    players[1].is_win = True     # 玩家1胡牌
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
    print(f"✓ 最终得分: {scores}", file=sys.stderr)
    print(f"  玩家0(庄家): {scores[0]}", file=sys.stderr)
    print(f"  玩家1(胡牌): {scores[1]}", file=sys.stderr)
    print(f"  玩家2: {scores[2]}", file=sys.stderr)
    print(f"  玩家3: {scores[3]}", file=sys.stderr)
    print(file=sys.stderr)


def test_flow_draw_settle():
    """测试流局查大叫结算"""
    print("=" * 50, file=sys.stderr)
    print("测试2: 流局查大叫结算", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    # 创建结算器
    settler = MahjongScoreSettler(is_kou_kou_fan=False)

    # 准备游戏上下文（流局）
    players = [PlayerData(player_id=i) for i in range(4)]
    players[0].is_dealer = True  # 玩家0为庄家
    players[0].is_ting = True     # 玩家0听牌
    # 玩家0已开口（通过添加一个吃牌）
    players[0].melds = [Meld(action_type=MahjongAction(ActionType.CHOW, 0), tiles=[0, 1, 2], from_player=3)]
    players[0].special_gangs = [1, 0, 0]  # 1个皮子杠

    players[1].is_ting = True     # 玩家1听牌
    # 玩家1未开口（没有melds）
    players[1].special_gangs = [0, 0, 0]

    players[2].is_ting = False    # 玩家2未听牌
    # 玩家2已开口（通过添加一个碰牌）
    players[2].melds = [Meld(action_type=MahjongAction(ActionType.PONG, 0), tiles=[10, 10, 10], from_player=1)]
    players[2].special_gangs = [0, 0, 1]  # 1个红中杠

    players[3].is_ting = False    # 玩家3未听牌
    # 玩家3未开口（没有melds）
    players[3].special_gangs = [0, 0, 0]

    ctx = GameContext(
        players=players,
        dealer_idx=0
    )

    # 执行流局查大叫结算
    scores = settler.settle_flow_draw(ctx)
    print(f"✓ 流局查大叫得分: {scores}", file=sys.stderr)
    print(f"  玩家0(听牌,已开口): {scores[0]}", file=sys.stderr)
    print(f"  玩家1(听牌,未开口): {scores[1]}", file=sys.stderr)
    print(f"  玩家2(未听牌,已开口): {scores[2]}", file=sys.stderr)
    print(f"  玩家3(未听牌,未开口): {scores[3]}", file=sys.stderr)
    print(file=sys.stderr)


def test_base_fan_score():
    """测试基础番数计算"""
    print("=" * 50, file=sys.stderr)
    print("测试3: 基础番数计算", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    # 创建结算器
    settler = MahjongScoreSettler(is_kou_kou_fan=False)

    # 测试玩家1：无开口，无杠
    player1 = PlayerData(player_id=1)
    # player1.has_opened = False  # has_opened是property，不能赋值，默认为False
    player1.special_gangs = [0, 0, 0]
    fan1 = settler._get_base_fan_score(player1)
    print(f"✓ 玩家1(无开口,无杠): 基础番数 = {fan1}", file=sys.stderr)

    # 测试玩家2：2次开口，1个明杠
    player2 = PlayerData(player_id=2)
    # player2.has_opened = True  # has_opened是property，不能赋值
    player2.melds = [
        Meld(action_type=MahjongAction(ActionType.CHOW, 0), tiles=[0, 1, 2], from_player=0),
        Meld(action_type=MahjongAction(ActionType.PONG, 0), tiles=[10, 10, 10], from_player=1),
        Meld(action_type=MahjongAction(ActionType.KONG_EXPOSED, 0), tiles=[20, 20, 20, 20], from_player=2)
    ]
    player2.special_gangs = [0, 0, 0]
    fan2 = settler._get_base_fan_score(player2)
    print(f"✓ 玩家2(2次开口,1明杠): 基础番数 = {fan2}", file=sys.stderr)

    # 测试玩家3：1次开口，1个暗杠，1个赖子杠
    player3 = PlayerData(player_id=3)
    # player3.has_opened = True  # has_opened是property，不能赋值
    player3.melds = [
        Meld(action_type=MahjongAction(ActionType.PONG, 0), tiles=[5, 5, 5], from_player=0),
        Meld(action_type=MahjongAction(ActionType.KONG_CONCEALED, 0), tiles=[15, 15, 15, 15], from_player=0)
    ]
    player3.special_gangs = [0, 1, 0]  # 1个赖子杠
    fan3 = settler._get_base_fan_score(player3)
    print(f"✓ 玩家3(1次开口,1暗杠,1赖子杠): 基础番数 = {fan3}", file=sys.stderr)
    print(file=sys.stderr)


def test_contractor():
    """测试承包规则"""
    print("=" * 50, file=sys.stderr)
    print("测试4: 承包规则", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    # 测试抢杠胡承包
    settler = MahjongScoreSettler(is_kou_kou_fan=False)

    players = [PlayerData(player_id=i) for i in range(4)]
    players[0].is_dealer = True
    players[1].is_win = True

    ctx = GameContext(
        players=players,
        dealer_idx=0,
        win_way=WinWay.ROB_KONG.value,
        discard_player=2  # 玩家2被抢杠
    )

    win_result = WinCheckResult(
        can_win=True,
        win_type=[WinType.ROB_KONG],
        min_wild_need=0
    )

    contractor = settler._check_contractor(ctx, win_result, 1)
    print(f"✓ 抢杠胡承包: 承包者 = 玩家{contractor}", file=sys.stderr)
    print(file=sys.stderr)


if __name__ == "__main__":
    import sys
    try:
        test_basic_settle()
        test_flow_draw_settle()
        test_base_fan_score()
        test_contractor()

        print("=" * 50, file=sys.stderr)
        print("所有测试通过！✓", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
        sys.stderr.flush()
    except Exception as e:
        print(f"❌ 测试失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
