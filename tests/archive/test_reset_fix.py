"""验证初始化重构的测试脚本"""

import sys
import os

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from example_mahjong_env import WuhanMahjongEnv
    from src.mahjong_rl.rules.round_info import RoundInfo

    print("=" * 60)
    print("测试1: 创建环境并重置")
    print("=" * 60)

    env = WuhanMahjongEnv(render_mode=None, training_phase=3)
    print(f"✓ Env 创建成功")
    print(f"  初始 round_info: {env.round_info}")

    obs, info = env.reset(seed=42)
    print(f"✓ Reset 成功")
    print(f"  庄家索引: {env.context.dealer_idx}")
    print(f"  当前玩家索引: {env.context.current_player_idx}")
    print(f"  RoundInfo: {env.round_info}")
    print(f"  玩家{env.context.dealer_idx}是庄家: {env.context.players[env.context.dealer_idx].is_dealer}")
    print()

    print("=" * 60)
    print("测试2: 庄家胡牌（连庄）")
    print("=" * 60)

    current_dealer = env.context.dealer_idx
    env.context.is_win = True
    env.context.winner_ids = [current_dealer]  # 庄家胡
    env._update_round_info()
    print(f"  当前庄家: {current_dealer}")
    print(f"  RoundInfo: {env.round_info}")
    print(f"  预期: 庄家位置不变 (连庄)")

    env.reset()
    print(f"  新局庄家索引: {env.context.dealer_idx}")
    assert env.context.dealer_idx == current_dealer, "庄家应该连庄"
    print(f"✓ 庄家连庄正确")
    print()

    print("=" * 60)
    print("测试3: 闲家胡牌（下庄）")
    print("=" * 60)

    current_dealer = env.context.dealer_idx
    non_dealer = (current_dealer + 1) % 4
    print(f"  当前庄家: {current_dealer}")
    print(f"  胡牌玩家: {non_dealer} (闲家)")

    env.context.is_win = True
    env.context.winner_ids = [non_dealer]
    env._update_round_info()
    print(f"  RoundInfo: {env.round_info}")
    print(f"  预期: 新庄家为 {non_dealer}")

    env.reset()
    print(f"  新局庄家索引: {env.context.dealer_idx}")
    assert env.context.dealer_idx == non_dealer, f"新庄家应该是{non_dealer}"
    print(f"✓ 闲家成为新庄家正确")
    print()

    print("=" * 60)
    print("测试4: 荒庄（连庄）")
    print("=" * 60)

    current_dealer = env.context.dealer_idx
    print(f"  当前庄家: {current_dealer}")

    env.context.is_win = False
    env.context.is_flush = True
    env._update_round_info()
    print(f"  RoundInfo: {env.round_info}")
    print(f"  预期: 庄家位置不变 (连庄)")

    env.reset()
    print(f"  新局庄家索引: {env.context.dealer_idx}")
    assert env.context.dealer_idx == current_dealer, "荒庄应该庄家连庄"
    print(f"✓ 荒庄连庄正确")
    print()

    print("=" * 60)
    print("✓ 所有测试通过!")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
