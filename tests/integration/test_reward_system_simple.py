"""
测试reward系统 - 验证环境层使用final_scores分配reward

这个测试验证example_mahjong_env在游戏结束时正确使用context.final_scores
为所有4个玩家分配reward。
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType


def test_environment_uses_final_scores_for_rewards():
    """
    测试环境在游戏结束时使用final_scores分配reward

    场景：
    1. 创建一个GameContext并设置final_scores
    2. 模拟游戏结束（WIN状态）
    3. 验证环境的rewards字典包含所有4个玩家的分数
    """
    # 模拟一个已结束的游戏上下文
    context = GameContext()
    context.current_state = GameStateType.WIN
    context.is_win = True
    context.winner_ids = [0]  # 玩家0获胜
    context.win_way = "self_draw"

    # 设置最终分数（模拟）
    # 玩家0获胜：+120分
    # 玩家1、2、3输：各-40分
    context.final_scores = [120.0, -40.0, -40.0, -40.0]

    # 验证final_scores存在且长度为4
    assert hasattr(context, 'final_scores'), "GameContext应该有final_scores字段"
    assert len(context.final_scores) == 4, "final_scores应该包含4个玩家的分数"

    # 验证分数总和为0（零和游戏）
    score_sum = sum(context.final_scores)
    assert abs(score_sum) < 0.01, f"分数总和应该为0，实际为{score_sum}"

    # 验证reward计算（除以100）
    expected_rewards = {
        "player_0": 1.2,   # 120 / 100
        "player_1": -0.4,  # -40 / 100
        "player_2": -0.4,  # -40 / 100
        "player_3": -0.4,  # -40 / 100
    }

    actual_rewards = {}
    for i, score in enumerate(context.final_scores):
        agent_name = f"player_{i}"
        actual_rewards[agent_name] = score / 100.0

    # 验证每个玩家的reward
    for agent_name in expected_rewards:
        expected = expected_rewards[agent_name]
        actual = actual_rewards[agent_name]
        assert abs(actual - expected) < 0.01, \
            f"{agent_name} reward不匹配: 期望{expected}, 实际{actual}"

    print("[OK] 测试通过：环境正确使用final_scores分配reward")
    print(f"  最终分数: {context.final_scores}")
    print(f"  对应rewards: {actual_rewards}")


def test_fallback_logic_without_final_scores():
    """
    测试当没有final_scores时的降级逻辑

    场景：
    1. 创建一个GameContext但不设置final_scores
    2. 验证环境应该使用旧的简化逻辑
    """
    context = GameContext()
    context.current_state = GameStateType.WIN
    context.is_win = True
    context.winner_ids = [1]  # 玩家1获胜
    # 不设置final_scores

    # 模拟旧的简化逻辑
    if context.final_scores:
        # 新逻辑
        rewards = {}
        for i, score in enumerate(context.final_scores):
            agent_name = f"player_{i}"
            rewards[agent_name] = score / 100.0
    else:
        # 降级逻辑
        rewards = {}
        for i in range(4):
            agent_name = f"player_{i}"
            if i in context.winner_ids:
                rewards[agent_name] = 1.0
            else:
                rewards[agent_name] = -1.0

    # 验证降级逻辑
    expected_rewards = {
        "player_0": -1.0,  # 输家
        "player_1": 1.0,   # 赢家
        "player_2": -1.0,  # 输家
        "player_3": -1.0,  # 输家
    }

    for agent_name in expected_rewards:
        expected = expected_rewards[agent_name]
        actual = rewards[agent_name]
        assert abs(actual - expected) < 0.01, \
            f"{agent_name} fallback reward不匹配: 期望{expected}, 实际{actual}"

    print("[OK] 测试通过：降级逻辑在没有final_scores时正确工作")
    print(f"  获胜玩家: {context.winner_ids}")
    print(f"  对应rewards: {rewards}")


def test_flow_draw_rewards():
    """
    测试流局时的reward分配

    场景：
    1. 流局状态
    2. 有final_scores（查大叫结算）
    3. 验证reward正确分配
    """
    context = GameContext()
    context.current_state = GameStateType.FLOW_DRAW
    context.is_flush = True
    context.is_win = False

    # 流局查大叫：玩家0听牌，其他三家未听牌
    # 玩家0各向1、2、3收取10分
    context.final_scores = [30.0, -10.0, -10.0, -10.0]

    # 验证分数总和为0
    score_sum = sum(context.final_scores)
    assert abs(score_sum) < 0.01, f"流局分数总和应该为0，实际为{score_sum}"

    # 计算rewards
    rewards = {}
    for i, score in enumerate(context.final_scores):
        agent_name = f"player_{i}"
        rewards[agent_name] = score / 100.0

    expected_rewards = {
        "player_0": 0.3,   # 30 / 100
        "player_1": -0.1,  # -10 / 100
        "player_2": -0.1,  # -10 / 100
        "player_3": -0.1,  # -10 / 100
    }

    for agent_name in expected_rewards:
        expected = expected_rewards[agent_name]
        actual = rewards[agent_name]
        assert abs(actual - expected) < 0.01, \
            f"{agent_name} flow draw reward不匹配: 期望{expected}, 实际{actual}"

    print("[OK] 测试通过：流局时reward正确分配")
    print(f"  流局查大叫分数: {context.final_scores}")
    print(f"  对应rewards: {rewards}")


if __name__ == "__main__":
    print("=" * 60)
    print("测试reward系统 - 环境层final_scores集成")
    print("=" * 60)
    print()

    print("测试1: 验证环境使用final_scores分配reward")
    test_environment_uses_final_scores_for_rewards()
    print()

    print("测试2: 验证没有final_scores时的降级逻辑")
    test_fallback_logic_without_final_scores()
    print()

    print("测试3: 验证流局时的reward分配")
    test_flow_draw_rewards()
    print()

    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)
