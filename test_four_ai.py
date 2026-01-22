#!/usr/bin/env python3
"""
测试用例：四个AI玩家（随机策略）对战到结束

用于验证游戏逻辑是否正确运行，无需人工干预。
"""

import sys
sys.path.insert(0, '.')

from example_mahjong_env import WuhanMahjongEnv
from src.mahjong_rl.agents.ai.random_strategy import RandomStrategy


def test_four_ai_game():
    """测试四个AI玩家对战"""
    print("=" * 60)
    print("测试：四个AI玩家对战")
    print("=" * 60)

    # 创建环境
    env = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,
        enable_logging=False
    )

    # 创建四个AI策略
    strategies = [RandomStrategy() for _ in range(4)]

    # 重置环境
    env.reset(seed=42)
    print(f"\n游戏开始！初始agent: {env.agent_selection}")
    print(f"初始状态: {env.state_machine.current_state_type.name}")

    step_count = 0
    max_steps = 1000  # 防止无限循环

    # 游戏主循环
    try:
        for agent in env.agent_iter():
            step_count += 1

            if step_count > max_steps:
                print(f"\n警告：达到最大步数 {max_steps}，强制退出")
                break

            obs, reward, terminated, truncated, info = env.last()

            # 每10步打印一次状态
            # if step_count % 10 == 0:
            print(f"\n步骤 {step_count}:")
            print(f"  当前agent: {agent}")
            print(f"  当前状态: {env.state_machine.current_state_type.name}")
            print(f"  剩余牌墙: {len(env.context.wall)}张")

            if terminated or truncated:
                print(f"\n{agent} 游戏结束")
                action = None
            else:
                # AI选择动作
                action_mask = obs['action_mask']
                agent_idx = env.agents_name_mapping[agent]
                strategy = strategies[agent_idx]
                action = strategy.choose_action(obs, action_mask)

                # 打印动作（调试用）
                if step_count <= 20:  # 只打印前20步
                    print(f"  {agent} 动作: {action}")

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)

            # 检查游戏是否结束
            if terminated or truncated:
                print(f"\n游戏结束！总步数: {step_count}")
                break

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 打印最终结果
        print("\n" + "=" * 60)
        print("游戏结果:")
        print("=" * 60)

        if env.context.is_win:
            winners = env.context.winner_ids
            win_way = env.context.win_way
            win_way_names = {0: "自摸", 1: "抢杠", 2: "杠开", 3: "点炮"}
            print(f"获胜者: 玩家{winners}")
            print(f"胜利方式: {win_way_names.get(win_way, '未知')}")

            # 打印每个玩家的分数
            for i, player in enumerate(env.context.players):
                print(f"玩家{i}分数: {player.score}")
        else:
            print("荒牌流局")

        print(f"总步数: {step_count}")
        print(f"剩余牌墙: {len(env.context.wall)}张")

        # 打印每个玩家的手牌
        print("\n玩家手牌:")
        for i, player in enumerate(env.context.players):
            from src.mahjong_rl.core.constants import Tiles
            hand_names = [Tiles.get_tile_name(t) for t in sorted(player.hand_tiles)]
            print(f"  玩家{i} ({len(player.hand_tiles)}张): {', '.join(hand_names)}")

        env.close()

    return step_count


def test_multiple_games(num_games=5):
    """测试多局游戏"""
    print("=" * 60)
    print(f"测试：{num_games}局AI对战")
    print("=" * 60)

    total_steps = 0

    for game_num in range(num_games):
        print(f"\n第 {game_num + 1} 局:")
        print("-" * 40)

        steps = test_four_ai_game()
        total_steps += steps

    print("\n" + "=" * 60)
    print("总结:")
    print("=" * 60)
    print(f"总局数: {num_games}")
    print(f"平均步数: {total_steps / num_games:.1f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='测试四个AI玩家对战')
    parser.add_argument('--games', type=int, default=1, help='总局数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    if args.games == 1:
        test_four_ai_game()
    else:
        test_multiple_games(args.games)

    print("\n测试完成！")
