"""
测试脚本：重现 KONG_LAZY 错误

该脚本用于调试赖子杠的错误，目标是：
1. 运行游戏直到出现 KONG_LAZY 错误
2. 收集详细的调试信息
3. 分析问题根源
"""

import sys
import traceback
from example_mahjong_env import WuhanMahjongEnv


def test_kong_lazy_error():
    """测试赖子杠错误"""
    print("=" * 80)
    print("测试 KONG_LAZY 错误重现")
    print("=" * 80)

    # 创建环境
    env = WuhanMahjongEnv(render_mode='human', training_phase=3, enable_logging=False)

    # 重置环境
    print("\n重置环境...")
    observation, info = env.reset(seed=42)
    print(f"初始agent: {env.agent_selection}")
    print(f"赖子: {env.context.lazy_tile}")
    print(f"皮子: {env.context.skin_tile}")
    print(f"红中: {env.context.red_dragon}")

    # 模拟游戏直到出现错误或达到最大步数
    max_steps = 50
    step_count = 0

    try:
        for step_count in range(1, max_steps + 1):
            print(f"\n{'=' * 80}")
            print(f"步骤 {step_count}")
            print('=' * 80)

            # 获取当前agent和观测
            current_agent = env.agent_selection
            if current_agent is None:
                print("游戏结束！")
                break

            obs, reward, terminated, truncated, info = env.last()

            print(f"当前agent: {current_agent}")
            print(f"当前状态: {env.state_machine.current_state_type.name}")
            print(f"剩余牌墙: {len(env.context.wall)}张")

            # 打印当前玩家手牌
            agent_idx = env.agents_name_mapping[current_agent]
            player = env.context.players[agent_idx]
            print(f"玩家{agent_idx}手牌: {sorted(player.hand_tiles)}")
            print(f"玩家{agent_idx}melds: {[(m.action_type.action_type.name, m.tiles) for m in player.melds]}")

            # 打印action_mask信息
            if 'action_mask' in obs:
                action_mask = obs['action_mask']
                print(f"action_mask 形状: {action_mask.shape}")
                print(f"action_mask 非零位置: {[(i, int(action_mask[i])) for i in range(len(action_mask)) if action_mask[i] > 0]}")

                # 检查特殊杠位
                print(f"  KONG_LAZY 位 (174-207): {action_mask[174:208]}")
                print(f"  KONG_RED 位 (140+31=171): {action_mask[171]}")
                print(f"  KONG_SKIN 位 (208-241): {action_mask[208:242]}")

            # 如果游戏结束，退出
            if terminated or truncated:
                print("游戏结束！")
                break

            # 使用RandomStrategy选择动作
            from src.mahjong_rl.agents.ai.random_strategy import RandomStrategy
            strategy = RandomStrategy()

            try:
                action = strategy.choose_action(obs, obs.get('action_mask', None))
                print(f"{current_agent} 动作: {action}")

                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)

                print(f"奖励: {reward}")
                print(f"是否结束: {terminated}")

            except Exception as e:
                print(f"\n{'!' * 80}")
                print(f"错误发生在步骤 {step_count}")
                print(f"{'!' * 80}")
                print(f"错误类型: {type(e).__name__}")
                print(f"错误信息: {e}")
                print(f"\n完整错误堆栈:")
                traceback.print_exc()

                # 打印详细的游戏状态
                print(f"\n错误发生时的游戏状态:")
                print(f"  当前状态: {env.state_machine.current_state_type.name}")
                print(f"  当前玩家: {current_agent} (索引 {agent_idx})")
                print(f"  玩家手牌: {sorted(player.hand_tiles)}")
                print(f"  玩家melds: {[(m.action_type.action_type.name, m.tiles) for m in player.melds]}")
                print(f"  赖子: {env.context.lazy_tile}")
                print(f"  皮子: {env.context.skin_tile}")
                print(f"  红中: {env.context.red_dragon}")

                break

    except KeyboardInterrupt:
        print("\n用户中断测试")

    finally:
        print(f"\n{'=' * 80}")
        print(f"测试完成，共执行 {step_count} 步")
        print('=' * 80)

        # 打印状态机日志
        if env.state_machine.logger:
            print("\n状态机转换日志:")
            for log_entry in env.state_machine.logger.get_history()[-20:]:  # 只显示最后20条
                print(f"  {log_entry}")


if __name__ == "__main__":
    test_kong_lazy_error()
