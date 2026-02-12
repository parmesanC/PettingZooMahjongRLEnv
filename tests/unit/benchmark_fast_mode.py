"""
性能基准测试：fast_mode 加速效果

对比 fast_mode=True 和 fast_mode=False 的性能差异
"""

import time
import numpy as np

from example_mahjong_env import WuhanMahjongEnv


def run_n_episodes(env, episodes=10):
    """
    运行指定数量的游戏回合

    Args:
        env: 麻将环境
        episodes: 回合数

    Returns:
        总步数
    """
    total_steps = 0

    for episode in range(episodes):
        env.reset()
        done = False

        while not done:
            # 获取当前玩家
            agent = env.agent_selection

            # 获取动作空间并采样随机动作
            action_space = env.action_space(agent)
            action = (action_space[0].sample(), action_space[1].sample())

            # 执行动作
            obs, reward, termination, truncation, info = env.step(action)

            # 检查游戏是否结束
            done = termination or truncation
            total_steps += 1

    return total_steps


def test_fast_mode_performance(benchmark_episodes=50):
    """
    性能基准测试：对比 fast_mode 的性能差异

    Args:
        benchmark_episodes: 基准测试的回合数
    """
    print(f"\n{'='*60}")
    print(f"性能基准测试：fast_mode vs normal_mode")
    print(f"回合数: {benchmark_episodes}")
    print(f"{'='*60}\n")

    # 测试普通模式
    print("测试普通模式 (fast_mode=False)...")
    env_normal = WuhanMahjongEnv(fast_mode=False, enable_logging=False)
    start_normal = time.time()
    steps_normal = run_n_episodes(env_normal, episodes=benchmark_episodes)
    time_normal = time.time() - start_normal

    # 测试快速模式
    print("测试快速模式 (fast_mode=True)...")
    env_fast = WuhanMahjongEnv(fast_mode=True, enable_logging=False)
    start_fast = time.time()
    steps_fast = run_n_episodes(env_fast, episodes=benchmark_episodes)
    time_fast = time.time() - start_fast

    # 计算统计信息
    speedup = time_normal / time_fast if time_fast > 0 else float('inf')
    time_saved = time_normal - time_fast
    time_saved_pct = (time_saved / time_normal) * 100

    # 输出结果
    print(f"\n{'='*60}")
    print(f"结果:")
    print(f"{'='*60}")
    print(f"普通模式:")
    print(f"  - 总时间: {time_normal:.3f} 秒")
    print(f"  - 总步数: {steps_normal}")
    print(f"  - 平均每回合时间: {time_normal/benchmark_episodes:.3f} 秒")
    print(f"  - 每秒步数: {steps_normal/time_normal:.1f}")

    print(f"\n快速模式:")
    print(f"  - 总时间: {time_fast:.3f} 秒")
    print(f"  - 总步数: {steps_fast}")
    print(f"  - 平均每回合时间: {time_fast/benchmark_episodes:.3f} 秒")
    print(f"  - 每秒步数: {steps_fast/time_fast:.1f}")

    print(f"\n性能提升:")
    print(f"  - 加速倍数: {speedup:.2f}x")
    print(f"  - 节省时间: {time_saved:.3f} 秒")
    print(f"  - 时间节省率: {time_saved_pct:.1f}%")
    print(f"{'='*60}\n")

    # 验证快速模式更快
    assert time_fast < time_normal, \
        f"快速模式应该更快，但实际时间: fast={time_fast:.3f}s, normal={time_normal:.3f}s"

    return {
        'time_normal': time_normal,
        'time_fast': time_fast,
        'speedup': speedup,
        'time_saved_pct': time_saved_pct,
        'steps_normal': steps_normal,
        'steps_fast': steps_fast
    }


if __name__ == "__main__":
    # 运行性能基准测试
    results = test_fast_mode_performance(benchmark_episodes=50)

    print("✅ 性能基准测试通过！")
    print(f"   加速效果: {results['speedup']:.2f}x")
