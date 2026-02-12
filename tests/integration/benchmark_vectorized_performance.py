"""
性能基准测试：向量化训练 vs 单环境训练

比较单环境和向量化环境的训练性能，验证向量化训练的加速效果。

预期结果：
- 2 环境: 1.5-2.5x 加速
- 4 环境: 2.5-3.5x 加速
- 8 环境: 3.5-4.5x 加速
"""

import time
import sys
import os
from typing import Dict, List

# 添加当前目录到 Python 路径
sys.path.insert(0, os.getcwd())

from src.drl.config import get_quick_test_config, Config
from src.drl.trainer import NFSPTrainer


def get_benchmark_config(episodes: int = 5) -> Config:
    """获取基准测试配置（小规模快速测试）"""
    config = get_quick_test_config()
    # 使用 quick_test 模式，这样 actual_total_episodes 会使用 quick_test_episodes
    config.training.mode = 'quick_test'
    # 设置小规模测试回合数
    config.training.quick_test_episodes = episodes
    config.training.device = 'cpu'
    return config


def run_single_env_benchmark(episodes: int = 5) -> Dict[str, float]:
    """
    运行单环境基准测试

    Args:
        episodes: 测试回合数

    Returns:
        统计字典
    """
    print(f"\n{'='*60}")
    print(f"单环境基准测试 ({episodes} 回合)")
    print(f"{'='*60}")

    config = get_benchmark_config(episodes)

    trainer = NFSPTrainer(
        config=config,
        device="cpu",
        use_vectorized_env=False,
    )

    start_time = time.time()
    stats = trainer.train()
    elapsed = time.time() - start_time

    trainer.close()

    print(f"\n单环境结果:")
    print(f"  - 回合数: {episodes}")
    print(f"  - 总耗时: {elapsed:.2f} 秒")
    print(f"  - 平均耗时/回合: {elapsed/episodes:.3f} 秒")

    return {
        "episodes": episodes,
        "total_time": elapsed,
        "avg_time_per_episode": elapsed / episodes,
    }


def run_vectorized_benchmark(num_envs: int, episodes: int = 5) -> Dict[str, float]:
    """
    运行向量化环境基准测试

    Args:
        num_envs: 环境数量
        episodes: 总测试回合数

    Returns:
        统计字典
    """
    print(f"\n{'='*60}")
    print(f"向量化基准测试 ({num_envs} 环境, {episodes} 回合)")
    print(f"{'='*60}")

    config = get_benchmark_config(episodes)

    trainer = NFSPTrainer(
        config=config,
        device="cpu",
        use_vectorized_env=True,
        num_envs=num_envs,
    )

    start_time = time.time()
    stats = trainer.train()
    elapsed = time.time() - start_time

    trainer.close()

    print(f"\n向量化 ({num_envs} 环境) 结果:")
    print(f"  - 回合数: {episodes}")
    print(f"  - 总耗时: {elapsed:.2f} 秒")
    print(f"  - 平均耗时/回合: {elapsed/episodes:.3f} 秒")

    return {
        "num_envs": num_envs,
        "episodes": episodes,
        "total_time": elapsed,
        "avg_time_per_episode": elapsed / episodes,
    }


def compare_performance(single_stats: Dict, vec_stats_list: List[Dict]) -> None:
    """
    比较并打印性能对比结果

    Args:
        single_stats: 单环境统计
        vec_stats_list: 向量化统计列表
    """
    print(f"\n{'='*60}")
    print("性能对比结果")
    print(f"{'='*60}")

    single_time = single_stats["total_time"]

    print(f"\n单环境基准: {single_time:.2f} 秒")
    print(f"\n向量化环境:")
    print(f"{'环境数':<10} {'总耗时(秒)':<15} {'加速比':<10} {'效率':<10}")
    print("-" * 50)

    for vec_stats in vec_stats_list:
        num_envs = vec_stats["num_envs"]
        vec_time = vec_stats["total_time"]
        speedup = single_time / vec_time
        efficiency = speedup / num_envs

        # 添加颜色标记
        if speedup >= num_envs * 0.8:
            status = "优秀"
        elif speedup >= num_envs * 0.5:
            status = "良好"
        else:
            status = "需优化"

        print(f"{num_envs:<10} {vec_time:<15.2f} {speedup:<10.2f}x {efficiency:<10.1%} [{status}]")


def main():
    """主函数"""
    print("="*60)
    print("向量化训练性能基准测试")
    print("="*60)

    # 测试参数（减少回合数以快速测试）
    episodes = 20
    env_counts = [2, 4]  # 只测试 2 和 4 个环境

    # 运行单环境基准测试
    single_stats = run_single_env_benchmark(episodes)

    # 运行向量化基准测试
    vec_stats_list = []
    for num_envs in env_counts:
        vec_stats = run_vectorized_benchmark(num_envs, episodes)
        vec_stats_list.append(vec_stats)

    # 比较性能
    compare_performance(single_stats, vec_stats_list)

    print(f"\n{'='*60}")
    print("测试完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
