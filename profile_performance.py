"""
性能分析脚本 - 分析训练瓶颈

使用 cProfile 和 snakeviz 分析性能瓶颈
"""

import cProfile
import pstats
import io
import time
import numpy as np
from collections import defaultdict
from contextlib import contextmanager

# 设置环境
import os
os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))

from example_mahjong_env import WuhanMahjongEnv
from src.drl.config import get_quick_test_config
from src.drl.agent import NFSPAgentPool, RandomOpponent


class PerformanceProfiler:
    """性能分析器"""

    def __init__(self):
        self.timings = defaultdict(list)
        self.call_counts = defaultdict(int)

        # 缓存统计
        self.validator_instantiations = 0
        self.win_checker_instantiations = 0
        self.cache_stats = {"hits": 0, "misses": 0}

    @contextmanager
    def profile(self, name):
        """上下文管理器：计时代码块"""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.timings[name].append(elapsed)
        self.call_counts[name] += 1

    def report(self):
        """打印性能报告"""
        print("\n" + "=" * 80)
        print("性能分析报告")
        print("=" * 80)

        total = sum(sum(times) for times in self.timings.values())

        # 按总时间排序
        items = sorted(self.timings.items(), key=lambda x: sum(x[1]), reverse=True)

        for name, times in items:
            total_time = sum(times)
            count = len(times)
            avg_time = total_time / count if count > 0 else 0
            pct = (total_time / total * 100) if total > 0 else 0

            print(f"{name:40s} | 总: {total_time:7.3f}s | 平均: {avg_time:7.5f}s | 调用: {count:5d} | 占比: {pct:5.1f}%")

        print("=" * 80)
        print(f"总时间: {total:.3f}s")
        print("=" * 80)

        # 缓存统计报告
        self._report_cache_stats()

    def _report_cache_stats(self):
        """报告缓存统计"""
        print("\n" + "-" * 80)
        print("缓存统计报告")
        print("-" * 80)

        # 对象实例化统计
        print(f"ActionValidator 实例化次数: {self.validator_instantiations}")
        print(f"WuhanMahjongWinChecker 实例化次数: {self.win_checker_instantiations}")

        # 缓存命中率
        total_cache_accesses = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total_cache_accesses > 0:
            hit_rate = self.cache_stats["hits"] / total_cache_accesses * 100
            print(f"ActionMask 缓存命中率: {hit_rate:.1f}% ({self.cache_stats['hits']}/{total_cache_accesses})")
        else:
            print("ActionMask 缓存: 无数据")

        print("-" * 80)

    def track_instantiation(self, component_type: str) -> None:
        """追踪对象实例化"""
        if component_type == "validator":
            self.validator_instantiations += 1
        elif component_type == "win_checker":
            self.win_checker_instantiations += 1

    def track_cache_hit(self) -> None:
        """追踪缓存命中"""
        self.cache_stats["hits"] += 1

    def track_cache_miss(self) -> None:
        """追踪缓存未命中"""
        self.cache_stats["misses"] += 1


def profile_single_episode():
    """分析单个episode的性能"""
    print("\n### 分析单个 Episode ###")

    profiler = PerformanceProfiler()

    # 创建环境
    with profiler.profile("创建环境"):
        env = WuhanMahjongEnv(
            render_mode=None,
            training_phase=1,
            enable_logging=False,
        )

    # 创建agents
    config = get_quick_test_config()
    device = "cpu"  # 使用CPU避免GPU开销干扰
    with profiler.profile("创建AgentPool"):
        agent_pool = NFSPAgentPool(config=config, device=device, num_agents=4, share_parameters=True)
    random_opponent = RandomOpponent()

    # 运行几个episode
    num_episodes = 5

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")

        with profiler.profile("env.reset"):
            obs, _ = env.reset()

        episode_steps = 0

        # PettingZoo 标准循环
        for agent_name in env.agent_iter():
            episode_steps += 1

            with profiler.profile("env.last"):
                obs, reward, terminated, truncated, info = env.last()

            agent_idx = int(agent_name.split("_")[1])
            action_mask = obs["action_mask"]

            with profiler.profile("选择动作"):
                action_type, action_param = random_opponent.choose_action(obs, action_mask)

            with profiler.profile("env.step"):
                env.step((action_type, action_param))

            if terminated or truncated:
                break

        print(f"  步数: {episode_steps}")

    profiler.report()

    # 关键指标
    print("\n### 关键指标 ###")
    reset_time = sum(profiler.timings["env.reset"])
    last_time = sum(profiler.timings["env.last"])
    step_time = sum(profiler.timings["env.step"])
    total = reset_time + last_time + step_time
    print(f"env.reset:   {reset_time:8.3f}s ({reset_time/total*100:.1f}%)")
    print(f"env.last:    {last_time:8.3f}s ({last_time/total*100:.1f}%)")
    print(f"env.step:    {step_time:8.3f}s ({step_time/total*100:.1f}%)")
    print(f"总计:        {total:8.3f}s")

    return profiler


def profile_with_cprofile():
    """使用 cProfile 分析"""
    print("\n### cProfile 分析 ###")

    profiler = cProfile.Profile()
    profiler.enable()

    # 创建环境和agents
    env = WuhanMahjongEnv(render_mode=None, training_phase=1, enable_logging=False)
    config = get_quick_test_config()
    agent_pool = NFSPAgentPool(config=config, device="cpu", num_agents=4, share_parameters=True)
    random_opponent = RandomOpponent()

    # 运行3个episode
    for _ in range(3):
        obs, _ = env.reset()
        for agent_name in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            agent_idx = int(agent_name.split("_")[1])
            action_mask = obs["action_mask"]
            action_type, action_param = random_opponent.choose_action(obs, action_mask)
            env.step((action_type, action_param))
            if terminated or truncated:
                break

    profiler.disable()

    # 打印结果
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # 打印前30个函数

    print(s.getvalue())


def profile_observation_building():
    """分析观测构建性能"""
    print("\n### 观测构建性能分析 ###")

    import time
    from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder

    env = WuhanMahjongEnv(render_mode=None, training_phase=1, enable_logging=False)
    obs, _ = env.reset()

    profiler = PerformanceProfiler()

    # 测试观测构建
    for i in range(100):
        with profiler.profile("构建单agent观测"):
            obs, _, _, _, _ = env.last()

        # 测试全局观测构建
        if i == 0:
            context = env.unwrapped.context
            obs_builder = env.unwrapped.state_machine.observation_builder

            with profiler.profile("构建全局观测 (首次)"):
                global_obs = obs_builder.build_global_observation(context, training_phase=1)

            with profiler.profile("构建全局观测 (后续)"):
                for _ in range(10):
                    global_obs = obs_builder.build_global_observation(context, training_phase=1)

    profiler.report()


def profile_network_forward():
    """分析网络前向传播性能"""
    print("\n### 网络前向传播性能分析 ###")

    import torch
    from src.drl.config import get_quick_test_config
    from src.drl.network import create_networks

    config = get_quick_test_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 创建网络
    actor_net, policy_net = create_networks(config, device)

    # 创建完整的虚拟输入（匹配 Wuhan7P4LObservationBuilder 的输出）
    # 注意：所有随机生成的范围必须匹配网络 embedding 层的约束
    # - action_type_embed: nn.Embedding(11, 32) → 索引 0-10
    # - action_param_embed: nn.Embedding(35, 32) → 索引 0-34
    # - player_embed: nn.Embedding(4, 16) → 索引 0-3
    batch_size = 4
    obs = {
        # 全局手牌 [batch, 136] (4 players * 34 tiles)
        "global_hand": torch.randn(batch_size, 136).to(device),
        # 私有手牌 [batch, 34]
        "private_hand": torch.randn(batch_size, 34).to(device),
        # 弃牌池 [batch, 34]
        "discard_pool_total": torch.randn(batch_size, 34).to(device),
        # 牌墙 [batch, 82]
        "wall": torch.randn(batch_size, 82).to(device),
        # 副露 [batch, 16], [batch, 256], [batch, 32]
        "melds": {
            "action_types": torch.randint(0, 11, (batch_size, 16)).to(device),  # 修复：包含 PASS action (value 10)
            "tiles": torch.randint(0, 2, (batch_size, 256)).float().to(device),
            "group_indices": torch.randint(0, 4, (batch_size, 32)).to(device),
        },
        # 动作历史 [batch, 80] each
        "action_history": {
            "types": torch.randint(0, 11, (batch_size, 80)).to(device),  # 修复：匹配 ActionType 范围 0-10
            "params": torch.randint(0, 35, (batch_size, 80)).to(device),
            "players": torch.randint(0, 4, (batch_size, 80)).to(device),
        },
        # 特殊杠 [batch, 12] (4 players × 3 types: pi_gang[0-7], lai_gang[0-3], zhong_gang[0-4])
        "special_gangs": torch.cat([
            torch.randint(0, 8, (batch_size, 1), device=device),   # p0: pi_gang 0-7
            torch.randint(0, 4, (batch_size, 1), device=device),   # p0: lai_gang 0-3
            torch.randint(0, 5, (batch_size, 1), device=device),   # p0: zhong_gang 0-4
            torch.randint(0, 8, (batch_size, 1), device=device),   # p1: pi_gang
            torch.randint(0, 4, (batch_size, 1), device=device),   # p1: lai_gang
            torch.randint(0, 5, (batch_size, 1), device=device),   # p1: zhong_gang
            torch.randint(0, 8, (batch_size, 1), device=device),   # p2: pi_gang
            torch.randint(0, 4, (batch_size, 1), device=device),   # p2: lai_gang
            torch.randint(0, 5, (batch_size, 1), device=device),   # p2: zhong_gang
            torch.randint(0, 8, (batch_size, 1), device=device),   # p3: pi_gang
            torch.randint(0, 4, (batch_size, 1), device=device),   # p3: lai_gang
            torch.randint(0, 5, (batch_size, 1), device=device),   # p3: zhong_gang
        ], dim=1).to(device),
        # 当前玩家 [batch, 1]
        "current_player": torch.randint(0, 4, (batch_size, 1)).float().to(device),
        # 番数 [batch, 4]
        "fan_counts": torch.randint(0, 600, (batch_size, 4)).to(device),  # 修复：匹配实际最大值 599
        # 特殊指示器 [batch, 2]
        "special_indicators": torch.randint(0, 34, (batch_size, 2)).to(device),
        # 剩余牌数 [batch, 1]
        "remaining_tiles": torch.randint(0, 136, (batch_size, 1)).float().to(device),
        # 庄家 [batch, 1]
        "dealer": torch.randint(0, 4, (batch_size, 1)).float().to(device),
        # 当前阶段 [batch, 1]
        "current_phase": torch.randint(0, 14, (batch_size, 1)).float().to(device),  # 修复：匹配 14 个游戏状态 (0-13)
    }
    action_mask = torch.ones(batch_size, 145).to(device)

    profiler = PerformanceProfiler()

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = actor_net.get_action_and_value(obs, action_mask)

    # 测试
    num_iterations = 100
    with torch.no_grad():
        for _ in range(num_iterations):
            with profiler.profile("Actor前向传播"):
                _ = actor_net.get_action_and_value(obs, action_mask)

    profiler.report()


if __name__ == "__main__":
    print("=" * 80)
    print("训练性能分析工具")
    print("=" * 80)

    # 1. 基础性能分析
    profile_single_episode()

    # 2. cProfile 详细分析
    profile_with_cprofile()

    # 3. 观测构建分析
    profile_observation_building()

    # 4. 网络前向传播分析
    profile_network_forward()

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
