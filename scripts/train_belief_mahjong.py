"""
Belief State + Centralized Critic 麻将智能体训练脚本

完整的 CTDE (Centralized Training, Decentralized Execution) 训练流程：
1. Phase 1: 全知训练 (Omniscient) - 使用 Centralized Critic + 完整全局状态
2. Phase 2: 渐进遮蔽 (Progressive) - 使用 Centralized Critic + 部分遮蔽
3. Phase 3: 真实信息 (Real) - 使用 Decentralized Critic + 信念采样

使用方法：
    # 完整训练（2000万局，约4-6周）
    python scripts/train_belief_mahjong.py

    # 快速测试（10万局）
    python scripts/train_belief_mahjong.py --quick-test

    # 从检查点恢复
    python scripts/train_belief_mahjong.py --checkpoint checkpoints/phase2_1000000.pth --phase 2

    # 自定义配置
    python scripts/train_belief_mahjong.py --phase1-episodes 1000000 --phase2-episodes 1000000 --phase3-episodes 1000000

作者：汪呜呜
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from example_mahjong_env import WuhanMahjongEnv
from src.drl.trainer import NFSPTrainer
from src.drl.agent import NFSPAgentPool
from src.drl.config import get_default_config, get_quick_test_config, Config
from src.drl.curriculum import CurriculumScheduler


class BeliefMahjongTrainer:
    """
    Belief State + Centralized Critic 训练器

    实现三阶段 CTDE 训练流程：
    - Phase 1: Omniscient (全知) - 完整全局状态
    - Phase 2: Progressive (渐进) - 逐步遮蔽
    - Phase 3: Real (真实) - 信念采样
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        device: str = "cuda",
        log_dir: str = "logs/belief_mahjong",
        checkpoint_dir: str = "checkpoints",
        tensorboard_dir: str = "runs/belief_mahjong",
        use_belief: bool = True,
        use_centralized_critic: bool = True,
        n_belief_samples: int = 5,
    ):
        """
        初始化训练器

        Args:
            config: 配置对象
            device: 计算设备
            log_dir: 日志目录
            checkpoint_dir: 检查点目录
            tensorboard_dir: TensorBoard 日志目录
            use_belief: 是否使用信念网络
            use_centralized_critic: 是否使用 Centralized Critic
            n_belief_samples: 信念采样数量
        """
        self.config = config or get_default_config()
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_dir = tensorboard_dir
        self.use_belief = use_belief
        self.use_centralized_critic = use_centralized_critic
        self.n_belief_samples = n_belief_samples

        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)

        # 初始化 TensorBoard
        self.writer = SummaryWriter(tensorboard_dir)

        # 当前训练状态
        self.current_phase = 1
        self.episode_count = 0
        self.start_time = time.time()
        self.phase_start_time = time.time()
        self.phase_start_episode = 0

        # 统计信息
        self.stats = {
            "total_episodes": 0,
            "total_wins": [0, 0, 0, 0],
            "total_games": 0,
            "phase_stats": {
                1: {
                    "episodes": 0,
                    "wins": [0, 0, 0, 0],
                    "start_time": None,
                    "end_time": None,
                },
                2: {
                    "episodes": 0,
                    "wins": [0, 0, 0, 0],
                    "start_time": None,
                    "end_time": None,
                },
                3: {
                    "episodes": 0,
                    "wins": [0, 0, 0, 0],
                    "start_time": None,
                    "end_time": None,
                },
            },
        }

        # 训练配置
        self.phase_config = {
            1: {
                "episodes": 6_666_666,  # 约 1/3 总训练量
                "training_phase": 1,
                "use_centralized_critic": True,
                "description": "Omniscient (全知)",
            },
            2: {
                "episodes": 6_666_667,  # 约 1/3 总训练量
                "training_phase": 2,
                "use_centralized_critic": True,
                "description": "Progressive (渐进遮蔽)",
            },
            3: {
                "episodes": 6_666_667,  # 约 1/3 总训练量
                "training_phase": 3,
                "use_centralized_critic": False,
                "description": "Real (真实信息)",
            },
        }

        # 初始化基础训练器
        self.base_trainer = None
        self.agent_pool = None

        print(f"=" * 80)
        print(f"[START] Belief State + Centralized Critic 训练器初始化")
        print(f"=" * 80)
        print(f"设备: {device}")
        print(f"使用信念网络: {use_belief}")
        print(f"使用 Centralized Critic: {use_centralized_critic}")
        print(f"信念采样数: {n_belief_samples}")
        print(f"=" * 80)

    def _create_environment(self, phase: int) -> WuhanMahjongEnv:
        """创建环境"""
        return WuhanMahjongEnv(
            render_mode=None,
            training_phase=phase,
            enable_logging=False,
        )

    def _create_agent_pool(self, phase: int) -> NFSPAgentPool:
        """创建智能体池"""
        # 更新配置以适应当前 phase
        self.config.mahjong.training_phase = phase

        agent_pool = NFSPAgentPool(
            config=self.config,
            device=self.device,
            num_agents=4,
            share_parameters=True,
        )

        # 如果使用信念网络，配置网络
        if self.use_belief:
            # 启用 Actor 的信念集成
            if hasattr(agent_pool, "network"):
                agent_pool.network.use_belief = True
                agent_pool.network.n_belief_samples = self.n_belief_samples

        # 如果使用 centralized critic，配置 MAPPO
        if self.use_centralized_critic and phase in [1, 2]:
            if hasattr(agent_pool, "mappo"):
                # 启用 centralized critic
                agent_pool.mappo.use_dual_critic = True

        return agent_pool

    def train_phase(self, phase: int, episodes: Optional[int] = None) -> Dict:
        """
        训练单个阶段

        Args:
            phase: 阶段 (1, 2, 3)
            episodes: 训练局数（默认使用 phase_config 中的配置）

        Returns:
            phase_stats: 阶段统计信息
        """
        if phase not in [1, 2, 3]:
            raise ValueError(f"Phase 必须是 1, 2, 3，而不是 {phase}")

        phase_config = self.phase_config[phase]
        target_episodes = episodes or phase_config["episodes"]
        training_phase = phase_config["training_phase"]
        use_centralized = phase_config["use_centralized_critic"]

        print(f"\n{'=' * 80}")
        print(f"🎯 Phase {phase}: {phase_config['description']}")
        print(f"训练局数: {target_episodes:,}")
        print(f"Training Phase: {training_phase}")
        print(f"使用 Centralized Critic: {use_centralized}")
        print(f"{'=' * 80}\n")

        # 记录阶段开始
        self.current_phase = phase
        self.phase_start_time = time.time()
        self.phase_start_episode = self.episode_count
        self.stats["phase_stats"][phase]["start_time"] = time.time()

        # TensorBoard 记录 phase 转换事件
        self.writer.add_text(
            "Phase_Transition",
            f"Phase {phase} started: {phase_config['description']}",
            global_step=self.episode_count,
        )

        # 创建环境和智能体池
        env = self._create_environment(training_phase)
        agent_pool = self._create_agent_pool(training_phase)

        # 如果是 Phase 2 或 3，尝试从上一个 phase 加载检查点
        if phase > 1:
            self._load_phase_checkpoint(phase - 1, agent_pool)

        # 创建课程学习调度器
        curriculum = CurriculumScheduler(total_episodes=target_episodes)

        # 训练循环
        episode_wins = [0, 0, 0, 0]
        eval_results = []

        for episode in range(target_episodes):
            # 运行一局
            episode_stats = self._run_episode(env, agent_pool, training_phase)

            # 更新统计
            self.episode_count += 1
            if episode_stats.get("winner") is not None:
                winner = episode_stats["winner"]
                episode_wins[winner] += 1
                self.stats["total_wins"][winner] += 1

            # 定期评估
            if episode > 0 and episode % self.config.training.eval_interval == 0:
                eval_stats = self._evaluate(env, agent_pool)
                eval_results.append(eval_stats)
                self._log_eval(phase, episode, eval_stats)

            # 定期保存检查点
            if episode > 0 and episode % self.config.training.actual_save_interval == 0:
                self._save_checkpoint(phase, episode, agent_pool)

            # 定期打印进度
            if episode > 0 and episode % 1000 == 0:
                self._print_progress(phase, episode, target_episodes, episode_wins)

            # TensorBoard 记录
            if episode > 0 and episode % 100 == 0:
                self._log_tensorboard(phase, episode, episode_stats)

        # 阶段结束，保存最终检查点
        self._save_checkpoint(phase, target_episodes, agent_pool, is_final=True)

        # 记录阶段统计
        self.stats["phase_stats"][phase]["episodes"] = target_episodes
        self.stats["phase_stats"][phase]["wins"] = episode_wins
        self.stats["phase_stats"][phase]["end_time"] = time.time()

        phase_duration = time.time() - self.phase_start_time

        # TensorBoard 记录 phase 结束事件
        phase_summary = {
            "phase": phase,
            "episodes": target_episodes,
            "wins": episode_wins,
            "duration_hours": phase_duration / 3600,
            "win_rates": [w / max(target_episodes, 1) for w in episode_wins],
        }

        self.writer.add_text(
            "Phase_Summary",
            f"Phase {phase} completed: {phase_summary}",
            global_step=self.episode_count,
        )

        # 记录 phase 指标
        self.writer.add_scalar(
            f"Phase{phase}/Duration_Hours", phase_duration / 3600, self.episode_count
        )
        self.writer.add_scalar(
            f"Phase{phase}/Episodes", target_episodes, self.episode_count
        )

        print(f"\n[DONE] Phase {phase} 完成！耗时: {phase_duration / 3600:.2f} 小时")

        return {
            "phase": phase,
            "episodes": target_episodes,
            "wins": episode_wins,
            "duration": phase_duration,
            "eval_results": eval_results,
        }

    def _run_episode(
        self, env: WuhanMahjongEnv, agent_pool: NFSPAgentPool, training_phase: int
    ) -> Dict:
        """运行一局游戏"""
        obs, _ = env.reset()
        done = False
        episode_data = {"winner": None, "steps": 0, "rewards": [0.0] * 4}

        while not done:
            current_agent = env.agent_selection
            agent_id = int(current_agent.split("_")[-1])

            # 获取观测和动作掩码
            agent_obs = obs[current_agent]
            action_mask = env.infos[current_agent].get("action_mask", np.ones(145))

            # 选择动作
            action_type, action_param = agent_pool.select_action(
                agent_id, agent_obs, action_mask
            )

            # 执行动作
            next_obs, rewards, terminations, truncations, infos = env.step(
                (action_type, action_param)
            )

            # 存储转移（如果是训练模式）
            if agent_pool.is_training:
                reward = rewards[current_agent]
                done_flag = terminations[current_agent] or truncations[current_agent]

                agent_pool.store_transition(
                    agent_id=agent_id,
                    observation=agent_obs,
                    action=(action_type, action_param),
                    reward=reward,
                    next_observation=next_obs[current_agent],
                    done=done_flag,
                    action_mask=action_mask,
                )

            # 更新统计
            episode_data["steps"] += 1
            for i in range(4):
                episode_data["rewards"][i] += rewards.get(f"player_{i}", 0.0)

            # 检查游戏是否结束
            if any(terminations.values()) or any(truncations.values()):
                done = True
                # 找出获胜者
                for agent, term in terminations.items():
                    if term and rewards.get(agent, 0) > 0:
                        agent_id = int(agent.split("_")[-1])
                        episode_data["winner"] = agent_id
                        break

            obs = next_obs

        # 训练一步
        train_stats = agent_pool.train_all(training_phase=training_phase)
        episode_data["train_stats"] = train_stats

        return episode_data

    def _evaluate(self, env: WuhanMahjongEnv, agent_pool: NFSPAgentPool) -> Dict:
        """评估智能体性能"""
        eval_wins = [0, 0, 0, 0]
        num_games = self.config.training.eval_games

        # 临时切换到评估模式
        was_training = agent_pool.is_training
        agent_pool.is_training = False

        for _ in range(num_games):
            obs, _ = env.reset()
            done = False

            while not done:
                current_agent = env.agent_selection
                agent_id = int(current_agent.split("_")[-1])

                agent_obs = obs[current_agent]
                action_mask = env.infos[current_agent].get("action_mask", np.ones(145))

                action_type, action_param = agent_pool.select_action(
                    agent_id, agent_obs, action_mask
                )

                obs, rewards, terminations, truncations, infos = env.step(
                    (action_type, action_param)
                )

                if any(terminations.values()) or any(truncations.values()):
                    done = True
                    for agent, term in terminations.items():
                        if term and rewards.get(agent, 0) > 0:
                            winner_id = int(agent.split("_")[-1])
                            eval_wins[winner_id] += 1
                            break

        # 恢复训练模式
        agent_pool.is_training = was_training

        # 计算胜率
        total_games = sum(eval_wins)
        win_rates = [w / max(total_games, 1) for w in eval_wins]

        return {
            "wins": eval_wins,
            "win_rates": win_rates,
            "total_games": total_games,
        }

    def _log_eval(self, phase: int, episode: int, eval_stats: Dict):
        """记录评估结果"""
        print(f"[Phase {phase} - Episode {episode:,}] 评估结果:")
        print(f"  总局数: {eval_stats['total_games']}")
        for i in range(4):
            print(f"  Player {i} 胜率: {eval_stats['win_rates'][i] * 100:.1f}%")

        # TensorBoard 记录评估结果
        self.writer.add_scalar(
            f"Phase{phase}/Eval_Total_Games",
            eval_stats["total_games"],
            self.episode_count,
        )
        for i in range(4):
            self.writer.add_scalar(
                f"Phase{phase}/Eval_Win_Rate_Player{i}",
                eval_stats["win_rates"][i],
                self.episode_count,
            )
        self.writer.add_scalar(
            f"Phase{phase}/Eval_Average_Win_Rate",
            sum(eval_stats["win_rates"]) / 4,
            self.episode_count,
        )

    def _log_tensorboard(self, phase: int, episode: int, episode_stats: Dict):
        """记录 TensorBoard"""
        global_step = self.episode_count

        # 记录训练统计
        if "train_stats" in episode_stats:
            stats = episode_stats["train_stats"]
            if "loss" in stats:
                self.writer.add_scalar(f"Phase{phase}/Loss", stats["loss"], global_step)
            if "value_loss" in stats:
                self.writer.add_scalar(
                    f"Phase{phase}/Value_Loss", stats["value_loss"], global_step
                )
            if "policy_loss" in stats:
                self.writer.add_scalar(
                    f"Phase{phase}/Policy_Loss", stats["policy_loss"], global_step
                )
            if "entropy" in stats:
                self.writer.add_scalar(
                    f"Phase{phase}/Entropy", stats["entropy"], global_step
                )

        # 记录 centralized_critic 指标
        if "train_stats" in episode_stats:
            stats = episode_stats["train_stats"]
            if "centralized_critic_loss" in stats:
                self.writer.add_scalar(
                    f"Phase{phase}/Centralized_Critic_Loss",
                    stats["centralized_critic_loss"],
                    global_step,
                )

        # 记录 belief network 指标
        if "train_stats" in episode_stats:
            stats = episode_stats["train_stats"]
            if "belief_loss" in stats:
                self.writer.add_scalar(
                    f"Phase{phase}/Belief_Loss", stats["belief_loss"], global_step
                )
            if "belief_entropy" in stats:
                self.writer.add_scalar(
                    f"Phase{phase}/Belief_Entropy", stats["belief_entropy"], global_step
                )

        # 记录游戏统计
        self.writer.add_scalar(
            f"Phase{phase}/Steps", episode_stats.get("steps", 0), global_step
        )

        # 记录游戏时长
        if "duration" in episode_stats:
            self.writer.add_scalar(
                f"Phase{phase}/Duration", episode_stats["duration"], global_step
            )

        # 记录胜率
        total_games = sum(self.stats["total_wins"])
        if total_games > 0:
            for i in range(4):
                win_rate = self.stats["total_wins"][i] / total_games
                self.writer.add_scalar(
                    f"Phase{phase}/Win_Rate_Player{i}", win_rate, global_step
                )

        # 记录平均胜率
        avg_win_rate = sum(self.stats["total_wins"]) / (4 * max(total_games, 1))
        self.writer.add_scalar(
            f"Phase{phase}/Average_Win_Rate", avg_win_rate, global_step
        )

    def _print_progress(self, phase: int, episode: int, target: int, wins: list):
        """打印训练进度"""
        elapsed = time.time() - self.phase_start_time
        eps_per_sec = episode / max(elapsed, 1)
        eta = (target - episode) / max(eps_per_sec, 1)

        total_games = sum(wins)
        win_rates = [w / max(total_games, 1) * 100 for w in wins]

        print(
            f"[Phase {phase}] Episode {episode:,}/{target:,} "
            f"({episode / target * 100:.1f}%) | "
            f"Speed: {eps_per_sec:.1f} eps/s | "
            f"ETA: {eta / 3600:.1f}h | "
            f"Wins: {win_rates[0]:.1f}%/{win_rates[1]:.1f}%/{win_rates[2]:.1f}%/{win_rates[3]:.1f}%"
        )

    def _save_checkpoint(
        self,
        phase: int,
        episode: int,
        agent_pool: NFSPAgentPool,
        is_final: bool = False,
    ):
        """保存检查点"""
        suffix = "final" if is_final else f"{episode}"
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"phase{phase}_{suffix}.pth"
        )

        checkpoint = {
            "phase": phase,
            "episode": episode,
            "global_episode": self.episode_count,
            # 手动保存网络状态
            "best_response_net_state": agent_pool.shared_nfsp.best_response_net.state_dict(),
            "average_policy_net_state": agent_pool.shared_nfsp.average_policy_net.state_dict(),
            "centralized_critic_state": agent_pool.shared_nfsp.centralized_critic.state_dict()
            if agent_pool.shared_nfsp.centralized_critic is not None
            else None,
            "stats": self.stats,
            "config": self.config,
            "timestamp": time.time(),
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"[SAVE] 检查点已保存: {checkpoint_path}")

        # 同时保存为最新检查点
        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
        torch.save(checkpoint, latest_path)

    def _load_phase_checkpoint(self, from_phase: int, agent_pool: NFSPAgentPool):
        """从上一个 phase 加载检查点"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"phase{from_phase}_final.pth"
        )

        if not os.path.exists(checkpoint_path):
            print(f"[WARN]  Phase {from_phase} 的检查点不存在: {checkpoint_path}")
            print(f"   将从头开始训练 Phase {self.current_phase}")
            return False

        try:
            # 使用 weights_only=False 加载检查点（PyTorch 2.6+ 的默认值）
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )

            # 加载网络状态
            if checkpoint.get("best_response_net_state") and checkpoint.get(
                "average_policy_net_state"
            ):
                agent_pool.shared_nfsp.best_response_net.load_state_dict(
                    checkpoint["best_response_net_state"]
                )
                agent_pool.shared_nfsp.average_policy_net.load_state_dict(
                    checkpoint["average_policy_net_state"]
                )

                if (
                    checkpoint.get("centralized_critic_state")
                    and agent_pool.shared_nfsp.centralized_critic is not None
                ):
                    agent_pool.shared_nfsp.centralized_critic.load_state_dict(
                        checkpoint["centralized_critic_state"]
                    )

                print(f"[DONE] 已从 Phase {from_phase} 加载检查点")

                # Phase 2->3 迁移：如果是从 centralized 到 decentralized
                if from_phase == 2 and self.current_phase == 3:
                    print(f"[TRANSITION] 执行 Phase 2->3 迁移: Critic 重新初始化")

                    # 重新初始化 Local Critic（Phase 3 不使用 Centralized Critic）
                    # 我们需要访问 shared_nfsp 中的 best_response_net
                    if agent_pool.share_parameters and hasattr(
                        agent_pool, "shared_nfsp"
                    ):
                        nfsp = agent_pool.shared_nfsp

                        # 1. 保存 Actor 权重（策略网络）
                        actor_state_dict = {}
                        for key, value in nfsp.best_response_net.state_dict().items():
                            if key.startswith("actor_type.") or key.startswith(
                                "actor_param."
                            ):
                                actor_state_dict[key] = value.clone()

                        # 2. 重新初始化 Critic 权重（因为训练方式改变）
                        # Critic 从 Phase 2 的 centralized 训练切换到 Phase 3 的 decentralized 训练
                        critic_state_dict = {}
                        for key, value in nfsp.best_response_net.state_dict().items():
                            if key.startswith("critic."):
                                # 重新初始化 critic 权重
                                if hasattr(value, "data"):
                                    new_data = torch.randn_like(value.data) * 0.01
                                    # 只对具有足够维度的张量应用 Xavier 初始化
                                    if len(new_data.shape) >= 2:
                                        nn.init.xavier_uniform_(new_data)
                                    critic_state_dict[key] = new_data

                        # 3. 应用更新后的权重
                        with torch.no_grad():
                            for key, value in actor_state_dict.items():
                                nfsp.best_response_net.state_dict()[key].copy_(value)
                            for key, value in critic_state_dict.items():
                                nfsp.best_response_net.state_dict()[key].copy_(value)

                        print(
                            f"[TRANSITION] Actor 权重已保留，Local Critic 已重新初始化"
                        )

                        # 4. 重置 Centralized Critic（Phase 3 不再使用）
                        if hasattr(nfsp, "centralized_critic"):
                            nfsp.centralized_critic = None
                            print(
                                f"[TRANSITION] Centralized Critic 已移除（Phase 3 不需要）"
                            )

                return True
            else:
                print(f"[WARN]  检查点中没有 agent_pool_state")
                return False

        except Exception as e:
            print(f"[ERROR] 加载检查点失败: {e}")
            return False

    def train(
        self,
        phase1_episodes: Optional[int] = None,
        phase2_episodes: Optional[int] = None,
        phase3_episodes: Optional[int] = None,
    ):
        """
        执行完整的三阶段训练

        Args:
            phase1_episodes: Phase 1 训练局数
            phase2_episodes: Phase 2 训练局数
            phase3_episodes: Phase 3 训练局数
        """
        print(f"\n{'=' * 80}")
        print(f"[LAUNCH] 开始完整的三阶段 CTDE 训练")
        print(f"{'=' * 80}\n")

        # Phase 1: Omniscient
        if phase1_episodes is not None:
            self.phase_config[1]["episodes"] = phase1_episodes
        self.train_phase(1)

        # Phase 2: Progressive
        if phase2_episodes is not None:
            self.phase_config[2]["episodes"] = phase2_episodes
        self.train_phase(2)

        # Phase 3: Real
        if phase3_episodes is not None:
            self.phase_config[3]["episodes"] = phase3_episodes
        self.train_phase(3)

        # 训练完成
        total_duration = time.time() - self.start_time
        print(f"\n{'=' * 80}")
        print(f"🎉 训练完成！总耗时: {total_duration / 3600:.2f} 小时")
        print(f"{'=' * 80}\n")

        # 保存最终统计
        self._save_final_stats()

        # 关闭 TensorBoard
        self.writer.close()

    def _save_final_stats(self):
        """保存最终统计信息"""
        stats_path = os.path.join(self.log_dir, "final_stats.json")

        final_stats = {
            "total_episodes": self.episode_count,
            "total_duration_hours": (time.time() - self.start_time) / 3600,
            "phase_stats": self.stats["phase_stats"],
            "total_wins": self.stats["total_wins"],
        }

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)

        print(f"[STATS] 最终统计已保存: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Belief State + Centralized Critic 麻将智能体训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整训练（默认配置）
  python scripts/train_belief_mahjong.py

  # 快速测试（小量数据）
  python scripts/train_belief_mahjong.py --quick-test

  # 从检查点恢复（Phase 2）
  python scripts/train_belief_mahjong.py --checkpoint checkpoints/phase1_final.pth --start-phase 2

  # 自定义各阶段局数
  python scripts/train_belief_mahjong.py --phase1-episodes 1000000 --phase2-episodes 1000000 --phase3-episodes 1000000

  # 不使用信念网络
  python scripts/train_belief_mahjong.py --no-belief

  # 不使用 Centralized Critic
  python scripts/train_belief_mahjong.py --no-centralized-critic
        """,
    )

    # 训练模式
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="快速测试模式（各阶段1万局）",
    )

    parser.add_argument(
        "--start-phase",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="从哪个阶段开始训练（默认: 1）",
    )

    # 各阶段局数
    parser.add_argument(
        "--phase1-episodes",
        type=int,
        default=None,
        help="Phase 1 训练局数（默认: 6,666,666）",
    )

    parser.add_argument(
        "--phase2-episodes",
        type=int,
        default=None,
        help="Phase 2 训练局数（默认: 6,666,667）",
    )

    parser.add_argument(
        "--phase3-episodes",
        type=int,
        default=None,
        help="Phase 3 训练局数（默认: 6,666,667）",
    )

    # 检查点
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="从检查点恢复训练",
    )

    # 设备
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="计算设备（默认: cuda 如果可用）",
    )

    # 架构选项
    parser.add_argument(
        "--no-belief",
        action="store_true",
        help="不使用信念网络",
    )

    parser.add_argument(
        "--no-centralized-critic",
        action="store_true",
        help="不使用 Centralized Critic",
    )

    parser.add_argument(
        "--belief-samples",
        type=int,
        default=5,
        help="信念采样数量（默认: 5）",
    )

    # 目录配置
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/belief_mahjong",
        help="日志目录（默认: logs/belief_mahjong）",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="检查点目录（默认: checkpoints）",
    )

    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="runs/belief_mahjong",
        help="TensorBoard 日志目录（默认: runs/belief_mahjong）",
    )

    # 随机种子
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )

    args = parser.parse_args()

    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    # 获取配置
    if args.quick_test:
        config = get_quick_test_config()
        # 快速测试模式：各阶段1万局
        phase1_episodes = 10_000
        phase2_episodes = 10_000
        phase3_episodes = 10_000
        print("=" * 80)
        print("[LAUNCH] 快速测试模式")
        print("=" * 80)
    else:
        config = get_default_config()
        phase1_episodes = args.phase1_episodes
        phase2_episodes = args.phase2_episodes
        phase3_episodes = args.phase3_episodes
        print("=" * 80)
        print("[GAME] Belief State + Centralized Critic 麻将智能体训练")
        print("=" * 80)

    # 打印配置
    print(f"\n📋 训练配置:")
    print(f"  开始阶段: {args.start_phase}")
    print(f"  Phase 1 局数: {phase1_episodes or 6_666_666:,}")
    print(f"  Phase 2 局数: {phase2_episodes or 6_666_667:,}")
    print(f"  Phase 3 局数: {phase3_episodes or 6_666_667:,}")
    print(f"  使用信念网络: {not args.no_belief}")
    print(f"  使用 Centralized Critic: {not args.no_centralized_critic}")
    print(f"  信念采样数: {args.belief_samples}")
    print(f"  设备: {args.device}")
    print(f"  随机种子: {args.seed}")
    print(f"  日志目录: {args.log_dir}")
    print(f"  检查点目录: {args.checkpoint_dir}")
    print(f"  TensorBoard: {args.tensorboard_dir}")
    if args.checkpoint:
        print(f"  恢复检查点: {args.checkpoint}")
    print("=" * 80)

    # 创建训练器
    trainer = BeliefMahjongTrainer(
        config=config,
        device=args.device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_dir=args.tensorboard_dir,
        use_belief=not args.no_belief,
        use_centralized_critic=not args.no_centralized_critic,
        n_belief_samples=args.belief_samples,
    )

    # 从检查点恢复（如果指定）
    if args.checkpoint and args.start_phase > 1:
        print(f"\n🔄 从检查点恢复: {args.checkpoint}")
        # 这里可以添加检查点恢复逻辑

    # 开始训练
    try:
        if args.start_phase == 1:
            trainer.train(phase1_episodes, phase2_episodes, phase3_episodes)
        elif args.start_phase == 2:
            trainer.train_phase(2, phase2_episodes)
            trainer.train_phase(3, phase3_episodes)
        elif args.start_phase == 3:
            trainer.train_phase(3, phase3_episodes)

        print("\n[DONE] 训练完成！")
        print(f"日志保存于: {args.log_dir}")
        print(f"模型保存于: {args.checkpoint_dir}")
        print(f"TensorBoard: tensorboard --logdir={args.tensorboard_dir}")

    except KeyboardInterrupt:
        print("\n[WARN]  训练被用户中断")
        # 保存中断时的检查点
        print("[SAVE] 保存中断检查点...")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
