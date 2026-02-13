"""
NFSP 训练器

完整的训练循环，包含：
1. 自对弈（Self-Play）
2. 对手策略切换（前期随机 → 后期历史版本）
3. 评估（与随机策略对战）
4. 模型保存和日志记录
"""

import os
import time
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import deque
import random
from tqdm import tqdm

from example_mahjong_env import WuhanMahjongEnv
from src.drl.agent import NFSPAgentPool, NFSPAgentWrapper, RandomOpponent, PolicyPoolManager
from src.drl.config import Config, get_default_config, get_quick_test_config
from src.drl.curriculum import CurriculumScheduler


class NFSPTrainer:
    """
    NFSP 训练器

    管理完整的训练流程：
    - 500万局自对弈
    - 前期（100万局）使用随机对手
    - 后期（400万局）使用历史版本的平均策略网络
    - 每1000局评估一次（与随机策略对战）
    - 每10000局保存一次模型
    """


    def close(self):
        """关闭训练器和环境"""
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
        if hasattr(self, 'vec_env') and self.vec_env is not None:
            self.vec_env.close()

    def __init__(
        self,
        config: Optional[Config] = None,
        device: str = "cuda",
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        checkpoint_path: Optional[str] = None,
        use_vectorized_env: bool = False,
        num_envs: int = 4,
    ):
        """
        初始化训练器

        Args:
            config: 配置对象
            device: 计算设备
            log_dir: 日志目录
            checkpoint_dir: 检查点目录
            checkpoint_path: 检查点路径（用于恢复训练）
        """
        self.config = config or get_default_config()
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.use_vectorized_env = use_vectorized_env
        self.num_envs = num_envs

        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 创建环境（单个或向量化）
        if use_vectorized_env:
            # 使用向量化环境
            from src.drl.vec_env import make_vec_env, EnvFactory

            env_factory = EnvFactory(
                env_type="WuhanMahjongEnv",
                render_mode=None,
                training_phase=self.config.mahjong.training_phase,
                enable_logging=False,
                fast_mode=True,  # 显式启用 fast_mode（禁用快照，提升性能）
            )
            self.vec_env = make_vec_env(
                env_factory, num_envs=num_envs, use_subprocess=False
            )
            self.env = None  # 向量化模式下不使用单环境
        else:
            # 使用单个环境
            self.env = WuhanMahjongEnv(
                render_mode=None,
                training_phase=self.config.mahjong.training_phase,
                enable_logging=False,
                fast_mode=True,  # 显式启用 fast_mode（禁用快照，提升性能）
            )
            self.vec_env = None

        # 创建 NFSP 智能体池（参数共享）
        self.agent_pool = NFSPAgentPool(
            config=self.config, device=device, num_agents=4, share_parameters=True
        )

        # 随机对手（用于前期训练和评估）
        self.random_opponent = RandomOpponent()

        # 策略池管理器（用于后期自对弈）
        self.policy_pool_manager = PolicyPoolManager(
            pool_size=self.config.training.policy_pool_size,
            device=device
        )

        # 探索期配置
        self.exploration_episodes = self.config.training.exploration_episodes

        # 课程学习调度器
        self.curriculum = CurriculumScheduler(
            total_episodes=self.config.training.actual_total_episodes
        )

        # 课程学习状态
        self.current_phase = 1
        self.current_progress = 0.0

        # 训练统计
        self.episode_count = 0
        self.start_time = time.time()
        self.eval_results = []

        # 日志
        self.log_file = os.path.join(log_dir, "training_log.jsonl")

        # 设置随机种子
        if self.config.training.seed is not None:
            self._set_seed(self.config.training.seed)

        # 从检查点恢复（如果提供）
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def _set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def train(self):
        """
        主训练循环

        训练规模根据模式决定：
        - quick_test: 10万局
        - full_training: 2000万局
        """
        print("=" * 80)
        print("NFSP 训练开始")
        print(f"训练模式: {self.config.training.mode}")
        print(f"总训练局数: {self.config.training.actual_total_episodes:,}")
        print(f"切换点: {self.config.training.switch_point:,} 局")
        print(f"评估间隔: 每 {self.config.training.eval_interval:,} 局")
        print(f"保存间隔: 每 {self.config.training.actual_save_interval:,} 局")
        print(f"设备: {self.device}")
        print("=" * 80)

        try:
            # 创建进度条
            pbar = tqdm(
                total=self.config.training.actual_total_episodes,
                desc="Training",
                unit="ep",
                ncols=100,
                initial=self.episode_count,
            )

            while self.episode_count < self.config.training.actual_total_episodes:
                # 更新课程学习阶段
                phase, progress = self.curriculum.get_phase(self.episode_count)
                self.current_phase = phase
                self.current_progress = progress

                # 更新环境训练阶段
                if self.env is not None:
                    self.env.training_phase = phase
                    self.env.training_progress = progress
                if self.vec_env is not None:
                    for env in self.vec_env.envs:
                        env.training_phase = phase
                        env.training_progress = progress

                # 更新进度条描述
                pbar.set_description(f"Ep {self.episode_count:,} | Phase {phase} | Progress {progress:.1%}")

                # 根据配置选择训练方式
                if self.use_vectorized_env:
                    # 向量化模式：运行一批回合
                    episode_stats_list = self._run_episode_vectorized()
                    # 使用第一个环境的统计作为代表
                    episode_stats = episode_stats_list[0] if episode_stats_list else {}
                    # 更新进度条（已递增 num_envs 次）
                    self.episode_count += len(episode_stats_list)
                    pbar.update(len(episode_stats_list))
                else:
                    # 单环境模式：运行一局
                    episode_stats = self._run_episode()
                    self.episode_count += 1
                    pbar.update(1)

                # 训练（传递当前训练阶段）
                train_stats = self.agent_pool.train_all(
                    training_phase=self.current_phase
                )

                # 定期评估
                if self.episode_count % self.config.training.eval_interval == 0:
                    eval_stats = self._evaluate()
                    self._log_stats(episode_stats, train_stats, eval_stats)
                    # 更新进度条后缀显示统计信息
                    if train_stats and "policy_loss" in train_stats:
                        pbar.set_postfix({
                            "loss": f"{train_stats.get('policy_loss', 0):.4f}",
                            "win_rate": f"{eval_stats.get('win_rate', 0):.1%}" if eval_stats else "N/A"
                        })

                # 定期保存
                if self.episode_count % self.config.training.actual_save_interval == 0:
                    self._save_checkpoint()
                    self._add_to_policy_pool()

            pbar.close()

        except KeyboardInterrupt:
            print("\n训练被中断")
            if 'pbar' in locals():
                pbar.close()
        finally:
            # 保存最终模型
            self._save_checkpoint(is_final=True)
            self._save_training_summary()
            print("\n训练完成！")

    def _run_episode(self) -> Dict:
        """
        运行一局自对弈（PettingZoo 标准模式）

        Returns:
            回合统计信息
        """
        # 重置环境
        obs, _ = self.env.reset()

        # 确定对手类型
        if self.episode_count < self.exploration_episodes:
            # 探索期：使用随机对手
            use_random_opponents = True
        else:
            # 自对弈期：使用策略池中的历史策略
            use_random_opponents = False

        # [NEW] 自对弈期：为每个位置预选对手
        episode_opponents = [None] * 4  # 4 个位置的对手
        if not use_random_opponents:
            for i in range(4):
                opponent = self.policy_pool_manager.sample_opponent()
                episode_opponents[i] = opponent  # None 表示池为空，使用当前策略

        # [NEW] Step-by-step data collection for centralized buffer
        # 每个时间步存储4个agents的数据
        step_data = []  # List of dicts, each containing data for one agent action
        agent_turn_count = 0  # Track how many agents have acted in current step

        # 回合数据
        episode_rewards = [0.0] * 4
        episode_steps = 0
        winner = None

        # 用于收集最终观测（诊断用）
        all_agents_observations = {}

        # PettingZoo 标准循环：使用 agent_iter() 和 last()
        for agent_name in self.env.agent_iter():
            episode_steps += 1
            agent_turn_count += 1

            # 关键：使用 env.last() 获取所有信息（obs, reward, terminated, truncated, info）
            obs, reward, terminated, truncated, info = self.env.last()
            agent_idx = int(agent_name.split("_")[1])

            all_agents_observations[agent_name] = obs

            # 从观测字典中获取 action_mask
            action_mask = obs["action_mask"]

            # 记录奖励
            episode_rewards[agent_idx] += reward

            # 选择动作
            if use_random_opponents:
                # 探索期：使用随机对手
                action_type, action_param = self.random_opponent.choose_action(
                    obs, action_mask
                )
                log_prob, value = 0.0, 0.0
            else:
                # 自对弈期：使用预选的对手
                opponent = episode_opponents[agent_idx]
                if opponent is None:
                    # 策略池为空，降级使用当前策略
                    agent = self.agent_pool.get_agent(agent_idx)
                    action_type, action_param = agent.choose_action(obs, action_mask)
                    if hasattr(agent, 'get_training_info'):
                        log_prob, value = agent.get_training_info()
                    else:
                        log_prob, value = 0.0, 0.0
                else:
                    # 使用历史策略对手
                    action_type, action_param = opponent.choose_action(
                        obs, action_mask, eta=self.config.nfsp.eta
                    )
                    log_prob, value = 0.0, 0.0  # 历史策略不提供训练信息

            # [NEW] 收集step数据（用于集中式critic训练）
            # Deep copy obs以避免引用问题
            import copy
            obs_copy = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
            action_mask_copy = action_mask.copy()

            step_data.append({
                'agent_idx': agent_idx,
                'obs': obs_copy,
                'action_mask': action_mask_copy,
                'action_type': action_type,
                'action_param': action_param,
                'log_prob': log_prob,
                'reward': reward,
                'value': value,
                'done': terminated or truncated,
            })

            # 执行动作（不使用返回值，下一次 last() 会提供新信息）
            self.env.step((action_type, action_param))

            # 检查游戏结束
            if terminated or truncated:
                # 从 info 中获取赢家信息
                if "winner" in info:
                    winner = info["winner"]
                elif "winners" in info and info["winners"]:
                    winner = info["winners"][0]

        # 返回回合统计
        episode_stats = {
            "rewards": episode_rewards,
            "steps": episode_steps,
            "winner": winner,
            "use_random_opponents": use_random_opponents,
            "curriculum_phase": self.current_phase,
            "curriculum_progress": self.current_progress,
        }

        # [NEW] 填充 CentralizedRolloutBuffer（用于 Phase 1-2）
        # Phase 1-2: 需要所有agents的观测用于 centralized critic
        if self.current_phase in [1, 2] and len(step_data) > 0:
            self._populate_centralized_buffer_from_steps(step_data)

        return episode_stats

    # ==================== 向量化训练辅助方法 ====================

    def _init_env_state(self, env_idx: int, obs: Dict) -> Dict:
        """
        初始化单个环境的状态字典

        Args:
            env_idx: 环境索引
            obs: 初始观测

        Returns:
            环境状态字典
        """
        return {
            'env_idx': env_idx,
            'current_agent': 0,  # 总是从 agent_0 开始
            'current_obs': obs,
            'action_mask': obs['action_mask'],
            'step_data': [],
            'rewards': [0.0] * 4,
            'step_count': 0,
            'done': False,
        }

    def _group_by_current_agent(self, env_states: List[Dict], active_envs: List[int]) -> Dict[int, List[int]]:
        """
        按当前智能体 ID 分组活跃环境

        Args:
            env_states: 所有环境的状态列表
            active_envs: 活跃环境的索引列表

        Returns:
            {agent_id: [env_indices]} - 例如 {0: [0, 3], 1: [1], 2: [2]}
        """
        groups = {0: [], 1: [], 2: [], 3: []}
        for idx in active_envs:
            agent_id = env_states[idx]['current_agent']
            groups[agent_id].append(idx)
        return groups

    def _execute_single_step(
        self,
        env_states: List[Dict],
        env_idx: int,
        action_type: int,
        action_param: int,
        log_prob: float = 0.0,
        value: float = 0.0,
    ):
        """
        执行单个环境的步骤并更新状态

        Args:
            env_states: 所有环境的状态列表
            env_idx: 要执行的环境索引
            action_type: 动作类型
            action_param: 动作参数
            log_prob: 动作对数概率
            value: 价值估计
        """
        import copy
        state = env_states[env_idx]

        # 记录当前步骤数据
        obs_copy = {k: copy.deepcopy(v) if isinstance(v, dict) else v
                    for k, v in state['current_obs'].items()}
        action_mask_copy = state['action_mask'].copy()

        state['step_data'].append({
            'agent_idx': state['current_agent'],
            'obs': obs_copy,
            'action_mask': action_mask_copy,
            'action_type': action_type,
            'action_param': action_param,
            'log_prob': log_prob,
            'value': value,
            'reward': 0.0,  # 稍后更新
            'done': False,   # 稍后更新
        })

        # 执行动作（支持单环境和向量化模式）
        if self.use_vectorized_env:
            # 向量化模式：使用 vec_env.step(env_idx, action)
            obs, reward, terminated, truncated, info = self.vec_env.step(env_idx, (action_type, action_param))
        else:
            # 单环境模式：使用 env.step(action)
            obs, reward, terminated, truncated, info = self.env.step((action_type, action_param))

        # 更新奖励
        state['rewards'][state['current_agent']] += reward
        if state['step_data']:
            state['step_data'][-1]['reward'] = reward

        # 更新状态
        state['step_count'] += 1
        state['done'] = terminated or truncated

        if not state['done']:
            # 更新到下一个智能体
            state['current_agent'] = (state['current_agent'] + 1) % 4
            state['current_obs'] = obs
            state['action_mask'] = obs['action_mask']

    def _step_agent_batch(
        self,
        env_states: List[Dict],
        env_indices: List[int],
        agent_id: int,
    ):
        """
        对一组环境批量选择动作并执行

        Args:
            env_states: 所有环境的状态
            env_indices: 需要处理的环境索引列表（都是 agent_id 的回合）
            agent_id: 当前智能体 ID
        """
        if not env_indices:
            return

        # 1. 收集观测和掩码
        obs_batch = [env_states[idx]['current_obs'] for idx in env_indices]
        mask_batch = [env_states[idx]['action_mask'] for idx in env_indices]

        # 2. 确定对手类型（探索期 vs 自对弈期）
        use_random_opponents = self.episode_count < self.exploration_episodes

        # 3. 批量选择动作
        if use_random_opponents:
            # 探索期：使用随机对手（逐个处理，无批量优化）
            for idx in env_indices:
                action_type, action_param = self.random_opponent.choose_action(
                    env_states[idx]['current_obs'],
                    env_states[idx]['action_mask']
                )
                self._execute_single_step(env_states, idx, action_type, action_param)
        else:
            # 自对弈期：使用 NFSP 批量推理
            actions_type, actions_param, log_probs, values = self.agent_pool.select_actions_batch(
                obs_batch, mask_batch
            )

            # 4. 执行步骤
            for i, idx in enumerate(env_indices):
                self._execute_single_step(
                    env_states, idx,
                    actions_type[i], actions_param[i],
                    log_probs[i], values[i]
                )

    def _finalize_episode(self, env_state: Dict) -> Dict:
        """
        完成一个环境的回合，收集数据

        Args:
            env_state: 环境状态字典
        """
        episode_stats = {
            'rewards': env_state['rewards'],
            'steps': env_state['step_count'],
            'winner': None,  # 可以从最后一个 step 的 info 获取
            'use_random_opponents': self.episode_count < self.exploration_episodes,
            'curriculum_phase': self.current_phase,
            'curriculum_progress': self.current_progress,
        }

        # 填充 centralized buffer（Phase 1-2）
        if self.current_phase in [1, 2] and len(env_state['step_data']) > 0:
            self._populate_centralized_buffer_from_steps(
                env_state['step_data'], env_state['env_idx']
            )

        return episode_stats

    def _run_episode_vectorized(self) -> List[Dict]:
        """
        运行一批向量化回合

        Returns:
            每个环境的回合统计列表
        """
        # 1. 重置所有环境
        observations = self.vec_env.reset()
        env_states = [self._init_env_state(i, obs) for i, obs in enumerate(observations)]

        # 2. 确定对手类型
        use_random_opponents = self.episode_count < self.exploration_episodes

        # 3. 预选自对弈期的对手（每个位置一个）
        episode_opponents = [None] * 4
        if not use_random_opponents:
            for i in range(4):
                episode_opponents[i] = self.policy_pool_manager.sample_opponent()

        # 4. 运行直到所有环境完成
        active_envs = list(range(self.num_envs))
        max_steps = 1000  # 防止无限循环
        step_count = 0

        while active_envs and step_count < max_steps:
            # 按智能体分组
            agent_groups = self._group_by_current_agent(env_states, active_envs)

            # 对每个分组批量执行
            for agent_id, env_indices in agent_groups.items():
                if env_indices:
                    self._step_agent_batch(env_states, env_indices, agent_id)

            # 检查完成的环境
            for idx in active_envs[:]:
                if env_states[idx]['done']:
                    active_envs.remove(idx)

            step_count += 1

        # 5. 返回所有环境的统计
        return [self._finalize_episode(s) for s in env_states]

    def _populate_centralized_buffer_from_steps(self, step_data: List[Dict], env_idx: int = None):
        """
        从收集的step数据填充CentralizedRolloutBuffer

        Args:
            step_data: List of dicts, each containing data for one agent action
                      Keys: agent_idx, obs, action_mask, action_type, action_param,
                            log_prob, reward, value, done
            env_idx: 环境索引（向量化模式下使用）
        """
        import numpy as np

        # 计算时间步数量（每次完整轮转4个agents）
        num_steps = len(step_data) // 4

        # 获取全局观测（用于添加 global_hand 字段）
        # global_hand 包含所有 4 个 agents 的手牌，用于 centralized critic
        global_obs = None
        try:
            # 根据模式选择环境
            if self.use_vectorized_env and env_idx is not None:
                env = self.vec_env.envs[env_idx]
            else:
                env = self.env
            context = env.unwrapped.context
            obs_builder = env.unwrapped.state_machine.observation_builder
            global_obs = obs_builder.build_global_observation(context, training_phase=self.current_phase)
        except Exception as e:
            # 如果无法获取全局观测，记录警告但继续
            if self.episode_count == 0:
                print(f"  [警告] 无法获取全局观测: {e}")

        # 构建 global_hand: [136] = 4 agents × 34 tile types
        # 从 player_i_hand [14, 34] one-hot 编码转换为 [34] count 编码
        global_hand = None
        if global_obs is not None:
            try:
                agent_hands = []
                for i in range(4):
                    hand_key = f"player_{i}_hand"
                    if hand_key in global_obs:
                        # [14, 34] one-hot -> [34] count (每张牌有多少个)
                        hand_onehot = global_obs[hand_key]  # [14, 34]
                        hand_count = hand_onehot.sum(axis=0)  # [34]
                        agent_hands.append(hand_count)
                    else:
                        # 如果没有手牌数据，使用零向量
                        agent_hands.append(np.zeros(34, dtype=np.float32))

                # 拼接成 [136]
                global_hand = np.concatenate(agent_hands)  # [136]
            except Exception as e:
                if self.episode_count == 0:
                    print(f"  [警告] 构建 global_hand 失败: {e}")
                global_hand = None

        # 为每个时间步收集4个agents的数据
        for step_idx in range(num_steps):
            step_start = step_idx * 4
            step_agents = step_data[step_start:step_start + 4]

            # 按agent_idx排序（确保顺序正确）
            step_agents_sorted = sorted(step_agents, key=lambda x: x['agent_idx'])

            # 提取各agents的数据
            all_obs = []
            for a in step_agents_sorted:
                obs = a['obs'].copy()  # 复制以避免修改原始数据
                # 添加 global_hand 字段
                if global_hand is not None:
                    obs['global_hand'] = global_hand.copy()
                all_obs.append(obs)

            all_action_masks = [a['action_mask'] for a in step_agents_sorted]
            all_actions_type = [a['action_type'] for a in step_agents_sorted]
            all_actions_param = [a['action_param'] for a in step_agents_sorted]
            all_log_probs = [a['log_prob'] for a in step_agents_sorted]
            all_rewards = [a['reward'] for a in step_agents_sorted]
            all_values = [a['value'] if a['value'] is not None else 0.0 for a in step_agents_sorted]
            all_dones = [a['done'] for a in step_agents_sorted]

            # 使用最后一个agent的done标志
            done = all_dones[-1] if all_dones else False

            # 使用add_multi_agent添加数据
            self.agent_pool.centralized_buffer.add_multi_agent(
                all_observations=all_obs,
                action_masks=all_action_masks,
                actions_type=all_actions_type,
                actions_param=all_actions_param,
                log_probs=all_log_probs,
                rewards=all_rewards,
                values=all_values,
                done=done,
            )

        if self.episode_count == 0:
            print(f"  填充 centralized_buffer: {num_steps} 时间步")
            if global_hand is not None:
                print(f"  [成功] 已添加 global_hand 字段到观测中")
            else:
                print(f"  [警告] 未添加 global_hand 字段")

        # After populating, call finish_episode to package the data
        self.agent_pool.centralized_buffer.finish_episode()


    def _evaluate(self) -> Dict:
        """
        评估当前策略 vs 随机对手

        Returns:
            评估统计信息
        """
        num_games = self.config.training.eval_games
        wins = 0
        total_rewards = [0.0] * 4

        for game_idx in range(num_games):
            obs, _ = self.env.reset()
            game_rewards = [0.0] * 4
            final_info = None

            # PettingZoo 标准循环
            for agent_name in self.env.agent_iter():
                obs, reward, terminated, truncated, info = self.env.last()
                final_info = info
                agent_idx = int(agent_name.split("_")[1])
                action_mask = obs["action_mask"]

                game_rewards[agent_idx] += reward

                # Agent 0 使用 NFSP，其他使用随机对手
                if agent_idx == 0:
                    agent = self.agent_pool.get_agent(0)
                    action_type, action_param = agent.choose_action(obs, action_mask)
                else:
                    action_type, action_param = self.random_opponent.choose_action(
                        obs, action_mask
                    )

                self.env.step((action_type, action_param))

            # 累计奖励
            for i in range(4):
                total_rewards[i] += game_rewards[i]

            # 检查赢家
            if final_info and "winner" in final_info and final_info["winner"] == 0:
                wins += 1

        win_rate = wins / num_games
        avg_reward = total_rewards[0] / num_games

        eval_stats = {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "wins": wins,
            "games": num_games,
        }

        self.eval_results.append(eval_stats)
        return eval_stats

    def _log_stats(self, episode_stats: Dict, train_stats: Dict, eval_stats: Dict):
        """记录统计信息"""
        log_entry = {
            "episode": self.episode_count,
            "timestamp": time.time() - self.start_time,
            "episode_stats": episode_stats,
            "train_stats": train_stats,
            "eval_stats": eval_stats,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _print_progress(self, train_stats: Dict, eval_stats: Dict):
        """打印训练进度"""
        elapsed = time.time() - self.start_time
        eps_per_sec = self.episode_count / elapsed if elapsed > 0 else 0
        remaining_eps = self.config.training.actual_total_episodes - self.episode_count
        eta_seconds = remaining_eps / eps_per_sec if eps_per_sec > 0 else 0

        print(
            f"\nEpisode {self.episode_count:,}/{self.config.training.actual_total_episodes:,}"
        )

        # 课程学习阶段
        phase_names = {1: "全知", 2: "渐进掩码", 3: "真实"}
        print(
            f"  课程学习: 阶段 {self.current_phase} ({phase_names.get(self.current_phase, '未知')})"
        )
        if self.current_phase == 2:
            print(f"  渐进进度: {self.current_progress:.1%}")

        print(f"  Win Rate vs Random: {eval_stats['win_rate']:.2%}")
        print(f"  Avg Reward: {eval_stats['avg_reward']:.3f}")
        print(f"  Speed: {eps_per_sec:.1f} eps/sec")
        print(f"  ETA: {eta_seconds / 3600:.1f} hours")

        if train_stats:
            print(f"  RL Loss: {train_stats.get('loss', 0):.4f}")
            print(f"  SL Loss: {train_stats.get('sl_loss', 0):.4f}")

    def _save_checkpoint(self, is_final: bool = False):
        """保存检查点"""
        if is_final:
            path = os.path.join(self.checkpoint_dir, "final_model.pth")
        else:
            path = os.path.join(
                self.checkpoint_dir, f"checkpoint_{self.episode_count}.pth"
            )

        self.agent_pool.save(path)
        print(f"  Saved checkpoint: {path}")

        # 保存元数据到单独文件
        metadata = {
            "episode": self.episode_count,
            "phase": self.current_phase,
            "progress": self.current_progress,
            "mode": self.config.training.mode,
            "total_episodes": self.config.training.actual_total_episodes,
            "timestamp": time.time(),
        }
        metadata_path = path.replace(".pth", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata: {metadata_path}")

    def _add_to_policy_pool(self):
        """将当前策略添加到历史池"""
        # 保存当前策略
        policy_path = os.path.join(
            self.checkpoint_dir, f"policy_{self.episode_count}.pth"
        )
        self.agent_pool.save(policy_path)

        # 添加到策略池管理器（会自动加载到内存并管理大小）
        success = self.policy_pool_manager.add_policy(policy_path)

        if success:
            pool_size = self.policy_pool_manager.size()
            print(f"  [Policy Pool] Added policy at episode {self.episode_count}, "
                  f"pool size: {pool_size}/{self.policy_pool_manager.pool_size}")

    def _save_training_summary(self):
        """保存训练总结"""
        summary = {
            "total_episodes": self.episode_count,
            "total_time": time.time() - self.start_time,
            "final_eval": self.eval_results[-1] if self.eval_results else None,
            "best_win_rate": max([r["win_rate"] for r in self.eval_results])
            if self.eval_results
            else 0,
            "config": {
                "total_episodes": self.config.training.total_episodes,
                "switch_point": self.config.training.switch_point,
                "eta": self.config.nfsp.eta,
                "hidden_dim": self.config.network.hidden_dim,
                "transformer_layers": self.config.network.transformer_layers,
            },
        }

        summary_path = os.path.join(self.log_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Training summary saved: {summary_path}")

    def load_checkpoint(self, path: str):
        """
        加载检查点并恢复训练状态

        Args:
            path: 检查点文件路径（.pth）
        """
        # 加载模型权重
        self.agent_pool.load(path)
        print(f"Loaded checkpoint: {path}")

        # 尝试加载元数据
        metadata_path = path.replace(".pth", "_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # 恢复训练状态
            self.episode_count = metadata.get("episode", 0)
            self.current_phase = metadata.get("phase", 1)
            self.current_progress = metadata.get("progress", 0.0)

            # 更新课程学习状态
            phase, progress = self.curriculum.get_phase(self.episode_count)
            self.current_phase = phase
            self.current_progress = progress

            print(f"  Restored training state:")
            print(f"    Episode: {self.episode_count:,}")
            print(f"    Phase: {self.current_phase}")
            print(f"    Progress: {self.current_progress:.2%}")
        else:
            print("  Warning: No metadata found, using episode_count=0")


def train_nfsp(
    config: Optional[Config] = None,
    device: str = "cuda",
    quick_test: bool = False,
    checkpoint_path: Optional[str] = None,
    use_vectorized_env: bool = False,
    num_envs: int = 4,
):
    """
    训练 NFSP 智能体的便捷函数

    Args:
        config: 配置对象（None 则使用默认配置）
        device: 计算设备
        quick_test: 是否使用快速测试配置
        checkpoint_path: 检查点路径（用于恢复训练）
        use_vectorized_env: 是否使用向量化环境
        num_envs: 向量化环境数量
    """
    if quick_test:
        config = get_quick_test_config()
        print("Using quick test config")
    elif config is None:
        config = get_default_config()

    # 检查 CUDA
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # 创建训练器
    trainer = NFSPTrainer(
        config=config,
        device=device,
        checkpoint_path=checkpoint_path,
        use_vectorized_env=use_vectorized_env,
        num_envs=num_envs,
    )

    # 开始训练
    trainer.train()

    return trainer


if __name__ == "__main__":
    # 快速测试
    train_nfsp(quick_test=True)
