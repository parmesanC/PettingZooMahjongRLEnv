"""
NFSP + MAPPO 经验回放缓冲区

包含：
1. RolloutBuffer: 用于 MAPPO 的轨迹缓冲区
2. ReservoirBuffer: 用于 NFSP 监督学习的蓄水池采样缓冲区
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class RolloutBuffer:
    """
    MAPPO 轨迹缓冲区
    存储一步的观测、动作、奖励、价值等信息
    """

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.clear()

    def clear(self):
        """清空缓冲区"""
        self.observations = []
        self.action_masks = []
        self.actions_type = []
        self.actions_param = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.next_observations = []
        self.next_action_masks = []
        self.size = 0

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: np.ndarray,
        action_type: int,
        action_param: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        next_obs: Optional[Dict[str, np.ndarray]] = None,
        next_action_mask: Optional[np.ndarray] = None,
    ):
        """
        添加一条经验

        Args:
            obs: 当前观测
            action_mask: 动作掩码
            action_type: 动作类型
            action_param: 动作参数
            log_prob: 动作对数概率
            reward: 奖励
            value: 价值估计
            done: 是否结束
            next_obs: 下一观测（可选）
            next_action_mask: 下一动作掩码（可选）
        """
        if self.size >= self.capacity:
            # 移除最旧的经验
            self.observations.pop(0)
            self.action_masks.pop(0)
            self.actions_type.pop(0)
            self.actions_param.pop(0)
            self.log_probs.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.dones.pop(0)
            if self.next_observations:
                self.next_observations.pop(0)
                self.next_action_masks.pop(0)

        self.observations.append(obs)
        self.action_masks.append(action_mask)
        self.actions_type.append(action_type)
        self.actions_param.append(action_param)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

        if next_obs is not None:
            self.next_observations.append(next_obs)
            self.next_action_masks.append(next_action_mask)

        self.size = len(self.observations)

    def get_batch(self, batch_size: int, device: str = "cuda") -> Tuple:
        """
        获取一个批次的数据

        Returns:
            批次数据元组
        """
        if self.size < batch_size:
            batch_size = self.size

        # 随机采样
        indices = np.random.choice(self.size, batch_size, replace=False)

        # 收集数据
        batch_obs = [self.observations[i] for i in indices]
        batch_action_masks = np.array([self.action_masks[i] for i in indices])
        batch_actions_type = np.array([self.actions_type[i] for i in indices])
        batch_actions_param = np.array([self.actions_param[i] for i in indices])
        batch_log_probs = np.array([self.log_probs[i] for i in indices])
        batch_rewards = np.array([self.rewards[i] for i in indices])
        batch_values = np.array([self.values[i] for i in indices])
        batch_dones = np.array([self.dones[i] for i in indices])

        # 转换为张量
        batch_action_masks = torch.FloatTensor(batch_action_masks).to(device)
        batch_actions_type = torch.LongTensor(batch_actions_type).to(device)
        batch_actions_param = torch.LongTensor(batch_actions_param).to(device)
        batch_log_probs = torch.FloatTensor(batch_log_probs).to(device)
        batch_rewards = torch.FloatTensor(batch_rewards).to(device)
        batch_values = torch.FloatTensor(batch_values).to(device)
        batch_dones = torch.FloatTensor(batch_dones).to(device)

        return (
            batch_obs,
            batch_action_masks,
            batch_actions_type,
            batch_actions_param,
            batch_log_probs,
            batch_rewards,
            batch_values,
            batch_dones,
        )

    def get_all(self, device: str = "cuda") -> Tuple:
        """获取所有数据"""
        return self.get_batch(self.size, device)

    def compute_returns_and_advantages(
        self, gamma: float = 0.99, gae_lambda: float = 0.95, next_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算回报和 GAE 优势

        Args:
            gamma: 折扣因子
            gae_lambda: GAE lambda
            next_value: 最后一步的下一状态价值

        Returns:
            returns: 回报
            advantages: GAE 优势
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)

        # 从后向前计算
        gae = 0
        next_val = next_value

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_val
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            # TD 误差
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]

            # GAE
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae

            # 回报
            returns[t] = advantages[t] + values[t]

        return returns, advantages

    def __len__(self):
        return self.size


class ReservoirBuffer:
    """
    蓄水池采样缓冲区
    用于 NFSP 的监督学习部分，存储最佳响应网络的行为

    特点：
    - 固定容量，当满时均匀随机替换已有样本
    - 保证每个样本被保留的概率相等
    """

    def __init__(self, capacity: int = 2000000):
        self.capacity = capacity
        self.buffer = []
        self.count = 0  # 总共添加的样本数

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: np.ndarray,
        action_type: int,
        action_param: int,
    ):
        """
        添加一条经验（使用蓄水池采样）

        Args:
            obs: 观测
            action_mask: 动作掩码
            action_type: 动作类型
            action_param: 动作参数
        """
        experience = (obs, action_mask, action_type, action_param)

        if len(self.buffer) < self.capacity:
            # 缓冲区未满，直接添加
            self.buffer.append(experience)
        else:
            # 缓冲区已满，以概率 capacity/count 替换已有样本
            idx = random.randint(0, self.count)
            if idx < self.capacity:
                self.buffer[idx] = experience

        self.count += 1

    def sample(self, batch_size: int, device: str = "cuda") -> Tuple:
        """
        随机采样一个批次

        Returns:
            批次数据元组
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # 随机采样
        samples = random.sample(self.buffer, batch_size)

        # 解包
        obs_list = []
        action_masks = []
        action_types = []
        action_params = []

        for exp in samples:
            obs, mask, a_type, a_param = exp
            obs_list.append(obs)
            action_masks.append(mask)
            action_types.append(a_type)
            action_params.append(a_param)

        # 转换为张量
        action_masks = torch.FloatTensor(np.array(action_masks)).to(device)
        action_types = torch.LongTensor(action_types).to(device)
        action_params = torch.LongTensor(action_params).to(device)

        return obs_list, action_masks, action_types, action_params

    def __len__(self):
        return len(self.buffer)


class EpisodeBuffer:
    """
    回合级缓冲区
    用于收集完整的回合数据
    """

    def __init__(self):
        self.episodes = []
        self.current_episode = {
            "observations": [],
            "action_masks": [],
            "actions_type": [],
            "actions_param": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }

    def add_step(self, **kwargs):
        """添加一步"""
        for key, value in kwargs.items():
            if key in self.current_episode:
                self.current_episode[key].append(value)

    def end_episode(self):
        """结束当前回合"""
        if len(self.current_episode["observations"]) > 0:
            self.episodes.append(self.current_episode)
            self.current_episode = {
                "observations": [],
                "action_masks": [],
                "actions_type": [],
                "actions_param": [],
                "log_probs": [],
                "rewards": [],
                "values": [],
                "dones": [],
            }

    def get_all_episodes(self) -> List[Dict]:
        """获取所有回合"""
        return self.episodes

    def clear(self):
        """清空"""
        self.episodes = []
        self.current_episode = {
            "observations": [],
            "action_masks": [],
            "actions_type": [],
            "actions_param": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }

    def __len__(self):
        return len(self.episodes)


class MixedBuffer:
    """
    混合缓冲区
    同时管理 RL 缓冲区和 SL 缓冲区
    """

    def __init__(self, rl_capacity: int = 100000, sl_capacity: int = 2000000):
        self.rl_buffer = RolloutBuffer(rl_capacity)
        self.sl_buffer = ReservoirBuffer(sl_capacity)

    def add_rl_experience(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: np.ndarray,
        action_type: int,
        action_param: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        next_obs: Optional[Dict[str, np.ndarray]] = None,
        next_action_mask: Optional[np.ndarray] = None,
    ):
        """添加 RL 经验"""
        self.rl_buffer.add(
            obs,
            action_mask,
            action_type,
            action_param,
            log_prob,
            reward,
            value,
            done,
            next_obs,
            next_action_mask,
        )

    def add_sl_experience(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: np.ndarray,
        action_type: int,
        action_param: int,
    ):
        """添加 SL 经验"""
        self.sl_buffer.add(obs, action_mask, action_type, action_param)

    def sample_rl_batch(self, batch_size: int, device: str = "cuda"):
        """采样 RL 批次"""
        return self.rl_buffer.get_batch(batch_size, device)

    def sample_sl_batch(self, batch_size: int, device: str = "cuda"):
        """采样 SL 批次"""
        return self.sl_buffer.sample(batch_size, device)

    def clear_rl(self):
        """清空 RL 缓冲区"""
        self.rl_buffer.clear()

    def clear_sl(self):
        """清空 SL 缓冲区"""
        self.sl_buffer = ReservoirBuffer(self.sl_buffer.capacity)

    def __len__(self):
        return len(self.rl_buffer) + len(self.sl_buffer)


class CentralizedRolloutBuffer:
    """
    Centralized Rollout Buffer 用于 centralized critic 训练
    存储所有4个智能体的观测、动作、奖励等信息
    """

    def __init__(self, capacity: int = 100000):
        """
        初始化 centralized rollout buffer

        Args:
            capacity: 缓冲区容量
        """
        self.capacity = capacity
        self.clear()

    def clear(self):
        """清空缓冲区"""
        # 存储4个agents的数据
        # List中的每个元素是一个episode的数据
        self.episodes = []

        # 当前episode的累积数据
        self.current_obs = [[] for _ in range(4)]  # 4个agents
        self.current_action_masks = [[] for _ in range(4)]
        self.current_actions_type = [[] for _ in range(4)]
        self.current_actions_param = [[] for _ in range(4)]
        self.current_log_probs = [[] for _ in range(4)]
        self.current_rewards = [[] for _ in range(4)]
        self.current_values = [[] for _ in range(4)]  # centralized critic的值
        self.current_dones = [[] for _ in range(4)]
        self.current_all_observations = [
            [] for _ in range(4)
        ]  # 所有agents的观测（用于centralized critic）

        self.size = 0
        self.episode_count = 0

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: np.ndarray,
        action_type: int,
        action_param: int,
        log_prob: float,
        reward: float,
        value: float = None,
        all_observations: List[Dict[str, np.ndarray]] = None,
        done: bool = False,
        agent_idx: int = 0,
    ):
        """
        添加一步的经验（指定agent）

        Args:
            obs: 当前agent的观测
            action_mask: 动作掩码
            action_type: 动作类型
            action_param: 动作参数
            log_prob: 动作对数概率
            reward: 奖励
            value: 价值估计（centralized critic的值）
            all_observations: 所有agents的观测（用于centralized critic）
            done: 是否结束
            agent_idx: agent索引 (0-3)
        """

        # 添加到当前episode
        self.current_obs[agent_idx].append(obs)
        self.current_action_masks[agent_idx].append(action_mask)
        self.current_actions_type[agent_idx].append(action_type)
        self.current_actions_param[agent_idx].append(action_param)
        self.current_log_probs[agent_idx].append(log_prob)
        self.current_rewards[agent_idx].append(reward)

        if value is not None:
            self.current_values[agent_idx].append(value)

        if all_observations is not None:
            self.current_all_observations[agent_idx].append(all_observations)

        self.current_dones[agent_idx].append(done)
        self.size += 1

    def add_multi_agent(
        self,
        all_observations: List[Dict[str, np.ndarray]],
        action_masks: List[np.ndarray],
        actions_type: List[int],
        actions_param: List[int],
        log_probs: List[float],
        rewards: List[float],
        values: List[float] = None,
        done: bool = False,
    ):
        """
        一次添加所有4个agents的经验（在episode结束时调用）

        Args:
            all_observations: 所有agents的观测（4个观测）
            action_masks: 所有agents的动作掩码（4个掩码）
            actions_type: 所有agents的动作类型（4个类型）
            actions_param: 所有agents的动作参数（4个参数）
            log_probs: 所有agents的对数概率（4个概率）
            rewards: 所有agents的奖励（4个奖励）
            values: 所有agents的价值估计（4个价值，可选）
            done: episode是否结束
        """
        # 对每个agent添加数据
        for agent_idx in range(4):
            self.current_obs[agent_idx].append(all_observations[agent_idx])
            self.current_action_masks[agent_idx].append(action_masks[agent_idx])
            self.current_actions_type[agent_idx].append(actions_type[agent_idx])
            self.current_actions_param[agent_idx].append(actions_param[agent_idx])
            self.current_log_probs[agent_idx].append(log_probs[agent_idx])
            self.current_rewards[agent_idx].append(rewards[agent_idx])

            # Store values if provided
            if values is not None:
                self.current_values[agent_idx].append(values[agent_idx])

            self.current_all_observations[agent_idx].append(all_observations)
            self.current_dones[agent_idx].append(done)

        self.size += 4

    def finish_episode(self):
        """
        结束当前episode，将数据存储到缓冲区

        Returns:
            episode_data: 当前episode的字典数据
        """
        # 计算每个agent的步数
        episode_lengths = [len(self.current_obs[i]) for i in range(4)]
        episode_data = {
            "observations": [
                list(obs_list) for obs_list in self.current_obs
            ],  # List[List[Dict]]
            "action_masks": [
                list(mask_list) for mask_list in self.current_action_masks
            ],  # List[List[np.ndarray]]
            "actions_type": [
                list(type_list) for type_list in self.current_actions_type
            ],  # List[List[int]]
            "actions_param": [
                list(param_list) for param_list in self.current_actions_param
            ],  # List[List[int]]
            "log_probs": [
                list(prob_list) for prob_list in self.current_log_probs
            ],  # List[List[float]]
            "rewards": [
                list(reward_list) for reward_list in self.current_rewards
            ],  # List[List[float]]
            "values": [list(value_list) for value_list in self.current_values]
            if self.current_values
            else [],  # List[List[float]]
            "dones": [
                list(done_list) for done_list in self.current_dones
            ],  # List[List[bool]]
            "all_observations": [
                list(obs_list) for obs_list in self.current_all_observations
            ],  # List[List[Dict]]
            "episode_lengths": episode_lengths,  # List[int]
        }

        # 存储到episodes列表
        self.episodes.append(episode_data)

        # 重置当前episode数据
        for agent_idx in range(4):
            self.current_obs[agent_idx] = []
            self.current_action_masks[agent_idx] = []
            self.current_actions_type[agent_idx] = []
            self.current_actions_param[agent_idx] = []
            self.current_log_probs[agent_idx] = []
            self.current_rewards[agent_idx] = []
            self.current_values[agent_idx] = []
            self.current_dones[agent_idx] = []
            self.current_all_observations[agent_idx] = []

        self.episode_count += 1

        # 如果超过容量，移除最旧的episode
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

        return episode_data

    def get_centralized_batch(self, batch_size: int, device: str = "cuda") -> Tuple:
        """
        获取centralized critic训练的批次数据

        Args:
            batch_size: 批次大小
            device: 设备类型

        Returns:
            all_observations: 所有agents的观测 [batch_size, 4, Dict]
            actions: 所有agents的动作 [batch_size, 4, ...]
            rewards: 所有agents的奖励 [batch_size, 4]
            values: centralized critic的价值（如果存在）
        """
        if len(self.episodes) < batch_size:
            batch_size = len(self.episodes)

        # 随机采样episode
        indices = np.random.choice(len(self.episodes), batch_size, replace=False)

        # 收集数据
        batch_all_observations = []
        batch_actions_type = []
        batch_actions_param = []
        batch_rewards = []
        batch_values = []
        batch_dones = []

        for idx in indices:
            episode = self.episodes[idx]

            # 确保所有agents的数据对齐
            # 我们需要4个agents的观测
            # episode['all_observations'] 应该是 List[List[Dict]]，其中内层是4个agents

            # 转置episode数据：从 [4, num_steps] 到 [num_steps, 4, agent_data]
            # 每个episode的长度应该相同（所有agents在同一episode步数相同）
            num_steps = episode["episode_lengths"][0]  # 所有agents步数相同

            # 转置数据
            batch_obs = []
            for agent_idx in range(4):
                agent_obs_list = [
                    episode["observations"][agent_idx][step_idx]
                    for step_idx in range(num_steps)
                ]
                batch_obs.append(agent_obs_list)

            # 转置数据：从 [4, num_steps, Dict] 到 [num_steps, 4, Dict]
            # episode["all_observations"] 的格式是 [4, num_steps]
            # 其中 outer list 是 4 个 agents，inner list 是该 agent 的 num_steps 个观测
            # 我们需要转置为 [num_steps, 4, Dict] 格式
            # 其中每个元素是一个步骤，包含所有 4 个 agents 的观测

            # IMPORTANT: Use "observations" not "all_observations"
            # - "observations" = single agent's observation (Dict)
            # - "all_observations" = what an agent sees of all agents (List[Dict])
            # Centralized critic needs single agent obs, not the "all" view

            num_steps = (
                len(episode["observations"][0])
                if episode["observations"]
                else 0
            )

            # 创建转置后的列表 - 从 [4, num_steps] 到 [num_steps, 4]
            episode_all_observations = []
            for step_idx in range(num_steps):
                step_all_agents_obs = []
                for agent_idx in range(4):
                    if agent_idx < len(episode["observations"]):
                        if step_idx < len(episode["observations"][agent_idx]):
                            step_all_agents_obs.append(
                                episode["observations"][agent_idx][step_idx]
                            )
                        else:
                            # 如果数据不完整，用默认值填充
                            step_all_agents_obs.append({})
                    else:
                        step_all_agents_obs.append({})
                episode_all_observations.append(step_all_agents_obs)

            # 现在 batch_all_observations 的格式是 [num_steps, 4, Dict]
            # Note: Use different variable names to avoid shadowing outer scope
            episode_actions_type = [
                [
                    episode["actions_type"][agent_idx][step_idx]
                    for step_idx in range(num_steps)
                ]
                for agent_idx in range(4)
            ]
            episode_actions_param = [
                [
                    episode["actions_param"][agent_idx][step_idx]
                    for step_idx in range(num_steps)
                ]
                for agent_idx in range(4)
            ]
            episode_rewards = [
                [
                    episode["rewards"][agent_idx][step_idx]
                    for step_idx in range(num_steps)
                ]
                for agent_idx in range(4)
            ]

            if "values" in episode and episode["values"]:
                episode_values = [
                    [
                        episode["values"][agent_idx][step_idx]
                        for step_idx in range(num_steps)
                    ]
                    for agent_idx in range(4)
                ]
            else:
                episode_values = None

            episode_dones = [
                [episode["dones"][agent_idx][step_idx] for step_idx in range(num_steps)]
                for agent_idx in range(4)
            ]

            batch_all_observations.append(episode_all_observations)
            batch_actions_type.append(episode_actions_type)
            batch_actions_param.append(episode_actions_param)
            batch_rewards.append(episode_rewards)
            batch_values.append(episode_values)
            batch_dones.append(episode_dones)

        # 转换为numpy数组
        # batch_all_observations 保持为 list（包含嵌套的 dict）
        # 其他字段转换为 numpy array
        batch_actions_type = np.array(batch_actions_type, dtype=np.int64)
        batch_actions_param = np.array(batch_actions_param, dtype=np.int64)
        batch_rewards = np.array(batch_rewards, dtype=np.float32)

        # Handle values - filter out None episodes
        if batch_values and any(v is not None for v in batch_values):
            # Only convert non-None values
            batch_values_filtered = []
            for v in batch_values:
                if v is not None:
                    batch_values_filtered.append(v)
            if batch_values_filtered:
                batch_values = np.array(batch_values_filtered, dtype=np.float32)
            else:
                batch_values = None
        else:
            batch_values = None

        batch_dones = np.array(batch_dones, dtype=np.bool_)

        return (
            batch_all_observations,  # [batch_size, num_steps, 4, Dict]
            batch_actions_type,  # [batch_size, num_steps, 4] numpy array
            batch_actions_param,  # [batch_size, num_steps, 4] numpy array
            batch_rewards,  # [batch_size, num_steps, 4] numpy array
            batch_values,  # [batch_size_filtered, num_steps, 4] numpy array or None
            batch_dones,  # [batch_size, num_steps, 4] numpy array
        )

    def get_decentralized_batch(
        self, batch_size: int, agent_idx: int = 0, device: str = "cuda"
    ) -> Tuple:
        """
        获取decentralized critic训练的批次数据（单个agent）

        Args:
            batch_size: 批次大小
            agent_idx: agent索引（0-3）
            device: 设备类型

        Returns:
            observations: 单个agent的观测 [batch_size, ...]
            actions: 单个agent的动作 [batch_size, ...]
            rewards: 单个agent的奖励 [batch_size, ...]
        """
        if len(self.episodes) < batch_size:
            batch_size = len(self.episodes)

        # 随机采样episode
        indices = np.random.choice(len(self.episodes), batch_size, replace=False)

        # 收集单个agent的数据
        batch_obs = []
        batch_action_masks = []
        batch_actions_type = []
        batch_actions_param = []
        batch_log_probs = []
        batch_rewards = []
        batch_dones = []

        for idx in indices:
            episode = self.episodes[idx]
            num_steps = episode["episode_lengths"][agent_idx]

            # 提取agent_idx的数据
            batch_obs.extend(
                episode["observations"][step_idx][agent_idx]
                for step_idx in range(num_steps)
            )
            batch_action_masks.extend(
                episode["action_masks"][step_idx][agent_idx]
                for step_idx in range(num_steps)
            )
            batch_actions_type.extend(
                episode["actions_type"][step_idx][agent_idx]
                for step_idx in range(num_steps)
            )
            batch_actions_param.extend(
                episode["actions_param"][step_idx][agent_idx]
                for step_idx in range(num_steps)
            )
            batch_log_probs.extend(
                episode["log_probs"][step_idx][agent_idx]
                for step_idx in range(num_steps)
            )
            batch_rewards.extend(
                episode["rewards"][step_idx][agent_idx] for step_idx in range(num_steps)
            )
            batch_dones.extend(
                episode["dones"][step_idx][agent_idx] for step_idx in range(num_steps)
            )

        # 转换为torch张量
        batch_obs = [np.array(obs) for obs in batch_obs]
        batch_action_masks = np.array(batch_action_masks)
        batch_actions_type = np.array(batch_actions_type)
        batch_actions_param = np.array(batch_actions_param)
        batch_log_probs = np.array(batch_log_probs)
        batch_rewards = np.array(batch_rewards)
        batch_dones = np.array(batch_dones)

        # 转换为torch张量
        batch_obs = torch.FloatTensor(batch_obs).to(device)
        batch_action_masks = torch.FloatTensor(batch_action_masks).to(device)
        batch_actions_type = torch.LongTensor(batch_actions_type).to(device)
        batch_actions_param = torch.LongTensor(batch_actions_param).to(device)
        batch_log_probs = torch.FloatTensor(batch_log_probs).to(device)
        batch_rewards = torch.FloatTensor(batch_rewards).to(device)
        batch_dones = torch.FloatTensor(batch_dones).to(device)

        return (
            batch_obs,
            batch_action_masks,
            batch_actions_type,
            batch_actions_param,
            batch_log_probs,
            batch_rewards,
            batch_dones,
        )

    def __len__(self):
        return self.size

    def get_episode_count(self):
        """返回episode数量"""
        return self.episode_count
