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
        next_action_mask: Optional[np.ndarray] = None
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
    
    def get_batch(
        self,
        batch_size: int,
        device: str = 'cuda'
    ) -> Tuple:
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
            batch_dones
        )
    
    def get_all(self, device: str = 'cuda') -> Tuple:
        """获取所有数据"""
        return self.get_batch(self.size, device)
    
    def compute_returns_and_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        next_value: float = 0.0
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
    
    def add(self, obs: Dict[str, np.ndarray], action_mask: np.ndarray, 
            action_type: int, action_param: int):
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
    
    def sample(
        self,
        batch_size: int,
        device: str = 'cuda'
    ) -> Tuple:
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
            'observations': [],
            'action_masks': [],
            'actions_type': [],
            'actions_param': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
    
    def add_step(self, **kwargs):
        """添加一步"""
        for key, value in kwargs.items():
            if key in self.current_episode:
                self.current_episode[key].append(value)
    
    def end_episode(self):
        """结束当前回合"""
        if len(self.current_episode['observations']) > 0:
            self.episodes.append(self.current_episode)
            self.current_episode = {
                'observations': [],
                'action_masks': [],
                'actions_type': [],
                'actions_param': [],
                'log_probs': [],
                'rewards': [],
                'values': [],
                'dones': []
            }
    
    def get_all_episodes(self) -> List[Dict]:
        """获取所有回合"""
        return self.episodes
    
    def clear(self):
        """清空"""
        self.episodes = []
        self.current_episode = {
            'observations': [],
            'action_masks': [],
            'actions_type': [],
            'actions_param': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
    
    def __len__(self):
        return len(self.episodes)


class MixedBuffer:
    """
    混合缓冲区
    同时管理 RL 缓冲区和 SL 缓冲区
    """
    
    def __init__(
        self,
        rl_capacity: int = 100000,
        sl_capacity: int = 2000000
    ):
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
        next_action_mask: Optional[np.ndarray] = None
    ):
        """添加 RL 经验"""
        self.rl_buffer.add(
            obs, action_mask, action_type, action_param,
            log_prob, reward, value, done, next_obs, next_action_mask
        )
    
    def add_sl_experience(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: np.ndarray,
        action_type: int,
        action_param: int
    ):
        """添加 SL 经验"""
        self.sl_buffer.add(obs, action_mask, action_type, action_param)
    
    def sample_rl_batch(self, batch_size: int, device: str = 'cuda'):
        """采样 RL 批次"""
        return self.rl_buffer.get_batch(batch_size, device)
    
    def sample_sl_batch(self, batch_size: int, device: str = 'cuda'):
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
