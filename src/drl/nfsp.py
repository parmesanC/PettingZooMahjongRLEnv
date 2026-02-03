"""
NFSP (Neural Fictitious Self-Play) 协调器

整合 MAPPO（强化学习）和监督学习，实现 NFSP 算法。

核心机制：
1. Anticipatory 策略：以概率 η 使用最佳响应网络，以概率 (1-η) 使用平均策略网络
2. 最佳响应网络：使用 MAPPO 训练，学习对当前对手策略的最佳响应
3. 平均策略网络：使用监督学习训练，模仿最佳响应网络的历史行为
4. 混合缓冲区：同时维护 RL 缓冲区（MAPPO）和 SL 缓冲区（Reservoir）
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import random

from .network import ActorCriticNetwork, AveragePolicyNetwork, create_networks
from .buffer import MixedBuffer, RolloutBuffer
from .mappo import MAPPO, SupervisedLearning
from .config import NFSPConfig, MAPPOConfig


class NFSP:
    """
    Neural Fictitious Self-Play 协调器
    
    管理两个网络和两个训练过程：
    - 最佳响应网络（Actor-Critic）：MAPPO 训练
    - 平均策略网络：监督学习训练
    """
    
    def __init__(
        self,
        config,
        device: str = 'cuda'
    ):
        """
        初始化 NFSP
        
        Args:
            config: 配置对象（包含 NFSPConfig 和 MAPPOConfig）
            device: 计算设备
        """
        self.config = config
        self.device = device
        
        # Anticipatory 参数
        self.eta = config.nfsp.eta
        
        # 创建网络
        self.best_response_net, self.average_policy_net = create_networks(
            config, device
        )
        
        # 创建缓冲区
        self.buffer = MixedBuffer(
            rl_capacity=config.nfsp.rl_buffer_size,
            sl_capacity=config.nfsp.sl_buffer_size
        )
        
        # 创建训练器
        self.mappo = MAPPO(
            network=self.best_response_net,
            lr=config.mappo.lr,
            gamma=config.mappo.gamma,
            gae_lambda=config.mappo.gae_lambda,
            clip_ratio=config.mappo.clip_ratio,
            value_coef=config.mappo.value_coef,
            entropy_coef=config.mappo.entropy_coef,
            max_grad_norm=config.mappo.max_grad_norm,
            ppo_epochs=config.mappo.ppo_epochs,
            device=device
        )
        
        self.sl_trainer = SupervisedLearning(
            network=self.average_policy_net,
            lr=config.nfsp.sl_lr if hasattr(config.nfsp, 'sl_lr') else 1e-4,
            device=device
        )
        
        # 统计
        self.rl_steps = 0
        self.sl_steps = 0
        self.total_episodes = 0
    
    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: np.ndarray,
        use_best_response: Optional[bool] = None
    ) -> Tuple[int, int, float, float]:
        """
        选择动作（Anticipatory 策略）
        
        Args:
            obs: 当前观测
            action_mask: 动作掩码
            use_best_response: 是否强制使用最佳响应网络（None 则根据 η 随机选择）
        
        Returns:
            action_type: 动作类型
            action_param: 动作参数
            log_prob: 动作对数概率（仅最佳响应网络）
            value: 价值估计（仅最佳响应网络）
        """
        # 决定使用哪个网络
        if use_best_response is None:
            use_best_response = random.random() < self.eta
        
        # 准备观测
        obs_tensor = self._prepare_obs(obs)
        action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        
        if use_best_response:
            # 使用最佳响应网络
            with torch.no_grad():
                action_type, action_param, log_prob, entropy, value = \
                    self.best_response_net.get_action_and_value(
                        obs_tensor, action_mask_tensor
                    )
            
            action_type = action_type.item()
            action_param = action_param.item()
            log_prob = log_prob.item()
            value = value.item()
            
            return action_type, action_param, log_prob, value
        else:
            # 使用平均策略网络
            with torch.no_grad():
                type_probs, param_probs = self.average_policy_net.get_action_probs(
                    obs_tensor, action_mask_tensor
                )
            
            # 采样动作
            type_dist = torch.distributions.Categorical(type_probs)
            param_dist = torch.distributions.Categorical(param_probs)
            
            action_type = type_dist.sample().item()
            action_param = param_dist.sample().item()
            
            # 平均策略网络不返回 log_prob 和 value
            return action_type, action_param, 0.0, 0.0
    
    def store_transition(
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
        is_best_response: bool = True
    ):
        """
        存储转移经验
        
        Args:
            obs: 当前观测
            action_mask: 动作掩码
            action_type: 动作类型
            action_param: 动作参数
            log_prob: 动作对数概率
            reward: 奖励
            value: 价值估计
            done: 是否结束
            next_obs: 下一观测
            next_action_mask: 下一动作掩码
            is_best_response: 是否来自最佳响应网络（决定是否存入 SL 缓冲区）
        """
        # 存入 RL 缓冲区（用于 MAPPO）
        self.buffer.add_rl_experience(
            obs, action_mask, action_type, action_param,
            log_prob, reward, value, done, next_obs, next_action_mask
        )
        
        # 如果是最佳响应网络的行为，存入 SL 缓冲区
        if is_best_response:
            self.buffer.add_sl_experience(
                obs, action_mask, action_type, action_param
            )
    
    def train_step(self) -> Dict:
        """
        执行一步训练（RL + SL）
        
        Returns:
            训练统计字典
        """
        stats = {}
        
        # 1. 训练最佳响应网络（MAPPO）
        if len(self.buffer.rl_buffer) >= self.config.nfsp.rl_batch_size:
            rl_stats = self.mappo.update(self.buffer.rl_buffer)
            stats.update(rl_stats)
            self.rl_steps += 1
        
        # 2. 训练平均策略网络（SL）
        if len(self.buffer.sl_buffer) >= self.config.nfsp.sl_batch_size:
            sl_stats = self.sl_trainer.update(
                self.buffer.sl_buffer,
                batch_size=self.config.nfsp.sl_batch_size
            )
            stats.update(sl_stats)
            self.sl_steps += 1
        
        return stats
    
    def end_episode(self, final_obs=None, final_action_mask=None):
        """
        结束一回合
        
        Args:
            final_obs: 最终观测
            final_action_mask: 最终动作掩码
        """
        # 计算回报和优势（这会清空 RL 缓冲区）
        if len(self.buffer.rl_buffer) > 0:
            # 使用最后一步的价值作为 bootstrap
            next_value = 0.0
            if final_obs is not None:
                with torch.no_grad():
                    obs_tensor = self._prepare_obs(final_obs)
                    if final_action_mask is not None:
                        mask_tensor = torch.FloatTensor(final_action_mask).unsqueeze(0).to(self.device)
                    else:
                        mask_tensor = torch.ones(1, 145).to(self.device)
                    next_value = self.best_response_net.get_value(obs_tensor).item()
            
            # 计算 GAE（这会使用 next_value）
            returns, advantages = self.buffer.rl_buffer.compute_returns_and_advantages(
                gamma=self.config.mappo.gamma,
                gae_lambda=self.config.mappo.gae_lambda,
                next_value=next_value
            )
            
            # 清空 RL 缓冲区（数据已用于计算优势）
            self.buffer.clear_rl()
        
        self.total_episodes += 1
    
    def _prepare_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        将观测转换为张量
        
        Args:
            obs: 观测字典
        
        Returns:
            张量观测字典
        """
        tensor_obs = {}
        for key, value in obs.items():
            if isinstance(value, dict):
                tensor_obs[key] = self._prepare_obs(value)
            else:
                tensor_obs[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
        return tensor_obs
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            'total_episodes': self.total_episodes,
            'rl_steps': self.rl_steps,
            'sl_steps': self.sl_steps,
            'rl_buffer_size': len(self.buffer.rl_buffer),
            'sl_buffer_size': len(self.buffer.sl_buffer),
        }
        
        # 添加训练统计
        rl_stats = self.mappo.get_training_stats()
        sl_stats = self.sl_trainer.get_training_stats()
        stats.update(rl_stats)
        stats.update(sl_stats)
        
        return stats
    
    def save(self, path: str):
        """保存模型"""
        checkpoint = {
            'best_response_net': self.best_response_net.state_dict(),
            'average_policy_net': self.average_policy_net.state_dict(),
            'mappo_optimizer': self.mappo.optimizer.state_dict(),
            'sl_optimizer': self.sl_trainer.optimizer.state_dict(),
            'rl_steps': self.rl_steps,
            'sl_steps': self.sl_steps,
            'total_episodes': self.total_episodes,
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.best_response_net.load_state_dict(checkpoint['best_response_net'])
        self.average_policy_net.load_state_dict(checkpoint['average_policy_net'])
        self.mappo.optimizer.load_state_dict(checkpoint['mappo_optimizer'])
        self.sl_trainer.optimizer.load_state_dict(checkpoint['sl_optimizer'])
        self.rl_steps = checkpoint.get('rl_steps', 0)
        self.sl_steps = checkpoint.get('sl_steps', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
    
    def get_average_policy_net(self) -> AveragePolicyNetwork:
        """获取平均策略网络（用于对手）"""
        return self.average_policy_net
    
    def set_eta(self, eta: float):
        """设置 anticipatory 参数"""
        self.eta = max(0.0, min(1.0, eta))
