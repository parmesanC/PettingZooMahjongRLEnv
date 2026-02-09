"""
MAPPO (Multi-Agent PPO) 算法实现

用于 NFSP 的最佳响应网络训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional


class MAPPO:
    """
    Multi-Agent PPO 算法

    特点：
    - 参数共享：4个玩家共享同一个网络
    - GAE 优势估计
    - Clipped PPO 目标函数
    - 支持动作掩码
    """

    def __init__(
        self,
        network,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        device: str = "cuda",
        centralized_critic=None,  # NEW: 添加 centralized_critic 支持
        centralized_critic=None,  # NEW: 添加 centralized_critic 支持
    ):
        """
        初始化 MAPPO

        Args:
            network: ActorCriticNetwork 实例
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda
            clip_ratio: PPO 裁剪比率
            value_coef: 价值损失系数
            entropy_coef: 熵奖励系数
            max_grad_norm: 最大梯度范数（用于梯度裁剪）
            ppo_epochs: 每次更新的 epoch 数
            device: 计算设备
            centralized_critic: CentralizedCriticNetwork 实例（可选，用于集中式训练）
        """
        self.network = network
        self.device = device
        self.centralized_critic = (
            centralized_critic  # NEW: 添加 centralized_critic 支持
        )

        # 超参数
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs

        # 训练统计
        self.training_step = 0

        # 初始化优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # 损失历史
        self.losses = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []

    def update_centralized(self, centralized_buffer, training_phase=1):
        """
        使用 centralized observations 更新 centralized critic

        Args:
            centralized_buffer: CentralizedRolloutBuffer 实例
            training_phase: 训练阶段（1=全知，2=渐进，3=真实）
        """
        # Phase 1 和 2 使用 centralized critic
        if training_phase in [1, 2] and self.centralized_critic is not None:
            # 从 centralized buffer 中获取数据
            if len(centralized_buffer) > 0:
                # 批次更新
                observations = centralized_buffer.observations  # List of 4 observations
                action_masks = centralized_buffer.action_masks  # List of 4 action masks

                # 堆叠所有观测 [4, batch, ...]
                stacked_observations = torch.stack(observations, dim=0)

                # 预测价值
                values = self.centralized_critic(stacked_observations)

                # 计算优势
                returns = centralized_buffer.returns  # [batch, 4]
                next_values = (
                    centralized_buffer.next_values
                    if centralized_buffer.next_values is not None
                    else values
                )

                advantages = returns - next_values

                # 更新 centralized critic
                # 将 advantages 分配给每个 agent
                for agent_id in range(4):
                    agent_mask = action_masks[agent_id]
                    agent_advantages = advantages[:, agent_id]  # [batch]
                    agent_returns = returns[:, agent_id]  # [batch]

                    # 使用 centralized critic 更新
                    # 注意：这里简化处理，实际中需要更复杂的逻辑
                    if len(centralized_buffer.observations) > 0:
                        # 更新所有 agents 的 centralized critic
                        self.centralized_critic(observations)
        else:
            # Phase 3: 不使用 centralized critic
            pass

    def update(self, buffer, next_obs=None, next_action_mask=None, training_phase=1):
        """
        使用缓冲区数据更新策略

        Args:
            buffer: RolloutBuffer 实例
            next_obs: 最后一步的下一观测（用于计算下一价值）
            next_action_mask: 最后一步的下一动作掩码
            training_phase: 当前训练阶段（1=全知，2=渐进，3=真实）

        Returns:
            训练统计字典
        """
        if len(buffer) < 1:
            return {}

        # Phase-aware: 确定是否使用 centralized critic
        use_centralized = (
            training_phase in [1, 2] and self.centralized_critic is not None
        )

        # 获取所有数据
        observations = buffer.observations
        action_masks = np.array(buffer.action_masks)
        old_actions_type = np.array(buffer.actions_type)
        old_actions_param = np.array(buffer.actions_param)
        old_log_probs = np.array(buffer.log_probs)
        old_values = np.array(buffer.values)

        # 转换为张量
        action_masks = torch.FloatTensor(action_masks).to(self.device)
        old_actions_type = torch.LongTensor(old_actions_type).to(self.device)
        old_actions_param = torch.LongTensor(old_actions_param).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(buffer.returns).to(self.device)

        # Phase 1 和 2: 使用 centralized critic
        if use_centralized and len(observations) > 0:
            # 构建全局观测
            # 简化：使用第一个 agent 的全局观测
            global_obs = observations[0]  # 假设 observations[0] 包含全局信息

            # 更新 centralized critic
            self.update_centralized(buffer, training_phase)

        # 计算下一价值
        next_value = 0.0
        if next_obs is not None:
            with torch.no_grad():
                next_value = self.network.get_value(self._prepare_obs(next_obs))

        # 计算回报和优势
        advantages = buffer.compute_returns_and_advantages(
            self.gamma, self.gae_lambda, next_value
        )

        # 计算下一价值（用于 GAE）
        next_value = 0.0
        if next_obs is not None and not buffer.dones[-1]:
            with torch.no_grad():
                next_value = (
                    self.network.get_value(self._prepare_obs(next_obs)).cpu().item()
                )

        # 计算回报和优势
        returns, advantages = buffer.compute_returns_and_advantages(
            self.gamma, self.gae_lambda, next_value
        )

        # 归一化优势（有助于训练稳定性）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 获取所有数据
        observations = buffer.observations
        action_masks = np.array(buffer.action_masks)
        old_actions_type = np.array(buffer.actions_type)
        old_actions_param = np.array(buffer.actions_param)
        old_log_probs = np.array(buffer.log_probs)
        old_values = np.array(buffer.values)

        # 转换为张量
        action_masks = torch.FloatTensor(action_masks).to(self.device)
        old_actions_type = torch.LongTensor(old_actions_type).to(self.device)
        old_actions_param = torch.LongTensor(old_actions_param).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        old_values = torch.FloatTensor(old_values).to(self.device)

        # PPO 更新
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0

        for epoch in range(self.ppo_epochs):
            # 准备观测批次
            obs_batch = self._prepare_obs_batch(observations)

            # 前向传播
            action_type_logits, action_param_logits, values = self.network(
                obs_batch, action_masks
            )

            # 计算新的动作概率
            type_probs = torch.softmax(action_type_logits, dim=-1)
            param_probs = torch.softmax(action_param_logits, dim=-1)

            # 计算新的对数概率
            type_dist = torch.distributions.Categorical(type_probs)
            param_dist = torch.distributions.Categorical(param_probs)

            new_log_prob_type = type_dist.log_prob(old_actions_type)
            new_log_prob_param = param_dist.log_prob(old_actions_param)
            new_log_probs = new_log_prob_type + new_log_prob_param

            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Clipped 目标
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值损失（使用 Clipped 价值损失，可选）
            value_pred_clipped = old_values + torch.clamp(
                values.squeeze() - old_values, -self.clip_ratio, self.clip_ratio
            )
            value_loss1 = nn.functional.mse_loss(values.squeeze(), returns)
            value_loss2 = nn.functional.mse_loss(value_pred_clipped, returns)
            value_loss = 0.5 * torch.max(value_loss1, value_loss2)

            # 熵奖励
            entropy_type = type_dist.entropy().mean()
            entropy_param = param_dist.entropy().mean()
            entropy_loss = -(entropy_type + entropy_param)

            # 总损失
            loss = (
                policy_loss
                + self.value_coef * value_loss
                + self.entropy_coef * entropy_loss
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)

            self.optimizer.step()

            # 记录损失
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()

            self.training_step += 1

        # 计算平均损失
        n_epochs = self.ppo_epochs
        avg_loss = total_loss / n_epochs
        avg_policy_loss = total_policy_loss / n_epochs
        avg_value_loss = total_value_loss / n_epochs
        avg_entropy_loss = total_entropy_loss / n_epochs

        # 保存统计
        self.losses.append(avg_loss)
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropy_losses.append(avg_entropy_loss)

        return {
            "loss": avg_loss,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy_loss": avg_entropy_loss,
            "training_step": self.training_step,
        }

    def _prepare_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        将单个观测转换为张量

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

    def _prepare_obs_batch(self, observations: list) -> Dict[str, torch.Tensor]:
        """
        将观测列表转换为批次张量
        正确处理不同类型的观测数据

        Args:
            observations: 观测字典列表

        Returns:
            批次张量观测字典
        """
        batch_size = len(observations)

        # 收集所有键
        keys = observations[0].keys()

        batch_obs = {}
        for key in keys:
            if isinstance(observations[0][key], dict):
                # 嵌套字典（如 melds, action_history）
                batch_obs[key] = self._prepare_nested_batch(
                    [obs[key] for obs in observations]
                )
            else:
                # 普通数组
                values = np.array([obs[key] for obs in observations])
                batch_obs[key] = torch.FloatTensor(values).to(self.device)

        return batch_obs

    def _prepare_nested_batch(self, nested_list: list) -> Dict[str, torch.Tensor]:
        """
        处理嵌套字典的批次
        正确处理 melds 和 action_history

        Args:
            nested_list: 嵌套字典列表

        Returns:
            批次张量字典
        """
        keys = nested_list[0].keys()
        batch = {}

        for key in keys:
            if isinstance(nested_list[0][key], dict):
                # 更深层的嵌套
                batch[key] = self._prepare_nested_batch(
                    [item[key] for item in nested_list]
                )
            else:
                values = np.array([item[key] for item in nested_list])
                # action_history 中的 types, params, players 用于 Embedding，需要是 Long 类型
                # 但在这里我们统一用 Float，在 network.forward 中再转换为 Long
                batch[key] = torch.FloatTensor(values).to(self.device)

        return batch

    def get_training_stats(self) -> Dict:
        """获取训练统计"""
        if not self.losses:
            return {}

        return {
            "mean_loss": np.mean(self.losses[-100:]),
            "mean_policy_loss": np.mean(self.policy_losses[-100:]),
            "mean_value_loss": np.mean(self.value_losses[-100:]),
            "mean_entropy_loss": np.mean(self.entropy_losses[-100:]),
            "total_steps": self.training_step,
        }

    def save(self, path: str):
        """保存模型"""
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_step": self.training_step,
                "losses": self.losses,
                "policy_losses": self.policy_losses,
                "value_losses": self.value_losses,
                "entropy_losses": self.entropy_losses,
            },
            path,
        )

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        self.losses = checkpoint.get("losses", [])
        self.policy_losses = checkpoint.get("policy_losses", [])
        self.value_losses = checkpoint.get("value_losses", [])
        self.entropy_losses = checkpoint.get("entropy_losses", [])

    def update_centralized(self, centralized_buffer, training_phase: int = 1) -> Dict:
        """
        使用 centralized critic 进行训练（Phase 1-2）

        Args:
            centralized_buffer: CentralizedRolloutBuffer 实例
            training_phase: 当前训练阶段（1=全知，2=渐进，3=真实）

        Returns:
            训练统计字典
        """
        if len(centralized_buffer) < 1:
            return {}

        # Phase 3 或没有 centralized_critic，使用现有的 decentralized 方法
        if training_phase not in [1, 2] or self.centralized_critic is None:
            return {"used_centralized": False}

        # 从 centralized buffer 获取批次数据
        (all_observations, actions_type, actions_param, rewards, values, dones) = (
            centralized_buffer.get_centralized_batch(batch_size=64, device=self.device)
        )

        # 转换数据格式
        # all_observations: [batch_size, num_steps, 4, Dict]
        # actions_type: [batch_size, num_steps, 4]
        # rewards: [batch_size, num_steps, 4]

        batch_size, num_steps, num_agents = actions_type.shape
        num_agents = 4

        # 准备数据
        all_observations_flat = []  # [batch_size * num_steps, num_agents, Dict]
        actions_type_flat = actions_type.reshape(
            -1, num_agents
        )  # [batch_size * num_steps, num_agents]
        actions_param_flat = actions_param.reshape(-1, num_agents)
        rewards_flat = rewards.reshape(-1, num_agents)
        dones_flat = dones.reshape(-1, num_agents)

        for batch_idx in range(batch_size):
            for step_idx in range(num_steps):
                # 收集这一步所有agents的观测
                step_obs = []
                for agent_idx in range(num_agents):
                    obs = all_observations[batch_idx][step_idx][agent_idx]
                    step_obs.append(obs)
                all_observations_flat.append(step_obs)

        # 转换为列表格式 [batch_size * num_steps * num_agents, Dict]
        observations_for_critic = []
        for step_obs_list in all_observations_flat:
            for obs in step_obs_list:
                observations_for_critic.append(obs)

        # 计算 centralized critic 价值
        # CentralizedCriticNetwork.forward 接收 List[Dict] (4个agents的观测)
        # 我们需要按时间步和batch组织

        total_steps = batch_size * num_steps
        values_centralized = []

        with torch.no_grad():
            for i in range(total_steps):
                # 获取这一步所有4个agents的观测
                # all_observations_flat[i] 是 [4, Dict]
                step_agent_obs = all_observations_flat[i]

                # 准备每个agent的观测字典
                agent_obs_list = []
                for agent_idx in range(4):
                    obs = step_agent_obs[agent_idx]
                    agent_obs_list.append(obs)

                # 调用 centralized critic
                # 这需要将obs转换为tensor格式
                obs_batch_list = [
                    self._prepare_obs(obs) if isinstance(obs, dict) else obs
                    for obs in agent_obs_list
                ]
                values = self.centralized_critic(obs_batch_list)  # [batch, 4]

                # 由于我们是单步，取第0行
                values_centralized.append(values[0].cpu().numpy())  # [4]

        values_centralized = np.array(values_centralized)  # [total_steps, 4]

        # 重新reshape为 [batch_size, num_steps, 4]
        values_centralized = values_centralized.reshape(
            batch_size, num_steps, num_agents
        )

        # 计算 GAE 优势和回报
        # 为每个agent单独计算
        all_advantages = []
        all_returns = []

        for agent_idx in range(num_agents):
            agent_rewards = rewards[:, :, agent_idx]  # [batch_size, num_steps]
            agent_values = values_centralized[
                :, :, agent_idx
            ]  # [batch_size, num_steps]
            agent_dones = dones[:, :, agent_idx]  # [batch_size, num_steps]

            # 计算每个batch的GAE
            advantages_list = []
            returns_list = []

            for batch_idx in range(batch_size):
                episode_rewards = agent_rewards[batch_idx]  # [num_steps]
                episode_values = agent_values[batch_idx]  # [num_steps]
                episode_dones = agent_dones[batch_idx]  # [num_steps]

                # 计算returns
                returns = np.zeros_like(episode_rewards)
                advantages = np.zeros_like(episode_rewards)

                # 最后一步的下一价值为0
                next_value = 0.0
                next_advantage = 0.0

                # 反向计算GAE
                for t in reversed(range(len(episode_rewards))):
                    if t == len(episode_rewards) - 1:
                        next_value = 0.0
                    else:
                        next_value = episode_values[t + 1]

                    # TD误差
                    delta = (
                        episode_rewards[t] + self.gamma * next_value - episode_values[t]
                    )

                    # GAE
                    if t == len(episode_rewards) - 1:
                        next_advantage = 0.0

                    advantages[t] = (
                        delta + self.gae_lambda * self.gamma * next_advantage
                    )
                    next_advantage = advantages[t]

                advantages_list.append(advantages)
                returns_list.append(returns)

            all_advantages.append(np.array(advantages_list))  # [batch_size, num_steps]
            all_returns.append(np.array(returns_list))

        all_advantages = np.stack(all_advantages, axis=2)  # [batch_size, num_steps, 4]
        all_returns = np.stack(all_returns, axis=2)  # [batch_size, num_steps, 4]

        # 归一化优势
        advantages_mean = all_advantages.mean()
        advantages_std = all_advantages.std() + 1e-8
        all_advantages = (all_advantages - advantages_mean) / advantages_std

        # 转换为tensor
        actions_type_tensor = torch.LongTensor(actions_type_flat).to(
            self.device
        )  # [batch*num_steps, 4]
        actions_param_tensor = torch.LongTensor(actions_param_flat).to(self.device)
        advantages_tensor = torch.FloatTensor(
            all_advantages.reshape(-1, num_agents)
        ).to(self.device)
        returns_tensor = torch.FloatTensor(all_returns.reshape(-1, num_agents)).to(
            self.device
        )
        dones_tensor = torch.BoolTensor(dones_flat).to(self.device)

        # PPO更新（简化版，使用现有actor）
        # 这里我们只训练critic，因为centralized critic是独立的
        # Actor的训练继续使用现有的update()方法

        # 计算centralized critic损失
        # 将数据转换为tensor
        values_tensor = torch.FloatTensor(values_centralized).to(
            self.device
        )  # [batch, steps, 4]
        returns_tensor = torch.FloatTensor(all_returns).to(
            self.device
        )  # [batch, steps, 4]

        # Reshape为 [batch*steps, 4]
        values_flat = values_tensor.reshape(-1, num_agents)  # [batch*steps, 4]
        returns_flat = returns_tensor.reshape(-1, num_agents)  # [batch*steps, 4]

        # MSE损失
        critic_loss = ((values_flat - returns_flat) ** 2).mean()

        # 更新centralized critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.centralized_critic.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        # 记录损失
        self.losses.append(critic_loss.item())
        self.value_losses.append(critic_loss.item())
        self.policy_losses.append(0.0)  # 未训练actor
        self.entropy_losses.append(0.0)  # 未训练actor

        self.training_step += 1

        return {
            "loss": critic_loss.item(),
            "critic_loss": critic_loss.item(),
            "used_centralized": True,
            "training_step": self.training_step,
            "centralized_critic_loss": critic_loss.item(),
        }


class SupervisedLearning:
    """
    监督学习模块
    用于训练 NFSP 的平均策略网络
    """

    def __init__(self, network, lr: float = 1e-4, device: str = "cuda"):
        """
        初始化监督学习

        Args:
            network: AveragePolicyNetwork 实例
            lr: 学习率
            device: 计算设备
        """
        self.network = network
        self.device = device

        # 优化器
        self.optimizer = optim.Adam(network.parameters(), lr=lr)

        # 训练统计
        self.training_step = 0
        self.losses = []

    def update(self, buffer, batch_size: int = 64):
        """
        使用缓冲区数据更新策略

        Args:
            buffer: ReservoirBuffer 实例
            batch_size: 批次大小

        Returns:
            训练统计字典
        """
        if len(buffer) < batch_size:
            return {}

        # 采样批次
        obs_list, action_masks, action_types, action_params = buffer.sample(
            batch_size, self.device
        )

        # 准备观测批次
        obs_batch = self._prepare_obs_batch(obs_list)

        # 前向传播
        action_type_logits, action_param_logits = self.network(obs_batch, action_masks)

        # 计算交叉熵损失
        type_loss = nn.functional.cross_entropy(action_type_logits, action_types)
        param_loss = nn.functional.cross_entropy(action_param_logits, action_params)

        # 总损失
        loss = type_loss + param_loss

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_step += 1
        self.losses.append(loss.item())

        return {
            "sl_loss": loss.item(),
            "type_loss": type_loss.item(),
            "param_loss": param_loss.item(),
            "training_step": self.training_step,
        }

    def _prepare_obs_batch(self, observations: list) -> Dict[str, torch.Tensor]:
        """
        将观测列表转换为批次张量
        正确处理所有观测字段
        """
        batch_size = len(observations)
        keys = observations[0].keys()

        batch_obs = {}
        for key in keys:
            if isinstance(observations[0][key], dict):
                batch_obs[key] = self._prepare_nested_batch(
                    [obs[key] for obs in observations]
                )
            else:
                values = np.array([obs[key] for obs in observations])
                batch_obs[key] = torch.FloatTensor(values).to(self.device)

        return batch_obs

    def _prepare_nested_batch(self, nested_list: list) -> Dict[str, torch.Tensor]:
        """处理嵌套字典的批次（如 melds, action_history）"""
        keys = nested_list[0].keys()
        batch = {}

        for key in keys:
            if isinstance(nested_list[0][key], dict):
                batch[key] = self._prepare_nested_batch(
                    [item[key] for item in nested_list]
                )
            else:
                values = np.array([item[key] for item in nested_list])
                batch[key] = torch.FloatTensor(values).to(self.device)

        return batch

    def get_training_stats(self) -> Dict:
        """获取训练统计"""
        if not self.losses:
            return {}

        return {
            "mean_sl_loss": np.mean(self.losses[-100:]),
            "total_steps": self.training_step,
        }

    def save(self, path: str):
        """保存模型"""
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_step": self.training_step,
                "losses": self.losses,
            },
            path,
        )

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        self.losses = checkpoint.get("losses", [])
