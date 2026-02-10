"""
NFSP 智能体封装

提供与麻将环境交互的接口，继承自 PlayerStrategy 基类。
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional
import random

from src.mahjong_rl.agents.base import PlayerStrategy
from src.drl.nfsp import NFSP
from src.drl.config import Config


class NFSPAgent(PlayerStrategy):
    """
    NFSP 智能体

    继承自 PlayerStrategy，可与麻将环境交互
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        device: str = "cuda",
        eta: Optional[float] = None,
        is_training: bool = True,
    ):
        """
        初始化 NFSP 智能体

        Args:
            config: 配置对象（None 则使用默认配置）
            device: 计算设备
            eta: Anticipatory 参数（None 则使用配置中的值）
            is_training: 是否处于训练模式
        """
        super().__init__()

        # 配置
        if config is None:
            from .config import get_default_config

            config = get_default_config()

        # 覆盖 eta（如果提供）
        if eta is not None:
            config.nfsp.eta = eta

        self.config = config
        self.device = device
        self.is_training = is_training

        # 创建 NFSP 协调器
        self.nfsp = NFSP(config, device)

        # 当前回合状态
        self.current_episode_reward = 0.0
        self.episode_count = 0

        # 统计
        self.total_steps = 0
        self.wins = 0
        self.games = 0

    def choose_action(
        self, observation: Dict[str, np.ndarray], action_mask: np.ndarray
    ) -> Tuple[int, int]:
        """
        选择动作（实现 PlayerStrategy 接口）

        Args:
            observation: 观测字典
            action_mask: 动作掩码（145位）

        Returns:
            (action_type, action_param) 元组
        """
        # 使用 NFSP 选择动作
        action_type, action_param, log_prob, value = self.nfsp.select_action(
            observation, action_mask
        )

        # 如果是训练模式，存储转移
        if self.is_training:
            # 这里需要在外部调用 store_transition，因为需要奖励和下一状态
            self.last_obs = observation
            self.last_action_mask = action_mask
            self.last_action = (action_type, action_param)
            self.last_log_prob = log_prob
            self.last_value = value

        self.total_steps += 1

        return action_type, action_param

    def store_transition(
        self,
        reward: float,
        done: bool,
        next_observation: Optional[Dict[str, np.ndarray]] = None,
        next_action_mask: Optional[np.ndarray] = None,
    ):
        """
        存储转移经验（训练模式）

        需要在环境返回奖励后调用

        Args:
            reward: 奖励
            done: 是否结束
            next_observation: 下一观测
            next_action_mask: 下一动作掩码
        """
        if not self.is_training:
            return

        if hasattr(self, "last_obs"):
            action_type, action_param = self.last_action

            self.nfsp.store_transition(
                obs=self.last_obs,
                action_mask=self.last_action_mask,
                action_type=action_type,
                action_param=action_param,
                log_prob=self.last_log_prob,
                reward=reward,
                value=self.last_value,
                done=done,
                next_obs=next_observation,
                next_action_mask=next_action_mask,
                is_best_response=True,  # 假设使用最佳响应网络
            )

            self.current_episode_reward += reward

    def end_episode(self, won: bool = False):
        """
        结束一回合

        Args:
            won: 是否获胜
        """
        if self.is_training:
            self.nfsp.end_episode()

        self.episode_count += 1
        self.games += 1
        if won:
            self.wins += 1

        # 重置回合奖励
        self.current_episode_reward = 0

    def train_step(self) -> Dict:
        """
        执行训练步骤（训练模式）

        Returns:
            训练统计
        """
        if not self.is_training:
            return {}

        return self.nfsp.train_step()

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            "total_steps": self.total_steps,
            "episodes": self.episode_count,
            "games": self.games,
            "wins": self.wins,
            "win_rate": self.wins / max(1, self.games),
        }

        # 添加 NFSP 统计
        nfsp_stats = self.nfsp.get_stats()
        stats.update(nfsp_stats)

        return stats

    def save(self, path: str):
        """保存模型"""
        self.nfsp.save(path)

    def load(self, path: str):
        """加载模型"""
        self.nfsp.load(path)

    def reset(self):
        """重置智能体状态（新回合开始时调用）"""
        self.current_episode_reward = 0.0

    def set_training_mode(self, training: bool):
        """设置训练模式"""
        self.is_training = training

        # 设置网络模式
        if training:
            self.nfsp.best_response_net.train()
            self.nfsp.average_policy_net.train()
        else:
            self.nfsp.best_response_net.eval()
            self.nfsp.average_policy_net.eval()

    def set_eta(self, eta: float):
        """设置 anticipatory 参数"""
        self.nfsp.set_eta(eta)


class NFSPAgentPool:
    """
    NFSP 智能体池

    管理多个 NFSP 智能体（用于自对弈）
    支持参数共享（4个玩家共享同一网络）
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        device: str = "cuda",
        num_agents: int = 4,
        share_parameters: bool = True,
    ):
        """
        初始化智能体池

        Args:
            config: 配置对象
            device: 计算设备
            num_agents: 智能体数量（4个玩家）
            share_parameters: 是否共享参数
        """
        self.config = config
        self.device = device
        self.num_agents = num_agents
        self.share_parameters = share_parameters

        if share_parameters:
            # 创建一个共享的 NFSP 实例
            self.shared_nfsp = NFSP(config, device)

            # 创建4个智能体，共享同一个 NFSP
            self.agents = [
                NFSPAgentWrapper(self.shared_nfsp, agent_id=i)
                for i in range(num_agents)
            ]
        else:
            # 每个智能体独立的 NFSP
            self.agents = [
                NFSPAgent(config, device, eta=config.nfsp.eta if config else 0.2)
                for i in range(num_agents)
            ]

        # 存储全局观测（用于集中式 critic 训练）
        self._global_observations = {}  # episode_num -> {agent_name: observation}

        # 创建 centralized buffer（用于 Phase 1-2）
        from .buffer import CentralizedRolloutBuffer

        if config is not None:
            rl_capacity = (
                config.nfsp.rl_buffer_size
                if hasattr(config.nfsp, "rl_buffer_size")
                else 10000
            )
        else:
            rl_capacity = 10000
        self.centralized_buffer = CentralizedRolloutBuffer(capacity=rl_capacity)

    def get_agent(self, agent_id: int) -> "NFSPAgentWrapper":
        """获取指定 ID 的智能体"""
        return self.agents[agent_id]

    def train_all(self, training_phase: int = 1) -> Dict:
        """
        训练所有智能体（参数共享时只训练一次）

        Args:
            training_phase: 训练阶段（1=全知，2=渐进，3=真实）

        Returns:
            训练统计字典
        """
        if self.share_parameters:
            # 传递 centralized buffer 给 NFSP.train_step()
            return self.shared_nfsp.train_step(
                training_phase=training_phase,
                centralized_buffer=self.centralized_buffer,
            )
        else:
            stats = {}
            for i, agent in enumerate(self.agents):
                agent_stats = agent.train_step()
                stats[f"agent_{i}"] = agent_stats
            return stats

    def get_average_policy_networks(self):
        """获取所有智能体的平均策略网络（用于对手）"""
        if self.share_parameters:
            return [self.shared_nfsp.get_average_policy_net()] * self.num_agents
        else:
            return [agent.nfsp.get_average_policy_net() for agent in self.agents]

    def save(self, path: str):
        """保存模型"""
        if self.share_parameters:
            self.shared_nfsp.save(path)
        else:
            for i, agent in enumerate(self.agents):
                agent.save(f"{path}_agent_{i}.pth")

    def load(self, path: str):
        """加载模型"""
        if self.share_parameters:
            self.shared_nfsp.load(path)
        else:
            for i, agent in enumerate(self.agents):
                agent.load(f"{path}_agent_{i}.pth")

    def store_global_observation(self, all_agents_observations, episode_info):
        """
        存储所有智能体的全局观测

        Args:
            all_agents_observations: Dict[str, Dict] - agent_name -> observation
            episode_info: Dict - 当前回合信息
        """
        self._global_observations[episode_info["episode_num"]] = all_agents_observations

    def get_global_observations(self, episode_num):
        """
        获取指定回合的所有智能体观测

        Args:
            episode_num: int - 回合编号

        Returns:
            Dict: agent_name -> observation
        """
        return self._global_observations.get(episode_num, {})


class NFSPAgentWrapper:
    """
    NFSP 智能体包装器

    用于参数共享场景，多个智能体共享同一个 NFSP 实例
    """

    def __init__(self, nfsp: NFSP, agent_id: int):
        self.nfsp = nfsp
        self.agent_id = agent_id
        self.total_steps = 0
        # 存储最后一次动作的训练信息
        self._last_log_prob = 0.0
        self._last_value = 0.0

    def choose_action(
        self, observation: Dict[str, np.ndarray], action_mask: np.ndarray
    ) -> Tuple[int, int]:
        """选择动作"""
        action_type, action_param, log_prob, value = self.nfsp.select_action(
            observation, action_mask
        )
        # 存储训练信息
        self._last_log_prob = log_prob
        self._last_value = value
        self.total_steps += 1
        return action_type, action_param

    def get_training_info(self) -> Tuple[float, float]:
        """
        获取最后一次动作的训练信息

        Returns:
            (log_prob, value): 动作的对数概率和价值估计
        """
        return self._last_log_prob, self._last_value

    def store_transition(self, *args, **kwargs):
        """存储转移"""
        self.nfsp.store_transition(*args, **kwargs)

    def end_episode(self, *args, **kwargs):
        """结束回合"""
        self.nfsp.end_episode(*args, **kwargs)

    def train_step(self):
        """训练步骤"""
        return self.nfsp.train_step()

    def get_stats(self):
        """获取统计"""
        return self.nfsp.get_stats()

    def save(self, path):
        """保存"""
        self.nfsp.save(path)

    def load(self, path):
        """加载"""
        self.nfsp.load(path)

    def reset(self):
        """重置"""
        pass


class RandomOpponent:
    """
    随机对手（用于前期训练）

    简单的随机策略，用于与 NFSP 智能体对弈
    """

    def __init__(self, action_mask_size: int = 145):
        self.action_mask_size = action_mask_size

    def choose_action(
        self, observation: Dict[str, np.ndarray], action_mask: np.ndarray
    ) -> Tuple[int, int]:
        """
        随机选择有效动作

        Args:
            observation: 观测（不使用）
            action_mask: 动作掩码（145位）

        Returns:
            (action_type, action_param) 元组
        """
        # 找到所有有效动作
        valid_indices = np.where(action_mask > 0)[0]

        if len(valid_indices) == 0:
            # 没有有效动作，返回 PASS
            return 10, -1  # PASS

        # 随机选择一个有效动作索引
        action_idx = random.choice(valid_indices)

        # 将索引转换为动作类型和参数
        action_type, action_param = self._index_to_action(action_idx)

        return action_type, action_param

    def _index_to_action(self, idx: int) -> Tuple[int, int]:
        """
        将动作掩码索引转换为动作类型和参数

        Args:
            idx: 动作掩码索引（0-144）

        Returns:
            (action_type, action_param)
        """
        if idx < 34:
            # DISCARD: 0-33
            return 0, idx
        elif idx < 37:
            # CHOW: 34-36
            return 1, idx - 34
        elif idx == 37:
            # PONG
            return 2, 0
        elif idx == 38:
            # KONG_EXPOSED
            return 3, 0
        elif idx < 73:
            # KONG_SUPPLEMENT: 39-72
            return 4, idx - 39
        elif idx < 107:
            # KONG_CONCEALED: 73-106
            return 5, idx - 73
        elif idx == 107:
            # KONG_RED
            return 6, 0
        elif idx < 142:
            # KONG_SKIN: 108-141
            return 7, idx - 108
        elif idx == 142:
            # KONG_LAZY
            return 8, 0
        elif idx == 143:
            # WIN
            return 9, -1
        else:
            # PASS: 144
            return 10, -1

    def reset(self):
        """重置（随机策略无需重置）"""
        pass
