"""
NFSP 智能体封装

提供与麻将环境交互的接口，继承自 PlayerStrategy 基类。
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional, List
import random
import os

from src.mahjong_rl.agents.base import PlayerStrategy
from src.drl.nfsp import NFSP
from src.drl.config import Config
from src.drl.network import ActorCriticNetwork, AveragePolicyNetwork


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

    def select_actions_batch(
        self,
        obs_batch: List[Dict[str, np.ndarray]],
        action_mask_batch: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        批量动作选择（用于向量化训练）

        Args:
            obs_batch: 观测字典列表
            action_mask_batch: 动作掩码列表

        Returns:
            actions_type: [batch_size] numpy 数组
            actions_param: [batch_size] numpy 数组
            log_probs: [batch_size] numpy 数组
            values: [batch_size] numpy 数组
        """
        if self.share_parameters:
            # 共享参数模式：使用共享 NFSP 进行批量选择
            return self.shared_nfsp.select_actions_batch(obs_batch, action_mask_batch)
        else:
            # 非共享参数模式：降级逐个处理（不推荐，但提供兼容性）
            import warnings
            warnings.warn(
                "使用非共享参数模式的批量动作选择，性能将下降。"
                "建议使用 share_parameters=True 以获得最佳性能。"
            )

            batch_size = len(obs_batch)
            actions_type = np.zeros(batch_size, dtype=np.int64)
            actions_param = np.zeros(batch_size, dtype=np.int64)
            log_probs = np.zeros(batch_size, dtype=np.float32)
            values = np.zeros(batch_size, dtype=np.float32)

            for i in range(batch_size):
                obs = obs_batch[i]
                mask = action_mask_batch[i]
                agent = self.agents[i]
                action_type, action_param = agent.choose_action(obs, mask)

                actions_type[i] = action_type
                actions_param[i] = action_param
                log_probs[i] = 0.0  # choose_action 不返回 log_prob
                values[i] = 0.0    # choose_action 不返回 value

            return actions_type, actions_param, log_probs, values

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

        # 验证：确保选择的动作确实在 mask 中（防御性检查）
        # 注意：action_type 不是展平索引，需要重新映射验证
        if action_type == 10:  # PASS
            if action_mask[144] != 1:
                # PASS 不可用，返回第一个有效动作
                action_idx = valid_indices[0]
                action_type, action_param = self._index_to_action(action_idx)
        elif action_type == 0:  # DISCARD
            if action_mask[action_param] != 1:
                # 选择的牌不可用，返回第一个有效动作
                action_idx = valid_indices[0]
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
        # 边界检查：防御性编程
        if idx < 0 or idx > 144:
            print(f"⚠️ RandomOpponent: 无效动作索引 {idx}，回退到 PASS")
            return 10, -1  # PASS

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


class HistoricalPolicyOpponent:
    """
    历史策略对手

    只包含推理所需的网络权重，不包含训练逻辑。
    用于 NFSP 自对弈期，从策略池加载的历史策略。
    """

    def __init__(self, policy_path: str, device: str = "cuda"):
        """
        初始化历史策略对手

        Args:
            policy_path: 策略文件路径（.pth）
            device: 计算设备
        """
        self.device = device
        self.policy_path = policy_path
        self.episode_number = self._extract_episode_number(policy_path)

        # 创建网络（只用于推理，不包含训练组件）
        # 注意：需要从配置或文件中读取网络架构参数
        # 这里使用默认参数，实际加载时会从文件中恢复权重
        self.best_response_net = ActorCriticNetwork(
            observation_space=None,  # 从文件恢复
            action_space=None,       # 从文件恢复
            hidden_dim=256,
            dropout=0.1,
        ).to(device)

        self.average_policy_net = AveragePolicyNetwork(
            observation_space=None,
            action_space=None,
            hidden_dim=256,
            dropout=0.1,
        ).to(device)

        # 加载权重
        self._load_weights()

        # 设置为评估模式
        self.best_response_net.eval()
        self.average_policy_net.eval()

    def _extract_episode_number(self, policy_path: str) -> int:
        """从策略文件路径提取 episode 编号"""
        import re
        match = re.search(r'policy_(\d+)\.pth', policy_path)
        if match:
            return int(match.group(1))
        return 0

    def _load_weights(self):
        """从 .pth 文件加载网络权重"""
        try:
            checkpoint = torch.load(self.policy_path, map_location=self.device)

            # 加载 BR 网络
            if 'best_response_net' in checkpoint:
                self.best_response_net.load_state_dict(checkpoint['best_response_net'])
            elif 'model_state_dict' in checkpoint:
                self.best_response_net.load_state_dict(checkpoint['model_state_dict'])

            # 加载 π̄ 网络
            if 'average_policy_net' in checkpoint:
                self.average_policy_net.load_state_dict(checkpoint['average_policy_net'])

        except Exception as e:
            print(f"[警告] 加载策略失败 {self.policy_path}: {e}")
            raise

    def choose_action(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: np.ndarray,
        eta: float = 0.2
    ) -> Tuple[int, int]:
        """
        选择动作（根据 η 概率混合 BR 和 π̄）

        Args:
            obs: 观测字典
            action_mask: 动作掩码
            eta: 使用 BR 的概率（默认 0.2）

        Returns:
            (action_type, action_param) 元组
        """
        import torch.nn.functional as F

        # 决定使用哪个网络
        use_br = random.random() < eta

        # 准备观测张量
        obs_tensor = self._prepare_obs(obs)
        action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if use_br:
                # 使用最佳响应网络
                action_type, action_param, _, _, _ = (
                    self.best_response_net.get_action_and_value(
                        obs_tensor, action_mask_tensor
                    )
                )
                action_type = action_type.item()
                action_param = action_param.item()
            else:
                # 使用平均策略网络
                type_probs, param_probs = self.average_policy_net.get_action_probs(
                    obs_tensor, action_mask_tensor
                )

                # 采样动作
                type_dist = torch.distributions.Categorical(type_probs)
                param_dist = torch.distributions.Categorical(param_probs)

                action_type = type_dist.sample().item()
                action_param = param_dist.sample().item()

        return action_type, action_param

    def _prepare_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """将观测字典转换为张量字典"""
        tensor_obs = {}
        for key, value in obs.items():
            if isinstance(value, dict):
                tensor_obs[key] = self._prepare_obs(value)
            elif isinstance(value, np.ndarray):
                tensor_val = torch.FloatTensor(value).to(self.device)
                if tensor_val.dim() == 1:
                    tensor_val = tensor_val.unsqueeze(0)
                elif tensor_val.dim() == 0:
                    tensor_val = tensor_val.unsqueeze(0).unsqueeze(0)
                tensor_obs[key] = tensor_val
            else:
                tensor_val = torch.FloatTensor([value]).unsqueeze(0).to(self.device)
                tensor_obs[key] = tensor_val
        return tensor_obs


class PolicyPoolManager:
    """
    策略池管理器

    负责策略的添加、加载和采样。维护已加载到内存的历史策略对手列表。
    """

    def __init__(self, pool_size: int = 10, device: str = "cuda"):
        """
        初始化策略池管理器

        Args:
            pool_size: 最大策略数量（默认 10）
            device: 计算设备
        """
        self.pool_size = pool_size
        self.device = device
        self.policies: List[HistoricalPolicyOpponent] = []
        self.policy_paths: List[str] = []

    def add_policy(self, policy_path: str) -> bool:
        """
        添加新策略到池中

        如果池已满，移除最旧的策略。

        Args:
            policy_path: 策略文件路径

        Returns:
            是否成功添加
        """
        try:
            # 加载策略到内存
            opponent = HistoricalPolicyOpponent(policy_path, self.device)

            # 添加到池
            self.policies.append(opponent)
            self.policy_paths.append(policy_path)

            # 超过容量时移除最旧的
            if len(self.policies) > self.pool_size:
                old_opponent = self.policies.pop(0)
                old_path = self.policy_paths.pop(0)

                # 清理文件
                if os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except Exception as e:
                        print(f"[警告] 无法删除旧策略文件 {old_path}: {e}")

                # 释放内存
                del old_opponent

            return True

        except Exception as e:
            print(f"[错误] 添加策略失败 {policy_path}: {e}")
            return False

    def sample_opponent(self) -> Optional[HistoricalPolicyOpponent]:
        """
        随机采样一个对手策略

        Returns:
            随机选择的历史策略对手，池为空时返回 None
        """
        if not self.policies:
            return None
        return random.choice(self.policies)

    def is_empty(self) -> bool:
        """检查策略池是否为空"""
        return len(self.policies) == 0

    def size(self) -> int:
        """返回当前策略数量"""
        return len(self.policies)
