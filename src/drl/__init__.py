"""
DRL (Deep Reinforcement Learning) 模块

NFSP + MAPPO + Transformer 麻将智能体

主要组件：
- network: 网络架构（Transformer + Actor-Critic + Average Policy）
- mappo: MAPPO 算法实现
- nfsp: NFSP 协调器（整合 RL + SL）
- buffer: 经验回放缓冲区
- agent: 智能体封装
- trainer: 训练循环
- config: 配置管理
- curriculum: 课程学习调度器

使用示例：
    from src.drl import NFSPAgent, train_nfsp, CurriculumScheduler

    # 创建智能体
    agent = NFSPAgent()

    # 训练
    train_nfsp(quick_test=True)
"""

from .config import (
    Config,
    NetworkConfig,
    NFSPConfig,
    MAPPOConfig,
    TrainingConfig,
    MahjongConfig,
    get_default_config,
    get_quick_test_config,
)


from .network import (
    ActorCriticNetwork,
    AveragePolicyNetwork,
    ObservationEncoder,
    TransformerHistoryEncoder,
    create_networks,
    init_weights,
)

from .buffer import RolloutBuffer, ReservoirBuffer, EpisodeBuffer, MixedBuffer

from .buffer import RolloutBuffer, ReservoirBuffer, EpisodeBuffer, MixedBuffer

from .buffer import RolloutBuffer, ReservoirBuffer, EpisodeBuffer, MixedBuffer

from .mappo import MAPPO, SupervisedLearning

from .nfsp import NFSP

from .agent import NFSPAgent, NFSPAgentPool, NFSPAgentWrapper, RandomOpponent

from .trainer import NFSPTrainer, train_nfsp

from .curriculum import CurriculumScheduler

__version__ = "1.0.0"
__author__ = "汪呜呜"

__all__ = [
    # 配置
    "Config",
    "NetworkConfig",
    "NFSPConfig",
    "MAPPOConfig",
    "TrainingConfig",
    "MahjongConfig",
    "get_default_config",
    "get_quick_test_config",
    # 网络
    "ActorCriticNetwork",
    "AveragePolicyNetwork",
    "ObservationEncoder",
    "TransformerHistoryEncoder",
    "create_networks",
    "init_weights",
    # 缓冲区
    "RolloutBuffer",
    "ReservoirBuffer",
    "EpisodeBuffer",
    "MixedBuffer",
    # 算法
    "MAPPO",
    "SupervisedLearning",
    "NFSP",
    # 智能体
    "NFSPAgent",
    "NFSPAgentPool",
    "NFSPAgentWrapper",
    "RandomOpponent",
    # 课程学习
    "CurriculumScheduler",
    # 训练器
    "NFSPTrainer",
    "train_nfsp",
]
