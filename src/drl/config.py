"""
NFSP + MAPPO + Transformer 配置

超参数配置，针对武汉麻将环境优化
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NetworkConfig:
    """网络架构配置"""
    # Transformer 配置
    transformer_layers: int = 4
    hidden_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1
    
    # MLP 编码器配置
    hand_encoder_dim: int = 64
    discard_encoder_dim: int = 32
    meld_encoder_dim: int = 64
    state_encoder_dim: int = 32
    
    # Actor-Critic 配置
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 128
    
    # 动作空间
    num_action_types: int = 11  # ActionType 枚举数量
    num_action_params: int = 34  # 牌的数量
    action_mask_size: int = 145  # 动作掩码总长度


@dataclass
class NFSPConfig:
    """NFSP 算法配置"""
    # Anticipatory 参数
    eta: float = 0.2  # 使用最佳响应网络的概率
    
    # 缓冲区配置
    rl_buffer_size: int = 100_000  # RL 缓冲区大小
    sl_buffer_size: int = 2_000_000  # SL 缓冲区大小（Reservoir Buffer）
    
    # 训练频率
    rl_train_freq: int = 1  # 每多少步训练一次 RL
    sl_train_freq: int = 1  # 每多少步训练一次 SL
    
    # 批次大小
    rl_batch_size: int = 64
    sl_batch_size: int = 64


@dataclass
class MAPPOConfig:
    """MAPPO 算法配置"""
    # PPO 参数
    ppo_epochs: int = 4  # 每次更新的 epoch 数
    clip_ratio: float = 0.2  # PPO 裁剪比率
    
    # GAE 参数
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE lambda
    
    # 损失系数
    value_coef: float = 0.5  # 价值损失系数
    entropy_coef: float = 0.01  # 熵奖励系数
    
    # 优化参数
    max_grad_norm: float = 0.5  # 梯度裁剪
    
    # 学习率
    lr: float = 3e-4


@dataclass
class TrainingConfig:
    """训练配置"""
    # 训练规模
    total_episodes: int = 5_000_000  # 总局数
    switch_point: int = 1_000_000  # 切换对手的局数（前期随机→后期历史）
    
    # 评估配置
    eval_interval: int = 1000  # 每多少局评估一次
    eval_games: int = 100  # 每次评估对战多少局
    
    # 保存配置
    save_interval: int = 10_000  # 每多少局保存一次模型
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # 设备配置
    device: str = "cuda"  # 使用 GPU
    num_workers: int = 4  # 数据加载并行数
    
    # 随机种子
    seed: Optional[int] = 42


@dataclass
class MahjongConfig:
    """麻将环境配置"""
    # 环境参数
    num_players: int = 4
    training_phase: int = 3  # 信息可见度阶段
    
    # 观测空间维度（来自 Wuhan7P4LObservationBuilder）
    global_hand_dim: int = 4 * 34  # 4个玩家的手牌
    private_hand_dim: int = 34
    discard_pool_dim: int = 34
    wall_dim: int = 82
    meld_action_types_dim: int = 16
    meld_tiles_dim: int = 256
    meld_group_indices_dim: int = 32
    action_history_types_dim: int = 80
    action_history_params_dim: int = 80
    action_history_players_dim: int = 80
    special_gangs_dim: int = 12  # 4玩家 × 3种特殊杠
    fan_counts_dim: int = 4
    special_indicators_dim: int = 2


@dataclass
class Config:
    """总配置"""
    network: NetworkConfig = None
    nfsp: NFSPConfig = None
    mappo: MAPPOConfig = None
    training: TrainingConfig = None
    mahjong: MahjongConfig = None
    
    def __post_init__(self):
        if self.network is None:
            self.network = NetworkConfig()
        if self.nfsp is None:
            self.nfsp = NFSPConfig()
        if self.mappo is None:
            self.mappo = MAPPOConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.mahjong is None:
            self.mahjong = MahjongConfig()


def get_default_config() -> Config:
    """获取默认配置"""
    return Config()


def get_quick_test_config() -> Config:
    """获取快速测试配置（小规模）"""
    config = Config()
    
    # 缩小网络规模
    config.network.transformer_layers = 2
    config.network.hidden_dim = 128
    config.network.num_heads = 2
    
    # 减小缓冲区
    config.nfsp.rl_buffer_size = 10_000
    config.nfsp.sl_buffer_size = 100_000
    
    # 减少训练量
    config.training.total_episodes = 10_000
    config.training.switch_point = 2_000
    config.training.eval_interval = 100
    config.training.save_interval = 1_000
    
    return config
