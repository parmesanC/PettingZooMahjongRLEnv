# NFSP 完整对手选择实现设计

## 概述

修复当前 NFSP 训练中的对手选择问题，实现真正的 Neural Fictitious Self-Play。

## 问题背景

### 当前问题
1. **随机对手期过长**：Phase 1 的 15% 时间（100万/666万局）在跟随机对手对战
2. **策略池未使用**：保存了历史策略但从未加载使用
3. **不是真正的自对弈**：后期只有当前策略，没有从策略池采样对手
4. **switch_point 与课程学习脱节**：固定 100 万局切换，不考虑课程学习阶段

### 设计目标
- 探索期（0-20万局）：使用随机对手进行探索
- 自对弈期（20万-2000万局）：从策略池随机选择历史策略作为对手
- 真正实现 NFSP 的对手选择机制

## 架构设计

### 新增组件

#### 1. HistoricalPolicyOpponent 类
```python
class HistoricalPolicyOpponent:
    """
    历史策略对手

    只包含推理所需的网络权重，不包含训练逻辑
    """
    def __init__(self, policy_path: str, device: str = "cuda")
    def choose_action(self, obs, action_mask, eta: float = 0.2) -> Tuple[int, int]
```

**特点**：
- 只加载推理网络（BR + π̄）
- 不包含缓冲区、优化器等训练组件
- 使用 `torch.no_grad()` 避免计算图
- 保留 η 参数用于混合 BR 和 π̄

#### 2. PolicyPoolManager 类
```python
class PolicyPoolManager:
    """策略池管理器"""
    def __init__(self, pool_size: int = 10, device: str = "cuda")
    def add_policy(self, policy_path: str) -> bool
    def sample_opponent(self) -> Optional[HistoricalPolicyOpponent]
    def is_empty(self) -> bool
    def size(self) -> int
```

**职责**：
- 维护已加载的 `HistoricalPolicyOpponent` 列表
- 自动管理内存和文件清理
- 提供随机采样接口

### 修改组件

#### 1. NFSPTrainer
- 用 `PolicyPoolManager` 替换 `policy_pool` 列表
- `exploration_episodes` = 200,000（替代 `switch_point`）
- 自对弈期为每个位置预选对手

#### 2. TrainingConfig
```python
# 新增字段
exploration_episodes: int = 200_000  # 探索期
policy_pool_size: int = 10  # 策略池大小
policy_save_interval: int = 1000  # 保存间隔

# 废弃字段（保留兼容）
switch_point: int = 200_000  # 使用 exploration_episodes
```

## 训练流程

### 阶段划分

| 阶段 | Episode 数量 | 对手选择 |
|------|-------------|---------|
| 探索期 | 0 - 200,000 | RandomOpponent |
| 自对弈期 | 200,000 - 20,000,000 | 从策略池随机选择 |

### 自对弈期对手选择逻辑

```
1. 每局游戏开始时，为 4 个位置分别预选对手：
   for i in range(4):
       opponent = policy_pool_manager.sample_opponent()
       if opponent is None:
           episode_opponents[i] = None  # 降级使用当前策略
       else:
           episode_opponents[i] = opponent

2. 游戏循环中，根据预选结果选择动作：
   if opponent is None:
       使用当前 NFSP 策略
   else:
       使用历史策略 opponent.choose_action(obs, mask, eta=0.2)
```

### 策略池管理

```
1. 每 1000 局保存当前策略
2. PolicyPoolManager 自动加载到内存
3. 池满（10个）时自动移除最旧的
4. 同时删除磁盘上的旧文件
```

## 数据流图

```
训练开始
    │
    ├─> PolicyPoolManager 初始化（空池）
    │
    └─> Episode 0 - 199,999（探索期）
           └─> 使用 RandomOpponent
           │
           └─> Episode 200,000（进入自对弈期）
                  ├─> 保存第 200 个策略
                  ├─> PolicyPoolManager 加载到内存
                  └─> 开始从池采样对手
                  │
                  └─> Episode 200,000 - 20,000,000（自对弈期）
                         ├─> 每 1000 局保存并加载新策略
                         ├─> 每个位置独立采样对手
                         └─> 池满时自动移除最旧的
```

## 边界情况处理

### 1. 策略池为空
- 在自对弈期初期，策略池可能还没有足够策略
- 降级策略：使用当前 NFSP 策略

### 2. 多个位置选择相同策略
- 允许发生，这是 NFSP 的正常行为
- 增加对特定历史策略的针对性训练

### 3. 内存管理
- 超过池大小时自动删除旧的 HistoricalPolicyOpponent
- 使用 `del` 显式释放内存
- 同时清理磁盘文件

## 实现文件

### 新增文件
- `src/drl/agent.py` - 添加 `HistoricalPolicyOpponent` 和 `PolicyPoolManager` 类

### 修改文件
- `src/drl/trainer.py` - 修改 `NFSPTrainer` 类
- `src/drl/config.py` - 修改 `TrainingConfig` 数据类

## 兼容性

- 保留 `switch_point` 字段作为废弃字段
- 旧代码可以继续工作，但会使用新逻辑
- 添加迁移指南注释

## 预期效果

1. **探索期缩短**：从 100 万局减少到 20 万局
2. **真正的自对弈**：对手从策略池随机选择
3. **策略多样性**：保持 10 个历史策略
4. **训练效率提升**：Phase 1 可以充分利用全知信息学习强策略

## 版本

- 创建日期：2025-02-11
- 设计者：Claude
- 状态：设计完成，待实施
