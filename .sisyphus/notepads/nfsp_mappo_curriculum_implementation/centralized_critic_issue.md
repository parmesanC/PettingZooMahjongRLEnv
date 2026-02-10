# MADDPG/MAPPO 中心化 Critic 架构问题分析

## 问题描述

**当前实现只使用自身观测训练 Critic，无法利用其他玩家信息进行更准确的价值估计**

---

## 当前架构

### 1. 参数共享结构（agent.py）

```python
class NFSPAgentPool:
    def __init__(self, share_parameters=True):
        if share_parameters:
            # 创建一个共享的 NFSP 实例
            self.shared_nfsp = NFSP(config, device)

            # 创建4个智能体，共享同一个 NFSP
            self.agents = [
                NFSPAgentWrapper(self.shared_nfsp, agent_id=i)
                for i in range(num_agents)
            ]
```

### 2. Critic 架构（network.py）

```python
class ActorCriticNetwork(nn.Module):
    def __init__(self):
        # Critic 头部
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出 [batch, 1] 的价值
        )
    
    def forward(self, obs, action_mask):
        features = self.encoder(obs)  # [batch, hidden_dim]
        value = self.critic(features)  # [batch, 1]
        return action_type_logits, action_param_logits, value
```

### 3. 观测编码器（network.py）

```python
class ObservationEncoder(nn.Module):
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 编码全局状态（所有玩家观测）
        state = torch.cat([
            current_player.float(),
            remaining_tiles.float(),
            fan_counts.float(),        # 所有玩家番数
            current_phase.float(),
            special_indicators.float(),
            dealer.float()
        ], dim=-1)
        
        return self.net(state)  # [batch, hidden_dim]
```

**注意**: `ObservationEncoder` 可以编码全局信息，但实际使用取决于传入的 obs。

---

## 问题分析

### 问题所在：Critic 只看到自身观测

#### 当前流程

```python
# 1. 训练循环（trainer.py）
for agent_name in self.env.agent_iter():
    obs, reward, terminated, truncated, info = self.env.last()
    agent_idx = int(agent_name.split('_')[1])
    
    # obs 是全局观测（所有玩家信息）✅
    
    # 2. Agent 选择动作（agent.py）
def choose_action(self, observation, action_mask):
    action_type, action_param, log_prob, value = self.nfsp.select_action(
        observation,  # 传递全局观测 ✅
        action_mask
    )
    
    # 3. 存储转移（agent.py）
if self.is_training:
    self.last_obs = observation  # 存储全局观测 ✅
    # self.nfsp.store_transition() 会调用 _prepare_obs(obs)

# 4. NFSP 存储转移（nfsp.py）
def store_transition(self, obs, ...):
    obs_tensor = self._prepare_obs(obs)  # ⚠️ 只处理当前玩家观测
    
# 5. Buffer 存储转移（buffer.py）
self.buffer.observations.append(obs)  # ⚠️ 只存储当前玩家观测

# 6. MAPPO 训练（mappo.py）
def train_step(self, buffer):
    obs_batch = self._prepare_obs_batch(buffer.observations)  # ⚠️ 只包含当前玩家观测
    
    # 前向传播
    action_type_logits, action_param_logits, values = self.network(
        obs_batch,  # ⚠️ Critic 只看到当前玩家信息
        action_masks
    )
    
    # 计算价值损失
    value_loss = nn.functional.mse_loss(
        values.squeeze(),
        returns
    )
```

#### 核心问题

| 步骤 | 当前实现 | 问题 |
|------|---------|------|
| 1. 环境观测 | `env.last()` 返回全局观测 ✅ | 无问题 |
| 2. Agent 接收 | `choose_action(obs)` 接收全局观测 ✅ | 无问题 |
| 3. Agent 存储 | `last_obs = observation` 存储全局观测 ✅ | 无问题 |
| 4. NFSP 准备 | `_prepare_obs(obs)` 只处理当前玩家观测 ⚠️ | **问题** |
| 5. Buffer 存储 | `buffer.observations.append(obs)` 存储当前玩家观测 ⚠️ | **问题** |
| 6. MAPPO 训练 | `_prepare_obs_batch()` 只包含当前玩家观测 ⚠️ | **问题** |

---

## 正确的 MADDPG/MAPPO 架构

### 架构要求

在多智能体强化学习中，**集中式价值函数** 应该能够访问所有智能体的观测信息，从而学习更准确的价值估计。

### 正确实现方式

#### 1. 中心化 Critic

```python
class CentralizedCriticNetwork(nn.Module):
    """
    中心化 Critic - 可以看到所有智能体的全局观测
    """
    def __init__(self, num_agents=4, hidden_dim=128):
        super().__init__()
        
        # 输入：所有智能体的全局观测拼接
        input_dim = hidden_dim * num_agents
        
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents)  # 输出每个智能体的价值
        )
    
    def forward(
        self,
        all_observations: List[Dict[str, torch.Tensor]]  # 所有智能体的观测
    ) -> torch.Tensor:
        """
        Args:
            all_observations: 所有智能体的观测列表
                [
                    obs_agent0,  # Agent 0 的观测
                    obs_agent1,  # Agent 1 的观测
                    obs_agent2,  # Agent 2 的观测
                    obs_agent3,  # Agent 3 的观测
                ]
        
        Returns:
            [batch, num_agents] 的价值估计
        """
        # 编码所有智能体的观测
        encoded_states = []
        for obs in all_observations:
            encoded = self.encoder(obs)  # [hidden_dim]
            encoded_states.append(encoded)
        
        # 拼接所有编码状态
        all_states = torch.cat(encoded_states, dim=-1)  # [batch, hidden_dim * num_agents]
        
        # Critic 输出每个智能体的价值
        values = self.critic(all_states)  # [batch, num_agents]
        
        return values
```

#### 2. 改进的 Buffer

```python
class CentralizedRolloutBuffer:
    """
    中心化回放缓冲区 - 存储所有智能体的观测
    """
    def __init__(self, capacity: int, num_agents: int):
        self.capacity = capacity
        self.num_agents = num_agents
        
        # 存储所有智能体的数据
        self.all_observations = []  # 所有智能体的观测列表
    
    def add(self, all_observations, all_actions, all_rewards, ...):
        """
        添加一条经验
        
        Args:
            all_observations: 所有智能体的观测列表
            all_actions: 所有智能体的动作列表
            all_rewards: 所有智能体的奖励列表
            ...
        """
        self.all_observations.append(all_observations)
        self.all_actions.append(all_actions)
        self.all_rewards.append(all_rewards)
        # ... 其他数据
    
    def get_batch(self, batch_size):
        """
        获取训练批次
        """
        # 返回所有智能体的数据
        return {
            'observations': self.all_observations[indices],  # [batch, num_agents, ...]
            'actions': self.all_actions[indices],
            'rewards': self.all_rewards[indices],
            # ...
        }
```

#### 3. 改进的训练循环

```python
def train_step_centralized(self, buffer):
    """
    使用中心化 Critic 训练
    """
    # 1. 获取批次数据
    batch_data = buffer.get_batch(batch_size)
    all_observations = batch_data['observations']  # [batch, num_agents, ...]
    all_actions = batch_data['actions']
    all_rewards = batch_data['rewards']
    
    # 2. Critic 前向传播
    values = self.centralized_critic(all_observations)  # [batch, num_agents]
    
    # 3. 计算每个智能体的优势
    advantages = []
    for i in range(self.num_agents):
        agent_values = values[:, i]  # [batch]
        agent_rewards = all_rewards[:, i]  # [batch]
        agent_advantages = compute_gae(agent_rewards, agent_values)
        advantages.append(agent_advantages)
    
    # 4. 训练 Actor
    for i in range(self.num_agents):
        agent_obs = all_observations[:, i]  # [batch, ...]
        agent_actions = all_actions[:, i]
        agent_advantages = advantages[i]
        
        # 训练 agent i 的 Actor
        loss_i = train_actor(agent_obs, agent_actions, agent_advantages)
    
    # 5. 训练 Critic
    value_loss = compute_value_loss(values, returns)
    
    total_loss = actor_loss + value_coef * value_loss
```

---

## 影响分析

### 当前实现的问题

1. **价值估计不准确**
   - Critic 只看到自身信息
   - 无法考虑其他玩家的状态
   - 导致价值估计偏差

2. **协调性差**
   - 多智能体缺乏全局视角
   - 难以实现有效合作

3. **非最优策略**
   - 在零和游戏中，每个 agent 只考虑自己
   - 无法学习全局最优策略

### 改进后的优势

1. **更准确的价值估计**
   - Critic 可以看到所有玩家信息
   - 价值估计考虑全局状态
   - 训练更稳定

2. **更好的协调性**
   - Critic 提供全局价值
   - Agent 可以学习考虑全局状态

3. **符合 MADDPG/MAPPO 规范**
   - 真正的集中式算法
   - 更好的理论保证

---

## 实现建议

### 选项 1：最小化修改（推荐）

只修改训练部分，不改变网络结构：

1. **修改 trainer.py**
   - 在训练循环中收集所有智能体的观测
   - 一次性传递给训练函数

2. **修改 mappo.py**
   - 修改 `train_step` 接受所有智能体观测
   - 重写 Critic 前向传播

3. **修改 buffer.py**
   - 添加存储所有智能体观测的方法
   - 保持与当前实现的兼容性

### 选项 2：完整重写

创建新的中心化 Critic 架构：

1. **创建 centralized_network.py**
   - 实现 `CentralizedCriticNetwork`
   - 输入：所有智能体的全局观测
   - 输出：每个智能体的价值

2. **修改 mappo.py**
   - 使用新的中心化 Critic
   - 调整训练逻辑

3. **修改 buffer.py**
   - 创建 `CentralizedRolloutBuffer`
   - 存储所有智能体的数据

---

## 参考资源

1. **MADDPG 论文**: Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
2. **MAPPO 论文**: Multi-Agent PPO
3. **OpenAI Baselines**: https://github.com/openai/baselines
4. **Stable-Baselines3**: https://github.com/DLR-RM/stable-baselines3

---

## 优先级

| 优先级 | 任务 | 预计工作量 |
|--------|------|----------|
| 高 | 修改 trainer.py 收集所有观测 | 2-4 小时 |
| 高 | 修改 mappo.py 训练逻辑 | 4-6 小时 |
| 高 | 创建中心化 Critic 网络 | 3-5 小时 |
| 中 | 测试和验证 | 2-3 小时 |
| 中 | 文档更新 | 1-2 小时 |

---

**总预计工作量**: 12-20 小时

---

## 注意事项

1. **向后兼容性**
   - 确保新实现不影响现有功能
   - 提供渐进迁移选项

2. **性能考虑**
   - 中心化 Critic 会增加计算开销
   - 优化批处理逻辑

3. **测试策略**
   - 先在小型环境测试
   - 逐步扩展到完整环境

4. **监控指标**
   - 记录 Critic 性能指标
   - 对比改进前后的性能

---

**创建日期**: 2026-02-08
**问题确认**: 是，当前实现存在此问题
**建议**: 尽快实施中心化 Critic 架构
