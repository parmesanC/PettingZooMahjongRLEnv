# MADDPG/MAPPO 中心化 Critic 架构实现计划

## 问题分析

### 当前架构问题

**当前实现**：
- Critic 只基于当前智能体的自身观测估计价值
- 无法看到其他智能体的全局状态
- 不符合 MADDPG/MAPPO 算法要求

**正确架构要求**：
- 中心化 Critic 应该能够访问所有智能体的观测
- 需要全局信息进行更准确的价值估计
- 支持 MARL（多智能体强化学习）的协作

---

## 实施计划

### 阶段1：创建中心化网络组件

#### 1.1 创建 CentralizedCriticNetwork 类

**文件**: `src/drl/centralized_network.py`（新建）

**核心功能**：
- 接收所有智能体的全局观测
- 输出每个智能体的价值估计
- 使用共享的观测编码器

**实现细节**：
```python
class CentralizedCriticNetwork(nn.Module):
    def __init__(
        self,
        num_agents: int = 4,
        hidden_dim: int = 128,
        observation_encoder: nn.Module  # 共享的 ObservationEncoder
    ):
        super().__init__()
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.observation_encoder = observation_encoder

        # Critic 头部：可以处理所有智能体的全局观测
        # 输入维度 = hidden_dim * num_agents
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
        all_observations: Dict[str, torch.Tensor]  # 所有智能体的观测字典
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            all_observations: {
                'agent_0': obs,  # Agent 0 的全局观测
                'agent_1': obs,  # Agent 1 的全局观测
                'agent_2': obs,  # Agent 2 的全局观测
                'agent_3': obs,  # Agent 3 的全局观测
            }
        
        Returns:
            values: [batch, num_agents] - 每个智能体的价值估计
        """
        # 编码所有智能体的全局观测
        encoded_obs_list = []
        for i in range(self.num_agents):
            obs_key = f'agent_{i}'
            obs = all_observations[obs_key]
            encoded = self.observation_encoder(obs)
            encoded_obs_list.append(encoded)

        # 拼接所有编码观测
        all_encoded = torch.cat(encoded_obs_list, dim=-1)  # [batch, hidden_dim * num_agents]

        # Critic 前向传播
        values = self.critic(all_encoded)  # [batch, num_agents]

        return values
```

---

### 阶段2：创建中心化回放缓冲区

#### 2.1 创建 CentralizedRolloutBuffer 类

**文件**: `src/drl/centralized_buffer.py`（新建）

**核心功能**：
- 存储所有智能体的观测、动作、奖励
- 支持中心化 Critic 训练

**实现细节**：
```python
class CentralizedRolloutBuffer:
    def __init__(self, capacity: int = 100000, num_agents: int = 4):
        self.capacity = capacity
        self.num_agents = num_agents
        self.size = 0

        # 存储所有智能体的数据
        self.all_observations = []  # List[Dict[str, np.ndarray]]
        self.all_actions = []      # List[np.ndarray]
        self.all_rewards = []       # List[np.ndarray]
        self.all_values = []        # List[np.ndarray]
        self.dones = []             # List[np.ndarray]

    def add(self, all_observations, all_actions, all_rewards, dones):
        """添加一条经验"""
        if self.size >= self.capacity:
            self.all_observations.pop(0)
            self.all_actions.pop(0)
            self.all_rewards.pop(0)
            self.dones.pop(0)
        
        self.all_observations.append(all_observations)
        self.all_actions.append(all_actions)
        self.all_rewards.append(all_rewards)
        self.dones.append(dones)
        self.size += 1

    def get_batch(self, batch_size: int):
        """获取训练批次"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        batch_all_observations = [self.all_observations[i] for i in indices]
        batch_all_actions = [self.all_actions[i] for i in indices]
        batch_all_rewards = [self.all_rewards[i] for i in indices]
        batch_all_dones = [self.dones[i] for i in indices]
        
        return {
            'observations': batch_all_observations,
            'actions': batch_all_actions,
            'rewards': batch_all_rewards,
            'dones': batch_all_dones
        }
```

---

### 阶段3：修改训练器

#### 3.1 修改 _run_episode 收集全局观测

**文件**: `src/drl/trainer.py`（修改）

**修改内容**：
```python
def _run_episode(self) -> Dict:
    # 重置环境
    obs, _ = self.env.reset()
    
    # 回合数据
    episode_all_observations = {}  # 每个智能体的观测列表
    episode_all_actions = {}       # 每个智能体的动作列表
    episode_all_rewards = {}        # 每个智能体的奖励列表
    episode_all_dones = {}         # 每个智能体的终止标志
    
    # 初始化
    for i in range(4):
        episode_all_observations[i] = []
        episode_all_actions[i] = []
        episode_all_rewards[i] = []
        episode_all_dones[i] = []
    
    # PettingZoo 标准循环
    for agent_name in self.env.agent_iter():
        obs, reward, terminated, truncated, info = self.env.last()
        agent_idx = int(agent_name.split('_')[1])
        
        # 记录所有智能体的信息
        for i in range(4):
            episode_all_observations[i].append(obs)  # 全局观测
            if agent_name == f'agent_{i}':
                episode_all_actions[i].append((action_type, action_param))
                episode_all_rewards[i].append(reward)
                episode_all_dones[i].append(terminated or truncated)
            else:
                # 其他智能体记录占位符
                episode_all_actions[i].append((0, 0))  # 占位动作
                episode_all_rewards[i].append(0.0)
                episode_all_dones[i].append(False)
    
        # 选择动作
        action_type, action_param = self.agent_pool.get_agent(agent_idx).choose_action(
            obs['action_mask']
        )
        self.env.step((action_type, action_param))
    
    return {
        'all_observations': episode_all_observations,
        'all_actions': episode_all_actions,
        'all_rewards': episode_all_rewards,
        'all_dones': episode_all_dones
    }
```

#### 3.2 修改训练循环使用中心化 Critic

**文件**: `src/drl/trainer.py`（修改）

**修改内容**：
```python
def train(self):
    # 初始化中心化 Critic
    from .centralized_network import CentralizedCriticNetwork
    self.centralized_critic = CentralizedCriticNetwork(
        num_agents=4,
        hidden_dim=self.config.network.hidden_dim,
        observation_encoder=self.env.observation_encoder  # 共享编码器
    ).to(self.device)
    
    # 创建中心化优化器
    from .centralized_optimizer import CentralizedOptimizer
    self.centralized_optimizer = CentralizedOptimizer(
        self.centralized_critic,
        lr=self.config.mappo.lr
    )
    
    while self.episode_count < self.config.training.actual_total_episodes:
        # 运行一局
        episode_data = self._run_episode()
        
        # 存储到中心化缓冲区
        self.centralized_buffer.add(
            episode_data['all_observations'],
            episode_data['all_actions'],
            episode_data['all_rewards'],
            episode_data['all_dones']
        )
        
        # 训练 Actor（保持现有逻辑）
        for i, agent in enumerate(self.agents):
            agent.train_step()
        
        # 训练中心化 Critic
        if len(self.centralized_buffer) >= batch_size:
            batch = self.centralized_buffer.get_batch(batch_size)
            
            # 计算优势（GAE）
            values = self.centralized_critic(batch['observations'])
            # 使用所有智能体的奖励计算
            advantages = self.compute_advantages(batch, values)
            
            # 更新中心化 Critic
            self.centralized_optimizer.step(
                batch['observations'],
                batch['actions'],
                advantages
            )
```

---

### 阶段4：实现中心化优化器

#### 4.1 创建 CentralizedOptimizer 类

**文件**: `src/drl/centralized_optimizer.py`（新建）

**核心功能**：
- 实现中心化 Critic 的优化步骤
- 计算价值损失
- 更新网络参数

---

## 实施优先级

### 高优先级任务

1. ✅ 创建详细问题分析文档
2. 创建 `CentralizedCriticNetwork` 类
3. 创建 `CentralizedRolloutBuffer` 类
4. 修改训练器收集全局观测
5. 创建 `CentralizedOptimizer` 类
6. 修改训练循环使用中心化 Critic

### 中优先级任务

7. 测试中心化 Critic 功能
8. 验证训练流程
9. 性能测试和优化

---

## 技术挑战

### 1. 观测维度管理

**问题**：需要确保所有智能体的观测维度一致
**解决**：使用共享的 ObservationEncoder，确保输出格式统一

### 2. 训练稳定性

**问题**：中心化 Critic 训练可能不稳定
**解决**：使用梯度裁剪、学习率调度等技术

### 3. 性能优化

**问题**：全局观测处理会增加计算开销
**解决**：批处理优化、内存管理

---

## 时间估算

| 阶段 | 预计时间 |
|--------|----------|
| 问题分析 | 1 小时 |
| 创建中心化网络 | 4-6 小时 |
| 创建中心化缓冲区 | 2-3 小时 |
| 修改训练器 | 4-6 小时 |
| 创建中心化优化器 | 2-3 小时 |
| 测试验证 | 2-3 小时 |
| **总计** | **15-22 小时** |

---

## 参考资料

1. **MADDPG 论文**: Multi-Agent Deep Deterministic Policy Gradient
2. **MAPPO 论文**: Multi-Agent PPO
3. **OpenAI Baselines3**: Stable-Baselines3 实现参考
4. **PettingZoo 文档**: AECEnv 使用指南

---

## 注意事项

1. **向后兼容性**：保持与现有架构的兼容性
2. **渐进式迁移**：可以逐步从当前架构迁移到中心化架构
3. **测试优先**：每次修改后立即测试
4. **文档更新**：及时更新相关文档

---

**创建日期**: 2026-02-08
**状态**: 计划中，等待实施
**优先级**: 高 - 关键架构改进
