# P0 严重问题修复计划

**日期**: 2026-02-10
**基于**: `.sisyphus/plans/belief-state-centralized-critic-improved/review_implementation.md`
**优先级**: P0 (立即修复)

---

## 概述

本计划修复 4 个严重问题，这些问题会导致核心功能失效：
1. `network.py` - 信念集成功能被覆盖
2. `mappo.py` - 重复计算导致逻辑混乱
3. `monte_carlo_sampler.py` - 第一个对手永远不被采样
4. `trainer.py` - CentralizedBuffer 永远不会被填充

---

## P0-1: 修复 `network.py` 中 ActorCriticNetwork 重复代码

### 问题描述

`ActorCriticNetwork.forward()` 方法存在严重的代码重复：
- 第 492-526 行：完整的 forward 实现（包含信念集成）
- 第 527-553 行：重复代码（覆盖了前面的实现）

**信念集成功能完全失效！**

### 代码位置

`src/drl/network.py:492-553`

```python
# 第一段：492-526 行（正确实现）
def forward(
    self,
    obs: Dict[str, torch.Tensor],
    action_mask: torch.Tensor,
    belief_samples: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 编码观测
    features = self.encoder(obs)  # [batch, hidden_dim]

    # 信念集成（可选）
    if self.use_belief and belief_samples is not None:
        # 编码信念采样
        belief_features = []
        for sample in belief_samples:
            # 每个采样包含对手的观测字典
            sample_feat = self.encoder(sample)  # [batch, hidden_dim]
            belief_features.append(sample_feat)

        # 平均所有采样的特征
        avg_belief_feat = torch.stack(belief_features, dim=0) / len(
            belief_features
        )  # [batch, hidden_dim]

        # 拼接到主特征
        combined = torch.cat([features, avg_belief_feat], dim=-1)
        features = combined

# 第二段：527-553 行（重复代码，覆盖了前面的内容）
"""
Args:
    obs: 观测字典
    action_mask: [batch, 145] 动作掩码
Returns:
    action_type_logits: [batch, 11]
    action_param_logits: [batch, 34]
    value: [batch, 1]
"""
# 编码观测
features = self.encoder(obs)  # [batch, hidden_dim]

# Actor 输出
action_type_logits = self.actor_type(features)  # [batch, 11]
action_param_logits = self.actor_param(features)  # [batch, 34]

# 应用动作掩码（在 softmax 前）
# 将 action_mask 转换为动作类型掩码和参数掩码
type_mask, param_mask = self._split_action_mask(action_mask)

# 对无效动作设置极小的 logits
action_type_logits = action_type_logits.masked_fill(~type_mask.bool(), -1e9)
action_param_logits = action_param_logits.masked_fill(~param_mask.bool(), -1e9)

# Critic 输出
value = self.critic(features)

return action_type_logits, action_param_logits, value
```

### 影响分析

- **信念集成功能完全失效**：即使设置了 `use_belief=True`，信念采样也不会被使用
- **第一段代码永远不会被执行**：被第二段覆盖
- **CTDE 架构核心功能损坏**：无法在 Phase 3 使用信念采样

### 修复步骤

#### 步骤 1: 定位重复代码范围
- 确认第 527-553 行是重复代码
- 确认第 492-526 行是正确实现

#### 步骤 2: 删除重复代码
删除第 527-553 行：
```python
"""
Args:
    obs: 观测字典
    action_mask: [batch, 145] 动作掩码
Returns:
    action_type_logits: [batch, 11]
    action_param_logits: [batch, 34]
    value: [batch, 1]
"""
# 编码观测
features = self.encoder(obs)  # [batch, hidden_dim]

# Actor 输出
action_type_logits = self.actor_type(features)  # [batch, 11]
action_param_logits = self.actor_param(features)  # [batch, 34]

# 应用动作掩码（在 softmax 前）
# 将 action_mask 转换为动作类型掩码和参数掩码
type_mask, param_mask = self._split_action_mask(action_mask)

# 对无效动作设置极小的 logits
action_type_logits = action_type_logits.masked_fill(~type_mask.bool(), -1e9)
action_param_logits = action_param_logits.masked_fill(~param_mask.bool(), -1e9)

# Critic 输出
value = self.critic(features)

return action_type_logits, action_param_logits, value
```

#### 步骤 3: 在第一段后添加 Actor 和 Critic 输出逻辑
在第 525 行（`features = combined`）之后，添加：

```python
# Actor 输出
action_type_logits = self.actor_type(features)  # [batch, 11]
action_param_logits = self.actor_param(features)  # [batch, 34]

# 应用动作掩码（在 softmax 前）
# 将 action_mask 转换为动作类型掩码和参数掩码
type_mask, param_mask = self._split_action_mask(action_mask)

# 对无效动作设置极小的 logits
action_type_logits = action_type_logits.masked_fill(~type_mask.bool(), -1e9)
action_param_logits = action_param_logits.masked_fill(~param_mask.bool(), -1e9)

# Critic 输出
value = self.critic(features)

return action_type_logits, action_param_logits, value
```

#### 步骤 4: 验证修复后的代码结构

修复后的 `forward()` 方法结构应该是：

```python
def forward(
    self,
    obs: Dict[str, torch.Tensor],
    action_mask: torch.Tensor,
    belief_samples: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        obs: 观测字典
        action_mask: [batch, 145] 动作掩码
        belief_samples: 采样的对手观测列表（可选）

    Returns:
        action_type_logits: [batch, 11]
        action_param_logits: [batch, 34]
        value: [batch, 1]
    """
    # 编码观测
    features = self.encoder(obs)  # [batch, hidden_dim]

    # 信念集成（可选）
    if self.use_belief and belief_samples is not None:
        # 编码信念采样
        belief_features = []
        for sample in belief_samples:
            # 每个采样包含对手的观测字典
            sample_feat = self.encoder(sample)  # [batch, hidden_dim]
            belief_features.append(sample_feat)

        # 平均所有采样的特征
        avg_belief_feat = torch.stack(belief_features, dim=0) / len(
            belief_features
        )  # [batch, hidden_dim]

        # 拼接到主特征
        combined = torch.cat([features, avg_belief_feat], dim=-1)
        features = combined

    # Actor 输出
    action_type_logits = self.actor_type(features)  # [batch, 11]
    action_param_logits = self.actor_param(features)  # [batch, 34]

    # 应用动作掩码（在 softmax 前）
    # 将 action_mask 转换为动作类型掩码和参数掩码
    type_mask, param_mask = self._split_action_mask(action_mask)

    # 对无效动作设置极小的 logits
    action_type_logits = action_type_logits.masked_fill(~type_mask.bool(), -1e9)
    action_param_logits = action_param_logits.masked_fill(~param_mask.bool(), -1e9)

    # Critic 输出
    value = self.critic(features)

    return action_type_logits, action_param_logits, value
```

### 验证方法

1. **语法检查**：
   ```bash
   python -m py_compile src/drl/network.py
   ```

2. **单元测试**：
   ```bash
   python tests/unit/test_actor_with_belief.py
   ```

3. **导入测试**：
   ```python
   from src.drl.network import ActorCriticNetwork
   net = ActorCriticNetwork(use_belief=True)
   print("✅ ActorCriticNetwork 实例化成功")
   ```

---

## P0-2: 修复 `mappo.py` 中 `update()` 方法重复代码

### 问题描述

`update()` 方法中存在重复计算：
- 第 99-127 行：首次计算 next_value 和调用 update_centralized()
- 第 128-150 行：重复计算 next_value 和调用 buffer.compute_returns_and_advantages()

**相同的计算被执行两次，逻辑混乱！**

### 代码位置

`src/drl/mappo.py:83-266`

### 影响分析

- **计算效率降低**：相同的值被计算两次
- **逻辑难以维护**：修改一处容易忘记另一处
- **潜在 Bug**：两次计算可能产生细微差异

### 修复步骤

#### 步骤 1: 分析现有逻辑
理解两部分代码的作用：
1. 第 99-127 行：Phase-aware 切换，Phase 1-2 使用 centralized critic
2. 第 128-150 行：计算 GAE 优势估计

#### 步骤 2: 统一 next_value 计算
将 next_value 的计算统一到一次：

```python
# 计算下一价值
next_value = 0.0
if next_obs is not None:
    with torch.no_grad():
        next_value = self.network.get_value(self._prepare_obs(next_obs))
```

#### 步骤 3: 重构 Phase-aware 逻辑
将 centralized critic 的调用整合到主流程中：

```python
# Phase-aware: 确定是否使用 centralized critic
use_centralized = (
    training_phase in [1, 2] and self.centralized_critic is not None
)

# 如果使用 centralized critic 且有数据
if use_centralized and len(observations) > 0:
    # 构建全局观测
    global_obs = observations[0]  # 假设 observations[0] 包含全局信息

    # 更新 centralized critic
    self.update_centralized(buffer, training_phase)
```

#### 步骤 4: 统一回报和优势计算
将 GAE 计算保留为一次调用：

```python
# 计算回报和优势
advantages = buffer.compute_returns_and_advantages(
    self.gamma, self.gae_lambda, next_value
)
```

### 验证方法

1. **语法检查**：
   ```bash
   python -m py_compile src/drl/mappo.py
   ```

2. **单元测试**：
   ```bash
   python tests/unit/test_dual_critic.py
   ```

---

## P0-3: 修复 `monte_carlo_sampler.py` 采样范围错误

### 问题描述

采样循环只处理对手 1 和 2，漏掉了对手 0：

```python
# 更新对手手牌（仅更新索引为 0, 1, 2 的玩家）
for opp in range(1, 3):  # ← 只处理对手 1 和 2，漏掉了对手 0
    opp_idx = opp + 1  # 对手索引为 1, 2, 3 对应 opponents 0, 1, 2
```

### 代码位置

`src/drl/monte_carlo_sampler.py:121`

### 影响分析

- **第一个对手的手牌永远不会被采样**：采样结果不完整
- **信念状态估计不准确**：缺少一个对手的信息

### 修复步骤

#### 步骤 1: 定位问题代码
找到第 121 行的循环。

#### 步骤 2: 修改循环范围
将 `range(1, 3)` 改为 `range(3)`：

```python
# 更新对手手牌（更新所有 3 个对手）
for opp in range(3):  # ← 处理所有对手
    opp_idx = opp  # 对手索引为 0, 1, 2
    # 更新对手的手牌...
```

或者更健壮的实现：
```python
# 更新对手手牌（更新所有对手）
for opp_idx in range(self.num_opponents):  # ← 更灵活
    # 更新对手的手牌...
```

### 验证方法

1. **语法检查**：
   ```bash
   python -m py_compile src/drl/monte_carlo_sampler.py
   ```

2. **单元测试**：
   ```bash
   python tests/unit/test_monte_carlo_sampler.py
   ```

3. **功能测试**：
   ```python
   from src.drl.monte_carlo_sampler import MonteCarloSampler
   sampler = MonteCarloSampler()
   # 验证采样包含所有对手
   ```

---

## P0-4: 实现 `trainer.py` 中 CentralizedBuffer 填充逻辑

### 问题描述

`trainer.py` 中 CentralizedBuffer 填充逻辑未实现：

```python
# [NEW] 填充 CentralizedRolloutBuffer（用于 Phase 1-2）
if self.current_phase in [1, 2]:
    # 需要收集每个 step 所有 agents 的数据
    # ... (大量注释)
    # 暂时跳过填充，因为当前架构限制
    pass  # ← 未实现！
```

### 代码位置

`src/drl/trainer.py:275-298`

### 影响分析

- **Centralized Critic 训练无法正常工作**：buffer 永远是空的
- **Phase 1-2 的训练效果受损**：无法使用全局状态训练

### 修复步骤

#### 步骤 1: 分析当前 episode 循环

理解当前的 `_run_episode()` 方法如何收集数据：
- 当前使用 PettingZoo 的 `agent_iter()`
- 每个时间步只能看到一个 agent 的数据

#### 步骤 2: 修改 episode 循环结构

重构 `_run_episode()` 以在每个时间步收集所有 agents 的数据：

```python
def _run_episode(self) -> Dict:
    """运行一局自对弈（PettingZoo 标准模式）"""
    # 重置环境
    obs, _ = self.env.reset()

    # 确定对手类型
    if self.episode_count < self.config.training.switch_point:
        # 前期：使用随机对手
        use_random_opponents = True
    else:
        # 后期：使用历史策略
        use_random_opponents = False

    # [NEW] CentralizedBuffer 数据收集
    episode_all_observations = []  # [num_steps, 4, Dict]
    episode_all_actions_type = []   # [num_steps, 4]
    episode_all_actions_param = []  # [num_steps, 4]
    episode_all_rewards = []       # [num_steps, 4]
    episode_all_values = []        # [num_steps, 4]

    # 回合数据
    episode_rewards_total = [0.0] * 4
    episode_steps = 0
    winner = None

    # Episode 循环
    while not self.env.done:
        # 在每个时间步收集所有 agents 的数据
        step_all_observations = {}
        step_all_actions_type = {}
        step_all_actions_param = {}
        step_all_rewards = {}
        step_all_values = {}

        for agent in self.env.agent_iter():
            agent_id = int(agent.split("_")[-1])

            # 获取观测和动作掩码
            agent_obs = obs[agent]
            action_mask = self.env.infos[agent].get("action_mask", np.ones(145))

            # 选择动作
            if use_random_opponents:
                action_type, action_param = self.random_opponent.choose_action(
                    agent_obs, action_mask
                )
            else:
                action_type, action_param = self.agent_pool.select_action(
                    agent_id, agent_obs, action_mask
                )

            # 计算价值
            _, _, _, value = self.agent_pool.nfsp.select_action(
                agent_obs, action_mask
            )

            # 存储这一步的数据
            step_all_observations[agent] = agent_obs
            step_all_actions_type[agent] = action_type
            step_all_actions_param[agent] = action_param
            step_all_values[agent] = value

            # 执行动作（需要等待所有 agents 选择动作）
            # ... (PettingZoo 的机制需要特殊处理)

        # 执行所有 agents 的动作
        # 这里需要处理 PettingZoo 的并行动作执行
        # 具体实现取决于环境接口

        # 收集所有 rewards
        for agent in self.env.agents:
            episode_all_rewards[agent] = self.env.rewards[agent]

        # 将这一步的数据添加到 episode 列表
        episode_all_observations.append(step_all_observations)
        episode_all_actions_type.append(step_all_actions_type)
        episode_all_actions_param.append(step_all_actions_param)
        episode_all_values.append(step_all_values)

        episode_steps += 1

    # [NEW] 填充 CentralizedBuffer
    if self.current_phase in [1, 2]:
        self.agent_pool.centralized_buffer.add_episode(
            observations=episode_all_observations,
            actions_type=episode_all_actions_type,
            actions_param=episode_all_actions_param,
            rewards=episode_all_rewards,
            values=episode_all_values,
        )

    return {...}
```

#### 步骤 3: 实现 CentralizedBuffer.add_episode() 方法

在 `src/drl/buffer.py` 中添加：

```python
def add_episode(
    self,
    observations: List[Dict],
    actions_type: List[Dict],
    actions_param: List[Dict],
    rewards: List[Dict],
    values: List[Dict],
):
    """
    添加完整 episode 数据到 centralized buffer

    Args:
        observations: [num_steps, 4, Dict] 每个时间步每个 agent 的观测
        actions_type: [num_steps, 4] 每个时间步每个 agent 的动作类型
        actions_param: [num_steps, 4] 每个 agent 的动作参数
        rewards: [num_steps, 4] 每个 agent 的奖励
        values: [num_steps, 4] 每个 agent 的价值估计
    """
    num_steps = len(observations)
    num_agents = 4

    # 展平数据：[num_steps * num_agents, ...]
    all_observations = []
    all_actions_type = []
    all_actions_param = []
    all_rewards = []
    all_values = []

    for step in range(num_steps):
        for agent_idx in range(num_agents):
            agent_name = f"player_{agent_idx}"
            all_observations.append(observations[step][agent_name])
            all_actions_type.append(actions_type[step][agent_name])
            all_actions_param.append(actions_param[step][agent_name])
            all_rewards.append(rewards[step].get(agent_name, 0.0))
            all_values.append(values[step][agent_name])

    # 添加到 buffer
    self.observations.extend(all_observations)
    self.actions_type.extend(all_actions_type)
    self.actions_param.extend(all_actions_param)
    self.rewards.extend(all_rewards)
    self.values.extend(all_values)
```

### 验证方法

1. **语法检查**：
   ```bash
   python -m py_compile src/drl/trainer.py
   python -m py_compile src/drl/buffer.py
   ```

2. **集成测试**：
   ```bash
   python tests/integration/test_dual_critic.py
   ```

---

## 执行顺序

推荐按以下顺序执行修复：

1. **P0-1**: 修复 network.py（影响核心功能）
2. **P0-3**: 修复 monte_carlo_sampler.py（简单，快速验证）
3. **P0-2**: 重构 mappo.py（较复杂，需要仔细）
4. **P0-4**: 实现 CentralizedBuffer 填充（需要重构 trainer.py）

---

## 验证计划

所有修复完成后：

1. **语法检查**: 所有文件通过 `python -m py_compile`
2. **导入测试**: 所有模块可以正常导入
3. **单元测试**: 所有单元测试通过
4. **功能测试**: 运行快速训练验证

---

**文档生成时间**: 2026-02-10
**下一步**: 运行 `/start-work` 开始执行修复
