# 信念状态与Centralized Critic实现Review

**日期**: 2026-02-10
**Reviewer**: Claude Code
**计划文档**: `.sisyphus/plans/belief-state-centralized-critic-improved/bk.md`

---

## 1. 总体完成情况

### 1.1 已完成的核心组件

| 组件 | 文件 | 状态 | 完成度 |
|------|------|------|--------|
| BeliefNetwork | `belief_network.py` | ✅ 已实现 | 95% |
| MonteCarloSampler | `monte_carlo_sampler.py` | ⚠️ 有问题 | 70% |
| CentralizedCriticNetwork | `network.py` | ✅ 已实现 | 100% |
| CentralizedRolloutBuffer | `buffer.py` | ✅ 已实现 | 90% |
| MAPPO (Dual-Critic) | `mappo.py` | ⚠️ 有问题 | 80% |
| NFSP协调器 | `nfsp.py` | ✅ 已实现 | 90% |
| CurriculumScheduler | `curriculum.py` | ✅ 已实现 | 100% |
| NFSPTrainer | `trainer.py` | ⚠️ 有问题 | 75% |
| ActorCriticNetwork | `network.py` | ⚠️ 有问题 | 70% |
| 配置系统 | `config.py` | ✅ 已实现 | 100% |

### 1.2 对照计划的Task完成状态

| Task | 计划状态 | 实际状态 | 备注 |
|------|----------|----------|------|
| Task 0: 修复CentralizedCritic | ✅ | ✅ | 已完成 |
| Task 1: BeliefNetwork实现 | ✅ | ✅ | 已实现，需要测试 |
| Task 2a: BeliefNetwork辅助损失 | ✅ | ✅ | 已实现 |
| Task 2b: 全局状态构建器 | ✅ | ❓ | 未在dr/目录下找到，可能在observation/目录 |
| Task 3: 单元测试框架 | ✅ | ❌ | 缺失 |
| Task 3b: 代码质量基础设施 | ✅ | ✅ | 已创建 |
| Task 4: MonteCarloSampler实现 | - | ⚠️ | 有bug |
| Task 5: CentralizedCriticNetwork | ✅ | ✅ | 已实现 |
| Task 6: 修改Actor集成信念 | - | ⚠️ | 不完整 |
| Task 7: DualCriticTraining | ✅ | ⚠️ | 有重复代码 |
| Task 8: 环境集成全局状态 | ✅ | ❓ | 未完全验证 |
| Task 8a: Phase间Checkpoint热启动 | ✅ | ⚠️ | trainer中有基础实现 |
| Task 9: 训练流程验证 | - | ❌ | 未完成 |
| Task 10: 集成测试 | - | ❌ | 未完成 |

---

## 2. 详细代码问题分析

### 2.1 严重问题 (Critical)

#### 2.1.1 `network.py` - ActorCriticNetwork重复代码

**位置**: `src/drl/network.py:492-553`

**问题描述**:
`ActorCriticNetwork.forward()` 方法包含重复的代码片段：

```python
# 第一段: 492-526行 (包含信念集成的代码)
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
        # ... 信念集成代码 ...

# 第二段: 527-553行 (覆盖了前面的代码)
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
```

**影响**:
- 信念集成功能完全失效（被第二段代码覆盖）
- 第一段代码永远不会被执行

**修复建议**:
删除527-553行的重复代码，保留492-526行的完整实现。

---

#### 2.1.2 `ObservationEncoder` - 信念集成代码未完成

**位置**: `src/drl/network.py:384-401`

**问题描述**:
`ObservationEncoder.forward()` 中有未完成的信念集成代码：

```python
# 信念集成（可选）
if self.use_belief and belief_samples is not None:
    # 编码信念采样
    belief_features = []
    for sample in belief_samples:
        # 每个采样包含对手的观测字典
        sample_feat = self.encoder(sample)  # [batch, hidden_dim]
        belief_features.append(sample_feat)
```

**问题**:
1. `self.use_belief` 属性未在 `__init__` 中定义
2. `belief_samples` 参数未在函数签名中声明
3. 这段代码永远不会被执行

**修复建议**:
要么完成信念集成实现，要么删除这段代码。

---

#### 2.1.3 `mappo.py` - update()方法重复代码

**位置**: `src/drl/mappo.py:83-266`

**问题描述**:
`update()` 方法中存在严重的代码重复：

```python
# 第一段: 99-127行
# Phase-aware: 确定是否使用 centralized critic
use_centralized = (
    training_phase in [1, 2] and self.centralized_critic is not None
)

# Phase 1 和 2: 使用 centralized critic
if use_centralized and len(observations) > 0:
    # ...
    self.update_centralized(buffer, training_phase)

# 第二段: 128-150行 (重复计算 next_value 和 advantages)
# 计算下一价值
next_value = 0.0
# ... 几乎相同的代码 ...

# 计算回报和优势
advantages = buffer.compute_returns_and_advantages(
    self.gamma, self.gae_lambda, next_value
)
```

**影响**:
- 相同的计算被执行两次
- 逻辑混乱，难以维护

**修复建议**:
重构 `update()` 方法，消除重复代码。

---

#### 2.1.4 `trainer.py` - CentralizedBuffer填充未实现

**位置**: `src/drl/trainer.py:275-298`

**问题描述**:
```python
# [NEW] 填充 CentralizedRolloutBuffer（用于 Phase 1-2）
if self.current_phase in [1, 2]:
    # 需要收集每个step所有agents的数据
    # ... (大量注释)
    # 暂时跳过填充，因为当前架构限制
    pass  # ← 未实现!
```

**影响**:
- Centralized Critic训练无法正常工作
- Phase 1-2的训练效果受损

**修复建议**:
需要重构 `_run_episode()` 方法，在每个时间步收集所有agents的数据。

---

### 2.2 中等问题 (Medium)

#### 2.2.1 `monte_carlo_sampler.py` - 采样范围错误

**位置**: `src/drl/monte_carlo_sampler.py:121`

**问题描述**:
```python
# 更新对手手牌（仅更新索引为0,1,2的玩家）
for opp in range(1, 3):  # ← 只处理对手1和2，漏掉了对手0
    opp_idx = opp + 1  # 对手索引为1,2,3对应opponents 0,1,2
```

**影响**:
- 第一个对手（索引0）的手牌永远不会被采样
- 采样结果不完整

**修复建议**:
改为 `for opp in range(3):` 或 `for opp in range(num_opponents):`。

---

#### 2.2.2 `mappo.py` - CentralizedCritic未初始化

**位置**: `src/drl/mappo.py:37-59`

**问题描述**:
```python
def __init__(
    self,
    network,
    # ...
    centralized_critic=None,  # NEW: 添加 centralized_critic 支持
):
    # ...
    self.centralized_critic = (
        centralized_critic  # NEW: 添加 centralized_critic 支持
    )
```

**问题**:
- `centralized_critic` 参数传递了，但没有优化器
- CentralizedCritic的参数不会被训练

**修复建议**:
为CentralizedCritic创建单独的优化器，或将其参数添加到现有优化器中。

---

#### 2.2.3 `nfsp.py` - CentralizedCritic未传递给MAPPO

**位置**: `src/drl/nfsp.py:66-77`

**问题描述**:
```python
self.mappo = MAPPO(
    network=self.best_response_net,
    lr=config.mappo.lr,
    # ...
    device=device,
    # 缺少: centralized_critic=... 参数
)
```

**影响**:
- 即使创建了CentralizedCriticNetwork，也不会被使用

**修复建议**:
在NFSP初始化时创建并传递CentralizedCriticNetwork。

---

### 2.3 轻微问题 (Minor)

#### 2.3.1 `buffer.py` - CentralizedRolloutBuffer数据结构问题

**位置**: `src/drl/buffer.py:679-684`

**问题描述**:
```python
batch_all_observations = episode["all_observations"]  # [4, num_steps, Dict]
# ... 但实际格式可能是 [num_steps, 4, Dict]
```

**影响**:
- 数据格式假设可能不正确
- 可能导致训练时维度错误

---

#### 2.3.2 `belief_network.py` - 贝叶斯更新逻辑简化

**位置**: `src/drl/belief_network.py:178-204`

**问题描述**:
贝叶斯更新只做了简单的概率降低，没有考虑更复杂的因素。

**修复建议**:
这是可以接受的简化实现，但可能需要后续改进。

---

## 3. 缺失的实现

### 3.1 单元测试 (Task 3)

计划的测试文件：
```
tests/unit/
├── test_belief_network.py
├── test_global_observation.py
├── test_centralized_critic.py
├── test_monte_carlo_sampler.py
└── conftest.py
```

实际状态：❌ 未找到

### 3.2 集成测试 (Task 10)

计划的测试文件：
```
tests/integration/
├── test_belief_training.py
├── test_dual_critic.py
└── test_end_to_end.py
```

实际状态：⚠️ 部分存在（test_belief_training.py存在但可能不完整）

### 3.3 全局状态构建器 (Task 2b)

需要在 `src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py` 中验证。

### 3.4 TensorBoard集成 (Task 4a)

未实现。

---

## 4. 架构问题

### 4.1 数据流问题

**问题**: 当前episode循环使用PettingZoo的`agent_iter()`，在每个时间步只能看到一个agent的数据。

**影响**: 无法收集完整的多agent数据用于CentralizedCritic训练。

**修复建议**:
重构episode循环，在每个时间步收集所有agents的数据后再执行动作。

### 4.2 检查点迁移 (Task 8a)

当前实现只保存了episode和phase信息，没有实现真正的权重迁移逻辑（Phase 2→3时的Critic重新初始化）。

---

## 5. 优先修复建议

### P0 (立即修复)

1. **修复`network.py`中ActorCriticNetwork的重复代码**
   - 删除527-553行的重复代码

2. **修复`mappo.py`中update()方法的重复代码**
   - 重构以消除重复计算

3. **修复`monte_carlo_sampler.py`的采样范围**
   - 改为 `range(3)` 或 `range(num_opponents)`

4. **实现`trainer.py`中的CentralizedBuffer填充**
   - 重构 `_run_episode()` 以收集完整数据

### P1 (尽快修复)

5. **在NFSP中创建并传递CentralizedCriticNetwork**
6. **为CentralizedCritic创建优化器**
7. **完成ObservationEncoder的信念集成或删除未完成代码**

### P2 (可以延后)

8. **添加单元测试**
9. **添加集成测试**
10. **实现TensorBoard集成**
11. **完成Checkpoint热启动逻辑**

---

## 6. 与计划文档的差异

### 6.1 架构差异

计划中的架构：
```
BeliefNetwork → MonteCarloSampler → Actor集成 → CentralizedCritic → Dual-CriticTraining
```

实际实现：
```
BeliefNetwork ✅ → MonteCarloSampler ⚠️ → Actor集成 ❌ → CentralizedCritic ✅ → Dual-Critic ⚠️
```

### 6.2 功能差异

| 计划功能 | 实际状态 | 差异 |
|----------|----------|------|
| 蒙特卡罗采样(N=5-10) | ⚠️ 有bug | 采样范围错误 |
| 信念集成到Actor | ❌ 未完成 | 代码存在但未启用 |
| Phase-aware Critic切换 | ⚠️ 部分实现 | 切换逻辑存在但数据收集有问题 |
| 辅助损失训练 | ✅ 已实现 | 符合计划 |

---

## 7. 总结

### 7.1 完成度评估

- **核心组件**: 85% 完成
- **集成工作**: 60% 完成
- **测试验证**: 20% 完成
- **整体**: 70% 完成

### 7.2 关键发现

1. **大部分核心组件已实现**，但存在严重的集成问题
2. **代码重复**是最突出的问题（network.py, mappo.py）
3. **数据收集架构**需要重构以支持CentralizedCritic
4. **测试覆盖严重不足**，需要补充

### 7.3 下一步行动

1. 修复P0级别的代码问题
2. 重构数据收集流程
3. 添加单元测试
4. 进行端到端验证

---

**文档生成时间**: 2026-02-10
**审查范围**: `src/drl/` 目录下所有Python文件
**对照文档**: `.sisyphus/plans/belief-state-centralized-critic-improved/bk.md`
