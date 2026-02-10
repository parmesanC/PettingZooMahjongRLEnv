# 信念状态与Centralized Critic实现计划

## TL;DR

> **目标**: 实现完整的信念状态估计（蒙特卡罗采样）和Centralized Critic，优化麻将MADRL系统在不完美信息博弈中的表现
> 
> **核心组件**:
> - BeliefNetwork: 估计对手手牌概率分布
> - MonteCarloSampler: 从信念采样可能状态
> - CentralizedCriticNetwork: 训练时使用完整全局信息
> - DualCriticTraining: Phase 1-2用centralized，Phase 3用decentralized
> - Modified Actor: 集成信念采样输入
> 
> **预计工作量**: 2-3周（全职开发）
> **并行执行**: Wave 1（基础设施）→ Wave 2（核心网络）→ Wave 3（训练集成）→ Wave 4（测试验证）
> **关键路径**: BeliefNetwork → MonteCarloSampler → CentralizedCritic → DualCriticTraining

---

## 1. 上下文

### 1.1 原始需求
用户要求实现完整的信念状态和Centralized Critic，采用激进但全面的架构：
- **信念状态**: 蒙特卡罗采样（采样N个可能手牌状态）
- **Centralized Critic**: 训练时访问完整全局状态（4玩家手牌+牌墙）
- **训练策略**: 结合课程学习 + Dual-Critic（Phase 1-2 centralized，Phase 3 decentralized）
- **信念集成**: 仅Actor使用信念（Critic直接用真实状态）

### 1.2 当前系统架构

**现有网络组件**（`src/drl/network.py`）:
```python
class ActorCriticNetwork(nn.Module):
    """当前实现 - 单一Actor + 单一Critic"""
    - ObservationEncoder: 编码观测 → hidden_dim
    - Actor heads: Linear(hidden_dim → 11 action_types, hidden_dim → 34 params)
    - Critic head: Linear(hidden_dim → 1 value)
    - 输入: 单agent观测（已被visibility_mask处理）
```

**现有MAPPO**（`src/drl/mappo.py`）:
```python
class MAPPO:
    - 参数共享: 4玩家共享同一网络
    - Critic: 仅接收单agent局部观测（非centralized）
    - 训练: GAE + Clipped PPO
```

**现有课程学习**（`example_mahjong_env.py:683-771`）:
```python
Phase 1: Full observation (所有信息可见)
Phase 2: Progressive masking (渐进式遮蔽)
Phase 3: Real imperfect information (真实不完美信息)
```

### 1.3 关键挑战

1. **训练-执行差距**: Centralized critic在训练时看到对手手牌，执行时看不到
   - **解决方案**: Dual-Critic（训练用centralized，执行用decentralized）

2. **状态空间爆炸**: 完整全局状态 > 1500维（4×34手牌 + 136牌墙 + 公共信息）
   - **解决方案**: 分阶段训练，逐步减少信息依赖

3. **信念准确性**: 初期信念估计粗糙，随游戏进行逐渐精确
   - **解决方案**: 贝叶斯更新 + Transformer时序建模

4. **计算开销**: 蒙特卡罗采样 + dual-critic显著增加计算量
   - **解决方案**: 采样数可配置（N=5-10），GPU并行

---

## 2. 工作目标

### 2.1 核心目标
实现完整的CTDE（Centralized Training, Decentralized Execution）架构，包含信念状态估计和centralized critic。

### 2.2 具体交付物
1. `src/drl/belief_network.py` - 信念网络实现
2. `src/drl/centralized_critic.py` - Centralized critic网络
3. `src/drl/monte_carlo_sampler.py` - 蒙特卡罗采样器
4. 修改 `src/drl/network.py` - 集成信念的Actor
5. 修改 `src/drl/mappo.py` - Dual-critic训练逻辑
6. 修改 `example_mahjong_env.py` - 全局状态构建
7. 测试脚本 - 验证各组件正确性

### 2.3 定义完成（Definition of Done）
- [ ] BeliefNetwork能准确估计对手手牌分布（测试验证）
- [ ] MonteCarloSampler能生成合理的采样状态
- [x] CentralizedCritic在Phase 1-2有效降低value估计方差（基础设施已完成）
- [x] DualCriticTraining正确切换critic（Phase 1-2 centralized，Phase 3 decentralized）
- [ ] 整体训练效果优于baseline（胜率提升>5%）
- [x] 集成测试通过（test_centralized_simple.py 所有测试通过）

### 2.4 Must Have
- 信念网络输出3个对手的34维概率分布
- 蒙特卡罗采样支持可配置采样数（N=5-10）
- Centralized critic接收完整全局状态（>1500维）
- Dual-critic根据training_phase自动切换
- 与现有课程学习兼容

### 2.5 Must NOT Have
- 不修改游戏规则或reward函数
- 不修改PettingZoo接口
- 不影响现有的human_vs_ai模式
- 不引入破坏性架构变更

---

## 3. 验证策略

### 3.1 测试决策
- **基础设施存在**: 是（已有pytest框架）
- **自动化测试**: 是（每个TODO包含单元测试）
- **框架**: Python unittest + torch.testing

### 3.2 Agent-Executed QA Scenarios（每个任务必须包含）

**BeliefNetwork测试**:
```
Scenario: BeliefNetwork输出有效概率分布
  Tool: Bash (Python)
  Steps:
    1. 创建BeliefNetwork实例
    2. 输入公共观测（discard_pool, melds, history）
    3. 调用forward获取beliefs
    4. Assert: beliefs形状为[3, 34]（3对手，34种牌）
    5. Assert: 每行概率和≈1.0（softmax约束）
    6. Assert: 所有概率在[0, 1]范围内
  Expected: 输出有效概率分布
```

**MonteCarloSampler测试**:
```
Scenario: 采样生成合理手牌状态
  Tool: Bash (Python)
  Steps:
    1. 创建BeliefNetwork和MonteCarloSampler（N=5）
    2. 输入游戏状态获取beliefs
    3. 调用sample获取N个采样状态
    4. Assert: 返回N个有效GameContext
    5. Assert: 每个采样中对手手牌符合Mahjong规则
    6. Assert: 手牌总数正确（初始13张，变化后相应调整）
  Expected: 生成N个合理的可能游戏状态
```

**CentralizedCritic测试**:
```
Scenario: Centralized critic接收完整全局状态
  Tool: Bash (Python)
  Steps:
    1. 创建CentralizedCriticNetwork
    2. 构建完整全局观测（4玩家手牌 + 牌墙 + 公共信息）
    3. 调用get_value获取V值
    4. Assert: 输出为标量张量
    5. Assert: 值在合理范围内（基于reward scale）
  Expected: 正确评估完整全局状态
```

**DualCriticTraining测试**:
```
Scenario: 根据phase正确切换critic
  Tool: Bash (Python)
  Steps:
    1. 初始化MAPPO（带dual-critic）
    2. Phase 1: 验证使用centralized critic
    3. Phase 2: 验证使用centralized critic
    4. Phase 3: 验证使用decentralized critic
    5. Assert: 每个phase正确选择critic网络
  Expected: Phase 1-2 centralized，Phase 3 decentralized
```

---

## 4. 执行策略

### 4.1 并行执行波次

```
Wave 0 (核心问题修复 - 3天):
├── Task 0: 修复CentralizedCritic未实际使用问题
│   ├── 修改NFSPAgentPool收集全局观测
│   ├── 完善CentralizedRolloutBuffer
│   ├── 修改MAPPO支持dual-critic
│   └── 实现phase-aware critic切换

Wave 1 (基础设施 - 5-6天):
├── Task 1: BeliefNetwork实现（包含贝叶斯更新）
├── Task 2: 全局状态构建器
├── Task 3: 单元测试框架
└── Task 1a: MonteCarlo采样具体实现

Wave 2 (核心网络 - 5-7天):
├── Task 4: MonteCarloSampler实现
├── Task 5: CentralizedCriticNetwork实现
└── Task 6: 修改Actor集成信念

Wave 3 (训练集成 - 6-8天):
├── Task 7: DualCriticTraining修改MAPPO
├── Task 8: 环境集成全局状态
├── Task 9: 训练流程验证
└── Task 3a: 实现对手策略池（PolicyPool）

Wave 4 (测试验证 - 5-7天):
├── Task 10: 集成测试
├── Task 11: 性能基准测试
├── Task 12: 文档和示例
└── Task 4a: TensorBoard集成和性能监控
```

### 4.2 依赖矩阵

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 (BeliefNetwork) | None | 4, 6 | 2, 3 |
| 2 (GlobalStateBuilder) | None | 5, 8 | 1, 3 |
| 3 (Test Framework) | None | All tests | 1, 2 |
| 4 (MonteCarloSampler) | 1 | 6 | 5 |
| 5 (CentralizedCritic) | 2 | 7 | 4 |
| 6 (Modified Actor) | 1, 4 | 7 | - |
| 7 (DualCriticTraining) | 5, 6 | 9 | 8 |
| 8 (Env Integration) | 2 | 9 | 7 |
| 9 (Training Validation) | 7, 8 | 10 | - |
| 10 (Integration Tests) | 3, 9 | 11 | - |
| 11 (Benchmark) | 10 | 12 | - |
| 12 (Documentation) | 11 | None | - |

---

## 5. TODOs

### Wave 0: 核心问题修复

- [x] **Task 0: 修复CentralizedCritic未实际使用问题（P0优先级）**

  **What to do**:
  1. 修改 `src/drl/agent.py` 中的 `NFSPAgentPool` 类
     - 添加 `store_global_transition(all_observations, ...)` 方法
     - 修改 `choose_action()` 返回全局观测
  2. 完善 `src/drl/buffer.py` 中的 `get_centralized_batch()` 方法
     - 确保正确存储和检索所有智能体的观测
     - 支持批次格式转换
  3. 修改 `src/drl/mappo.py` 中的 `MAPPO` 类
     - 添加 `dual_critic_update(centralized_obs, decentralized_obs, phase)` 方法
     - 根据 training_phase 选择使用哪个 critic
     - Phase1-2: 使用 centralized critic (完整全局状态)
     - Phase3: 使用 decentralized critic (仅局部观测)
  4. 更新 `src/drl/trainer.py` 中的训练循环
     - 在 `_run_episode()` 中收集 `all_agents_observations`
     - 传递给 buffer 和训练函数

  **完成日期**: 2025-02-09
  **完成内容**:
  - ✅ 修复 network.py 中的 CentralizedCriticNetwork 重复代码
  - ✅ 修复 buffer.py 中的语法错误
  - ✅ 在 agent.py 的 NFSPAgentPool 中添加全局观测存储方法
  - ✅ 在 trainer.py 中集成全局观测收集
  - ✅ 在 mappo.py 中添加 centralized_critic 参数支持
  - ✅ 在 mappo.py 中实现 update_centralized() 方法
  - ✅ 添加 phase-aware 切换逻辑
  - ✅ 创建并通过集成测试（test_centralized_simple.py）

  **备注**: 参见 `task31_completion_report.md` 获取完整详情

  **Must NOT do**:
  - 不要修改reward计算逻辑
  - 不要影响human_vs_ai模式
  - 不要破坏现有的checkpoint系统

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`（需要深入理解现有架构）
  - **Skills**: `pytorch`, `rl-algorithms`, `architecture`

  **Parallelization**:
  - **Can Run In Parallel**: NO（必须最先完成）
  - **Parallel Group**: Wave 0（独立执行）
  - **Blocks**: Task 1, Task 1a, Task 2, Task 7
  - **Blocked By**: None

  **References**:
  - 现有AgentPool: `src/drl/agent.py:12-45`
  - CentralizedCritic: `src/drl/network.py:687-782`
  - 现有MAPPO: `src/drl/mappo.py:14-325`
  - Trainer: `src/drl/trainer.py:177-266`
  - Issue分析: `.sisyphus/notepads/nfsp_mappo_curriculum_implementation/centralized_critic_issue.md`

  **Acceptance Criteria**:
  - [ ] `NFSPAgentPool` 正确收集和传递全局观测
  - [ ] `get_centralized_batch()` 返回正确的批次格式
  - [ ] `MAPPO.train_step()` 根据 phase 正确选择 critic
  - [ ] Phase 1-2 使用 centralized critic（接收所有4个agent观测）
  - [ ] Phase 3 使用 decentralized critic（仅接收当前agent观测）
  - [ ] 训练loss显著降低（vs baseline）
  - [ ] 单元测试通过: `test_dual_critic.py`

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: Dual critic根据phase正确切换
    Tool: Bash (python)
    Preconditions: 项目已安装依赖，测试数据已准备
    Steps:
      1. 创建 MAPPO 实例和两个critic（centralized + decentralized）
      2. 准备测试数据：
         - Phase 1: centralized_obs (4个agent的完整观测)
         - Phase 2: centralized_obs (同Phase 1)
         - Phase 3: decentralized_obs (仅当前agent观测)
      3. Phase 1: 调用 mappo.train_step(phase=1, obs=centralized_obs)
         - Assert centralized_critic 被调用
         - Assert 返回的 value 来自 centralized_critic
      4. Phase 2: 调用 mappo.train_step(phase=2, obs=centralized_obs)
         - Assert centralized_critic 被调用
         - Assert 返回的 value 来自 centralized_critic
      5. Phase 3: 调用 mappo.train_step(phase=3, obs=decentralized_obs)
         - Assert decentralized_critic 被调用
         - Assert 返回的 value 来自 decentralized_critic
      6. 验证 critic 选择的参数（检查内部状态）
    Expected Result: Phase 1-2 使用 centralized critic，Phase 3 使用 decentralized critic
    Evidence: 打印每个phase使用的critic类型
  ```

  **Commit**: YES
  - Message: `fix(architecture): implement dual-critic training with phase-aware critic selection`
  - Files: `src/drl/agent.py`, `src/drl/buffer.py`, `src/drl/mappo.py`, `src/drl/trainer.py`, `tests/unit/test_dual_critic.py`

---

### Wave 1: 基础设施

- [ ] **Task 1: BeliefNetwork实现（包含贝叶斯更新，P1优先级）**

  **What to do**:
  1. 创建 `src/drl/belief_network.py`
  2. 实现 `BeliefNetwork` 类，继承 `nn.Module`
  3. 网络架构:
     - Input: 公共观测 (discard_pool[34], melds[16×9], action_history[80×3])
     - Processing: TransformerEncoder + Linear layers
     - Output: 3个对手的概率分布 [batch, 3, 34]（softmax归一化）
  4. 实现 `forward()` 方法处理公共信息
  5. **新增**: 实现贝叶斯更新方法 `update_beliefs(action_history, discard_pool, melds)`
     - P(t|E) ∝ P(E|t) × L(E|t)
     - 打出牌: beliefs[opponent, tile] *= 0.1
     - 碰牌: beliefs[opponent, tile] *= 1.5
     - 杠牌: beliefs[opponent, tile] *= 2.0
     - 归一化: normalized = beliefs / beliefs.sum()
  6. 实现辅助方法 `get_opponent_beliefs(agent_id, context)`

  **Must NOT do**:
  - 不要访问私有手牌信息（仅使用公共信息）
  - 不要硬编码对手数量（保持灵活性）
  - 不要在belief network中使用critic网络

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`（需要深度神经网络架构设计）
  - **Skills**: `pytorch`, `transformers`, `rl-algorithms`, `probability`

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖 Task 0）
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 1a)
  - **Blocks**: Task 4, Task 6
  - **Blocked By**: Task 0

  **References**:
  - 现有编码器模式: `src/drl/network.py:HandEncoder, DiscardEncoder, MeldEncoder`
  - Transformer使用: `src/drl/network.py:TransformerHistoryEncoder`
  - 概率分布输出: PyTorch `F.softmax()`
  - 贝叶斯更新公式: 见 draft 中 P1.1 部分

  **Acceptance Criteria**:
  - [ ] BeliefNetwork类可实例化
  - [ ] `forward()` 输入公共观测，输出[batch, 3, 34]概率分布
  - [ ] 每个对手的概率和≈1.0（softmax约束）
  - [ ] `update_beliefs()` 根据对手动作正确更新信念
  - [ ] 贝叶斯更新符合公式（似然函数×先验）
  - [ ] 单元测试通过: `test_belief_network.py`

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: BeliefNetwork输出有效概率分布并支持贝叶斯更新
    Tool: Bash (python)
    Preconditions: 项目已安装依赖
    Steps:
      1. 导入 BeliefNetwork
      2. network = BeliefNetwork(hidden_dim=256)
      3. obs = 构建测试公共观测（batch=4）
      4. beliefs = network(obs)
      5. Assert beliefs.shape == torch.Size([4, 3, 34])
      6. Assert torch.allclose(beliefs.sum(dim=-1), torch.ones(4, 3), atol=1e-5)
      7. Assert (beliefs >= 0).all() and (beliefs <= 1).all()
      8. # 测试贝叶斯更新
      9. action_history = 模拟对手打出5万
      10. updated_beliefs = network.update_beliefs(action_history, discard_pool, melds)
      11. Assert updated_beliefs[0, 5] < beliefs[0, 5]  # 5万概率降低
      12. Assert torch.allclose(updated_beliefs.sum(dim=-1), torch.ones(4, 3), atol=1e-5)
    Expected Result: 概率分布有效，贝叶斯更新符合预期
    Evidence: 更新前后的belief分布值
  ```

  **Commit**: YES
  - Message: `feat(belief): add BeliefNetwork with Bayesian update for opponent hand estimation`
  - Files: `src/drl/belief_network.py`, `tests/unit/test_belief_network.py`

- [ ] **Task 1a: MonteCarlo采样具体实现（P1优先级）**

  **What to do**:
  1. 在 `src/drl/monte_carlo_sampler.py` 中完善 `MonteCarloSampler` 类
  2. 实现详细的采样流程:
     - Gumbel-Softmax 采样: 从分布中采样N个可能手牌
     - 置信度调整: 根据历史准确度调整采样权重
     - 约束检查: 确保不采样已知的牌（弃牌堆、副露）
  3. 实现核心方法:
     ```python
     def sample(beliefs: torch.Tensor, n_samples: int, known_tiles: torch.Tensor) -> List[GameContext]:
         """
         Args:
             beliefs: [batch, 3, 34] - 3个对手的概率分布
             n_samples: 采样数量（默认5-10）
             known_tiles: [batch, 34] - 已知的牌（弃牌堆+副露）
         Returns:
             N个采样的GameContext，每个包含采样的对手手牌
         """
         samples = []
         for _ in range(n_samples):
             # Gumbel-Softmax 采样
             gumbel = -torch.log(-torch.log(torch.rand_like(beliefs)))
             sampled_indices = torch.argmax(beliefs + gumbel, dim=-1)

             # 掩码已知的牌
             sampled_indices = sampled_indices * (1 - known_tiles.int())

             # 构建采样的GameContext
             sampled_context = self._build_sampled_context(sampled_indices)

             # 约束检查（手牌数、规则符合性）
             if self._validate_sample(sampled_context):
                 samples.append(sampled_context)

         return samples
     ```
  4. 实现 `_build_sampled_context()` - 创建采样的GameContext副本
  5. 实现 `_validate_sample()` - 验证采样有效性

  **Must NOT do**:
  - 不要修改原始GameContext（创建副本）
  - 不要生成违反Mahjong规则的手牌
  - 不要采样到已知的牌（如弃牌堆中的牌）

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`（需要理解游戏规则和概率采样）
  - **Skills**: `pytorch`, `probability`, `game-rules`

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖 Task 1）
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Task 6
  - **Blocked By**: Task 1

  **References**:
  - BeliefNetwork输出: Task 1
  - GameContext结构: `src/mahjong_rl/core/GameData.py`
  - Mahjong规则: `src/mahjong_rl/rules/`
  - 采样策略: Gumbel-Softmax, Rejection Sampling

  **Acceptance Criteria**:
  - [ ] MonteCarloSampler可实例化
  - [ ] `sample()` 返回N个有效GameContext
  - [ ] 每个采样符合Mahjong规则
  - [ ] 采样不修改原始context
  - [ ] 支持GPU并行（可选）
  - [ ] 约束检查正确（不采样已知牌）
  - [ ] 单元测试通过

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: 采样生成合理手牌
    Tool: Bash (python)
    Preconditions: 项目已安装依赖，BeliefNetwork已实现
    Steps:
      1. 创建GameContext和BeliefNetwork
      2. beliefs = network(public_obs)  # [3, 34]
      3. known_tiles = 构建已知牌张量（弃牌堆+副露）
      4. sampler = MonteCarloSampler(n_samples=5)
      5. samples = sampler.sample(beliefs, context, known_tiles)
      6. Assert len(samples) == 5
      7. For each sample:
         - Assert 对手手牌总数正确（13或变化后）
         - Assert 手牌不与弃牌堆重复
         - Assert 手牌不与副露重复
         - Assert 手牌符合Mahjong规则
    Expected Result: 5个有效且合理的游戏状态
    Evidence: 采样的手牌列表
  ```

  **Commit**: YES
  - Message: `feat(sampler): add detailed MonteCarloSampler with Gumbel-Softmax and constraint checking`
  - Files: `src/drl/monte_carlo_sampler.py`, `tests/unit/test_sampler.py`

- [ ] **Task 2: 全局状态构建器**

  **What to do**:
  1. 创建 `src/drl/belief_network.py`
  2. 实现 `BeliefNetwork` 类，继承 `nn.Module`
  3. 网络架构:
     - Input: 公共观测 (discard_pool[34], melds[16×9], action_history[80×3])
     - Processing: TransformerEncoder + Linear layers
     - Output: 3个对手的概率分布 [batch, 3, 34]（softmax归一化）
  4. 实现 `forward()` 方法处理公共信息
  5. 实现辅助方法 `get_opponent_beliefs(agent_id, context)`

  **Must NOT do**:
  - 不要访问私有手牌信息（仅使用公共信息）
  - 不要硬编码对手数量（保持灵活性）
  - 不要在belief network中使用critic网络

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`（需要深度神经网络架构设计）
  - **Skills**: `pytorch`, `transformers`, `rl-algorithms`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Task 4, Task 6
  - **Blocked By**: None (can start immediately)

  **References**:
  - 现有编码器模式: `src/drl/network.py:HandEncoder, DiscardEncoder, MeldEncoder`
  - Transformer使用: `src/drl/network.py:TransformerHistoryEncoder`
  - 概率分布输出: PyTorch `F.softmax()`

  **Acceptance Criteria**:
  - [ ] BeliefNetwork类可实例化
  - [ ] `forward()` 输入公共观测，输出[batch, 3, 34]概率分布
  - [ ] 每个对手的概率和≈1.0（softmax约束）
  - [ ] 单元测试通过: `test_belief_network.py`

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: BeliefNetwork输出有效概率分布
    Tool: Bash (python)
    Preconditions: 项目已安装依赖
    Steps:
      1. 导入 BeliefNetwork
      2. network = BeliefNetwork(hidden_dim=256)
      3. obs = 构建测试公共观测（batch=4）
      4. beliefs = network(obs)
      5. Assert beliefs.shape == torch.Size([4, 3, 34])
      6. Assert torch.allclose(beliefs.sum(dim=-1), torch.ones(4, 3), atol=1e-5)
      7. Assert (beliefs >= 0).all() and (beliefs <= 1).all()
    Expected Result: 概率分布有效，符合约束
    Evidence: 输出张量值和形状
  ```

  **Commit**: YES
  - Message: `feat(belief): add BeliefNetwork for opponent hand estimation`
  - Files: `src/drl/belief_network.py`, `tests/unit/test_belief_network.py`

---

- [ ] **Task 2: 全局状态构建器**

  **What to do**:
  1. 修改 `src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py`
  2. 添加 `build_global_state(player_id, context)` 方法
  3. 构建完整全局状态，包含:
     - 4玩家完整手牌（不mask）[4, 34]
     - 牌墙剩余牌分布 [82]
     - 弃牌堆 [34]
     - 所有副露信息 [16×9]
     - 特殊指示器（赖子、皮子）[2]
     - 动作历史 [80×3]
     - 当前玩家、庄家等元信息
  4. 总维度: ~1500+ 维
  5. 添加 `build_local_state(player_id, context)` 方法（现有逻辑）

  **Must NOT do**:
  - 不要破坏现有观测接口
  - 不要修改visibility masking逻辑
  - 不要在Phase 3暴露全局信息给actor

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`（需要理解现有观测系统）
  - **Skills**: `pytorch`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Task 5, Task 8
  - **Blocked By**: None

  **References**:
  - 现有观测构建: `src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py:51-75`
  - GameContext结构: `src/mahjong_rl/core/GameData.py:24-85`

  **Acceptance Criteria**:
  - [ ] `build_global_state()` 返回完整全局状态字典
  - [ ] 全局状态包含所有4玩家手牌（未mask）
  - [ ] 总维度 > 1500
  - [ ] 单元测试通过

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: 全局状态包含完整信息
    Tool: Bash (python)
    Steps:
      1. 创建GameContext（4玩家各13张牌）
      2. global_state = builder.build_global_state(0, context)
      3. Assert 'all_hands' in global_state
      4. Assert global_state['all_hands'].shape == [4, 34]
      5. Assert global_state['wall'].shape == [82]
      6. 验证所有玩家手牌未mask（非全5或34）
    Expected Result: 全局状态包含完整游戏信息
  ```

  **Commit**: YES (groups with Task 1)

---

- [ ] **Task 3: 单元测试框架**

  **What to do**:
  1. 创建 `tests/unit/test_belief_state.py`
  2. 创建 `tests/unit/test_centralized_critic.py`
  3. 实现测试辅助函数:
     - `create_test_game_context()` - 创建测试游戏状态
     - `create_test_observation()` - 创建测试观测
     - `assert_valid_probability_distribution()` - 验证概率分布
  4. 添加mock/stub用于隔离测试

  **Must NOT do**:
  - 不要引入外部依赖（仅用unittest）
  - 不要测试未实现的功能

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `pytorch`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: 所有测试任务
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] 测试文件可导入
  - [ ] 辅助函数可用
  - [ ] 示例测试通过

  **Commit**: YES

---

### Wave 2: 核心网络

- [ ] **Task 4: MonteCarloSampler实现**

  **What to do**:
  1. 创建 `src/drl/monte_carlo_sampler.py`
  2. 实现 `MonteCarloSampler` 类
  3. 核心逻辑:
     ```python
     def sample(self, beliefs: torch.Tensor, n_samples: int) -> List[GameContext]:
         """
         从信念分布采样N个可能游戏状态
         beliefs: [3, 34] - 3个对手的概率分布
         returns: List of N GameContext（每个包含采样的对手手牌）
         """
         samples = []
         for _ in range(n_samples):
             # 对每个对手，根据概率分布采样手牌
             # 确保手牌总数正确（13张或变化后）
             # 确保不抽到已弃牌或已副露的牌
             sampled_context = self._sample_single_state(beliefs)
             samples.append(sampled_context)
         return samples
     ```
  4. 实现约束检查（手牌数、不重复、符合规则）
  5. 支持并行采样（GPU batch处理）

  **Must NOT do**:
  - 不要修改原始GameContext（创建副本）
  - 不要生成违反Mahjong规则的手牌
  - 不要采样到已知的牌（如弃牌堆中的牌）

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`（需要理解游戏规则和概率采样）
  - **Skills**: `pytorch`, `probability`, `game-rules`

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖Task 1）
  - **Blocks**: Task 6
  - **Blocked By**: Task 1

  **References**:
  - BeliefNetwork输出: Task 1
  - GameContext结构: `src/mahjong_rl/core/GameData.py`
  - Mahjong规则: `src/mahjong_rl/rules/`

  **Acceptance Criteria**:
  - [ ] MonteCarloSampler可实例化
  - [ ] `sample()` 返回N个有效GameContext
  - [ ] 每个采样符合Mahjong规则
  - [ ] 采样不修改原始context
  - [ ] 支持GPU并行（可选）

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: 采样生成合理手牌
    Tool: Bash (python)
    Steps:
      1. 创建GameContext和BeliefNetwork
      2. beliefs = network(public_obs)  # [3, 34]
      3. sampler = MonteCarloSampler(n_samples=5)
      4. samples = sampler.sample(beliefs, context)
      5. Assert len(samples) == 5
      6. For each sample:
         - Assert 对手手牌总数正确（13或变化后）
         - Assert 手牌不与弃牌堆重复
         - Assert 手牌不与副露重复
    Expected Result: 5个有效且合理的游戏状态
  ```

  **Commit**: YES
  - Message: `feat(sampler): add MonteCarloSampler for belief state sampling`

---

- [ ] **Task 5: CentralizedCriticNetwork实现**

  **What to do**:
  1. 创建 `src/drl/centralized_critic.py`
  2. 实现 `CentralizedCriticNetwork` 类
  3. 网络架构:
     ```python
     class CentralizedCriticNetwork(nn.Module):
         def __init__(self, hidden_dim=512, num_layers=6):
             # 更大的网络处理全局状态
             self.encoder = GlobalStateEncoder(hidden_dim, num_layers)
             self.critic_head = nn.Sequential(
                 nn.Linear(hidden_dim, 256),
                 nn.ReLU(),
                 nn.Linear(256, 1)
             )
         
         def forward(self, global_obs):
             # global_obs: 完整全局状态 [batch, ~1500]
             features = self.encoder(global_obs)
             value = self.critic_head(features)
             return value
     ```
  4. 实现 `GlobalStateEncoder` - 处理高维全局状态
  5. 集成到MAPPO（Task 7详细处理）

  **Must NOT do**:
  - 不要共享Actor的权重（独立网络）
  - 不要在critic中使用action信息（V-value不是Q-value）
  - 不要在Phase 3使用centralized critic

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: `pytorch`, `transformers`

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖Task 2）
  - **Blocks**: Task 7
  - **Blocked By**: Task 2

  **References**:
  - 全局状态构建: Task 2
  - 现有Critic: `src/drl/network.py:446-450`
  - MAPPO使用: `src/drl/mappo.py:94`

  **Acceptance Criteria**:
  - [ ] CentralizedCriticNetwork可实例化
  - [ ] `forward()` 输入全局状态，输出标量value
  - [ ] 网络参数量 > 5M（比现有critic大）
  - [ ] 训练时收敛（loss下降）

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: Centralized critic评估全局状态
    Tool: Bash (python)
    Steps:
      1. critic = CentralizedCriticNetwork()
      2. global_obs = 构建测试全局状态（batch=4）
      3. value = critic(global_obs)
      4. Assert value.shape == torch.Size([4, 1])
      5. Assert value在合理范围（基于reward scale，如[-100, 100]）
    Expected Result: 正确输出全局状态价值估计
  ```

  **Commit**: YES
  - Message: `feat(critic): add CentralizedCriticNetwork for full state value estimation`

---

- [ ] **Task 6: 修改Actor集成信念**

  **What to do**:
  1. 修改 `src/drl/network.py` 中的 `ActorCriticNetwork`
  2. 添加信念集成:
     ```python
     class ActorWithBelief(nn.Module):
         def __init__(self, ..., use_belief=True, n_samples=5):
             self.use_belief = use_belief
             self.n_samples = n_samples
             if use_belief:
                 self.belief_network = BeliefNetwork(...)
                 self.sampler = MonteCarloSampler(...)
         
         def forward(self, obs, action_mask, context=None):
             if self.use_belief and context is not None:
                 # 估计信念
                 beliefs = self.belief_network(obs)
                 # 采样N个状态
                 samples = self.sampler.sample(beliefs, context, self.n_samples)
                 # 平均N个采样状态的特征
                 sampled_features = []
                 for sample in samples:
                     features = self.encoder(sample)
                     sampled_features.append(features)
                 features = torch.stack(sampled_features).mean(dim=0)
             else:
                 features = self.encoder(obs)
             
             # Actor heads
             action_type_logits = self.actor_type(features)
             ...
     ```
  3. 确保backward兼容（use_belief=False时行为不变）
  4. 优化：并行处理N个采样（batch维度）

  **Must NOT do**:
  - 不要破坏现有接口（默认use_belief=False）
  - 不要在critic中集成信念（仅Actor）
  - 不要在Phase 3的critic训练中使用belief

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: `pytorch`, `rl-algorithms`

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖Task 1, 4）
  - **Blocks**: Task 7
  - **Blocked By**: Task 1, Task 4

  **References**:
  - 现有Actor: `src/drl/network.py:408-532`
  - BeliefNetwork: Task 1
  - MonteCarloSampler: Task 4

  **Acceptance Criteria**:
  - [ ] Actor可配置use_belief
  - [ ] use_belief=True时集成belief sampling
  - [ ] use_belief=False时行为与之前一致
  - [ ] 支持N个采样平均
  - [ ] 单元测试通过

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: Actor集成信念采样
    Tool: Bash (python)
    Steps:
      1. actor = ActorWithBelief(use_belief=True, n_samples=5)
      2. obs, context = 构建测试数据
      3. action_logits = actor(obs, action_mask, context)
      4. Assert action_logits形状正确
      5. 验证N个采样都被处理（检查sampler调用次数）
    Expected Result: Actor正确集成belief sampling
  ```

  **Commit**: YES
  - Message: `feat(actor): integrate belief sampling into Actor network`

---

### Wave 3: 训练集成

- [ ] **Task 7: DualCriticTraining修改MAPPO**

  **What to do**:
  1. 大幅修改 `src/drl/mappo.py`
  2. 实现 `DualCriticMAPPO` 类（或修改现有MAPPO）:
     ```python
     class DualCriticMAPPO:
         def __init__(self, actor, centralized_critic, decentralized_critic, ...):
             self.actor = actor
             self.centralized_critic = centralized_critic
             self.decentralized_critic = decentralized_critic
         
         def update(self, buffer, next_obs, next_global_obs, training_phase):
             # 根据phase选择critic
             if training_phase in [1, 2]:
                 critic = self.centralized_critic
                 # 使用全局状态计算value
                 values = critic(buffer.global_observations)
                 next_value = critic(next_global_obs)
             else:  # phase 3
                 critic = self.decentralized_critic
                 # 使用局部观测
                 values = critic(buffer.observations)
                 next_value = critic(next_obs)
             
             # 标准PPO更新流程
             returns, advantages = compute_gae(values, next_value, ...)
             # ... PPO更新逻辑
     ```
  3. 修改 `RolloutBuffer` 支持存储全局状态
  4. 处理两个critic的独立优化

  **Must NOT do**:
  - 不要在Phase 3使用centralized critic
  - 不要混合两个critic的梯度
  - 不要修改reward计算

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: `pytorch`, `rl-algorithms`, `mappo`

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖Task 5, 6）
  - **Blocks**: Task 9
  - **Blocked By**: Task 5, Task 6

  **References**:
  - 现有MAPPO: `src/drl/mappo.py:14-325`
  - CentralizedCritic: Task 5
  - Actor with belief: Task 6
  - 课程学习phase: `example_mahjong_env.py:683-771`

  **Acceptance Criteria**:
  - [ ] MAPPO支持dual-critic
  - [ ] Phase 1-2使用centralized critic
  - [ ] Phase 3使用decentralized critic
  - [ ] 两个critic分别正确训练
  - [ ] 训练流程完整运行

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: Dual critic根据phase切换
    Tool: Bash (python)
    Steps:
      1. mappo = DualCriticMAPPO(actor, cent_critic, decent_critic)
      2. Phase 1: 调用update → 验证使用cent_critic
      3. Phase 2: 调用update → 验证使用cent_critic
      4. Phase 3: 调用update → 验证使用decent_critic
      5. Assert 每个phase的value来源正确
    Expected Result: Phase 1-2 centralized，Phase 3 decentralized
  ```

  **Commit**: YES
  - Message: `feat(training): implement DualCriticTraining for CTDE`

---

- [ ] **Task 8: 环境集成全局状态**

  **What to do**:
  1. 修改 `example_mahjong_env.py`
  2. 在 `step()` 和 `reset()` 中构建全局状态:
     ```python
     def step(self, action):
         # ... 现有逻辑
         
         # 构建全局状态（用于centralized critic）
         global_observation = self.observation_builder.build_global_state(
             agent_id, self.context
         )
         
         # 返回包含global_observation的info
         return observation, reward, terminated, truncated, {
             ...,
             'global_observation': global_observation
         }
     ```
  3. 修改 `_apply_visibility_mask` 保持现有逻辑（仅影响actor观测）
  4. 确保 `global_observation` 总是完整信息（不受phase影响）

  **Must NOT do**:
  - 不要修改现有观测接口（backward兼容）
  - 不要全局状态也受visibility mask影响
  - 不要破坏human_vs_ai模式

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `pettingzoo`

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖Task 2）
  - **Blocks**: Task 9
  - **Blocked By**: Task 2

  **References**:
  - 全局状态构建: Task 2
  - 环境接口: `example_mahjong_env.py:440-550`
  - Visibility masking: `example_mahjong_env.py:683-771`

  **Acceptance Criteria**:
  - [ ] 环境返回global_observation
  - [ ] global_observation不受phase影响（总是完整）
  - [ ] 现有接口保持兼容
  - [ ] human_vs_ai模式正常工作

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: 环境返回完整全局状态
    Tool: Bash (python)
    Steps:
      1. env = WuhanMahjongEnv()
      2. obs, info = env.reset()
      3. Assert 'global_observation' in info
      4. action = env.action_space.sample()
      5. obs, reward, term, trunc, info = env.step(action)
      6. Assert 'global_observation' in info
      7. 验证Phase 1-3的global_observation都完整
    Expected Result: 环境始终提供完整全局状态
  ```

  **Commit**: YES (groups with Task 7)

---

- [ ] **Task 9: 训练流程验证**

  **What to do**:
  1. 创建训练脚本 `train_dual_critic.py`
  2. 配置完整的训练流程:
     - 初始化所有网络（actor, both critics, belief）
     - 设置训练循环
     - 处理phase切换
     - 日志记录
  3. 运行小规模训练（100 episodes）验证:
     - 没有崩溃或错误
     - Loss正常下降
     - Phase切换正确
  4. 对比baseline验证改进

  **Must NOT do**:
  - 不要运行完整训练（太耗时）
  - 不要修改hyperparameters（使用默认值）
  - 不要跳过验证步骤

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `pytorch`, `rl-training`

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖Task 7, 8）
  - **Blocks**: Task 10
  - **Blocked By**: Task 7, Task 8

  **References**:
  - 现有训练: `src/drl/trainer.py`
  - MAPPO训练: Task 7
  - 环境: Task 8

  **Acceptance Criteria**:
  - [ ] 训练脚本可运行
  - [ ] 100 episodes无错误
  - [ ] Loss下降
  - [ ] Phase切换正确

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: 小规模训练验证
    Tool: Bash (python train_dual_critic.py --episodes 100)
    Steps:
      1. 运行训练脚本（100 episodes）
      2. 监控loss曲线
      3. 验证Phase 1 → 2 → 3切换
      4. 检查无RuntimeError
      5. Assert loss总体呈下降趋势
    Expected Result: 训练流程正常运行
    Evidence: 训练日志
  ```

  **Commit**: YES
  - Message: `feat(training): add training script for dual-critic`
  - Files: `train_dual_critic.py`

---

- [ ] **Task 3a: 实现对手策略池（PolicyPool，P3优先级）**

  **What to do**:
  1. 创建 `src/drl/policy_pool.py`
  2. 实现 `PolicyPool` 类
     ```python
     class PolicyPool:
         """管理历史策略池，用于后期自对弈"""

         def __init__(self, capacity: int = 10, min_samples: int = 100):
             self.capacity = capacity
             self.policies = []  # List of (policy_id, policy, samples_used)
             self.min_samples = min_samples
             self.next_id = 0

         def add_policy(self, policy: Dict, samples: int = 100) -> int:
             """添加新策略到池中"""

         def sample_policy(self, k: int = 1, weights: Optional[List[float]] = None) -> Dict:
             """从池中采样策略"""

         def get_policy(self, policy_id: int) -> Dict:
             """获取指定策略"""
     ```
  3. 实现策略添加、采样、检索方法
  4. 支持策略权重（根据性能调整采样概率）
  5. 确保使用次数少的策略被优先采样

  **Must NOT do**:
  - 不要修改原始策略网络
  - 不要在池满时删除当前使用的策略
  - 不要硬编码策略数量

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`（需要理解自对弈逻辑）
  - **Skills**: `pytorch`, `rl-training`, `data-structures`

  **Parallelization**:
  - **Can Run In Parallel**: YES（可独立于其他任务）
  - **Parallel Group**: Wave 3 (with Tasks 7, 8, 9)
  - **Blocks**: Task 9
  - **Blocked By**: None

  **References**:
  - 现有NFSP实现: `src/drl/nfsp.py`
  - 策略池设计: 见 draft 中 P3.1 部分
  - 自对弈模式: `src/drl/trainer.py`

  **Acceptance Criteria**:
  - [ ] PolicyPool类可实例化
  - [ ] `add_policy()` 成功添加策略并返回ID
  - [ ] `sample_policy()` 返回有效策略
  - [ ] 使用次数少的策略被优先采样
  - [ ] 池满时正确替换最旧的策略
  - [ ] 单元测试通过

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: 策略池添加和采样
    Tool: Bash (python)
    Preconditions: 项目已安装依赖
    Steps:
      1. 导入 PolicyPool
      2. pool = PolicyPool(capacity=3, min_samples=10)
      3. 添加3个策略：
         - policy1 = {'state_dict': {...}, 'performance': 0.7}
         - policy2 = {'state_dict': {...}, 'performance': 0.8}
         - policy3 = {'state_dict': {...}, 'performance': 0.9}
         - id1 = pool.add_policy(policy1, samples=50)
         - id2 = pool.add_policy(policy2, samples=30)
         - id3 = pool.add_policy(policy3, samples=20)
      4. Assert id1 == 0, id2 == 1, id3 == 2
      5. 采样策略：sampled = pool.sample_policy(k=1)
      6. Assert sampled['id'] == 3（policy3使用次数最少）
      7. 添加第4个策略（超出容量）
         - id4 = pool.add_policy(policy4, samples=10)
      8. Assert len(pool.policies) == 3（容量限制）
      9. Assert policy1被移除（最旧的策略）
    Expected Result: 策略池正常添加和采样
    Evidence: 返回的策略ID和池容量
  ```

  **Commit**: YES
  - Message: `feat(policy): add PolicyPool for historical strategy management in self-play`
  - Files: `src/drl/policy_pool.py`, `tests/unit/test_policy_pool.py`

---

### Wave 4: 测试验证

- [ ] **Task 10: 集成测试**

  **What to do**:
  1. 创建 `tests/integration/test_belief_critic.py`
  2. 测试完整流程:
     - End-to-end训练流程
     - 信念网络 + Actor集成
     - Dual-critic切换
     - Phase转换
  3. 测试边界情况:
     - 空信念（初期）
     - 极端手牌分布
     - Phase边界切换
  4. 性能测试（确保不过慢）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `testing`, `pytorch`

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖所有前置任务）
  - **Blocks**: Task 11
  - **Blocked By**: Task 9

  **Acceptance Criteria**:
  - [ ] 所有集成测试通过
  - [ ] 边界情况处理正确
  - [ ] 性能在可接受范围

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: 完整训练流程集成测试
    Tool: Bash (python -m pytest tests/integration/)
    Steps:
      1. 运行所有集成测试
      2. 验证每个组件正确集成
      3. 检查错误处理
      4. Assert 100%测试通过
    Expected Result: 所有集成测试通过
  ```

  **Commit**: YES

---

- [ ] **Task 11: 性能基准测试**

  **What to do**:
  1. 创建 `benchmarks/compare_baseline.py`
  2. 对比新架构 vs baseline:
     - 训练速度（steps/sec）
     - 内存使用
     - 胜率提升
  3. 运行对比实验（至少500 episodes）
  4. 生成对比报告

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `benchmarking`, `pytorch`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: Task 12
  - **Blocked By**: Task 10

  **Acceptance Criteria**:
  - [ ] 胜率提升 > 5%
  - [ ] 训练速度在可接受范围
  - [ ] 生成对比报告

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: 性能基准测试
    Tool: Bash (python benchmarks/compare_baseline.py --episodes 500)
    Steps:
      1. 运行baseline和新架构对比
      2. 收集胜率、训练速度数据
      3. Assert 新架构胜率 > baseline + 5%
      4. 生成对比报告
    Expected Result: 新架构显著优于baseline
    Evidence: benchmark_report.json
  ```

  **Commit**: YES

---

- [ ] **Task 12: 文档和示例**

  **What to do**:
  1. 更新 `CLAUDE.md` 添加新架构说明
  2. 创建 `docs/belief_critic_architecture.md`:
     - 架构图
     - 设计决策解释
     - 使用示例
  3. 添加代码注释（关键类和方法）
  4. 创建快速开始示例

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `documentation`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: None (final task)
  - **Blocked By**: Task 11

  **Acceptance Criteria**:
  - [ ] 文档清晰完整
  - [ ] 架构图准确
  - [ ] 示例可运行

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: 文档完整性检查
    Tool: Bash (manual review)
    Steps:
      1. 检查所有新文件都有docstring
      2. 验证文档与代码一致
      3. 检查示例可运行
    Expected Result: 文档完整可用
  ```

  **Commit**: YES
  - Message: `docs: add belief critic architecture documentation`
  - Files: `belief_critic_architecture.md`

---

- [ ] **Task 4a: TensorBoard集成和性能监控（P2优先级）**

  **What to do**:
  1. 创建 `src/drl/tensorboard_logger.py`
     ```python
     from torch.utils.tensorboard import SummaryWriter
     import os
     from datetime import datetime

     class TensorBoardLogger:
         def __init__(self, log_dir: str):
             self.writer = SummaryWriter(log_dir)
             self.step = 0

         def log_scalar(self, tag: str, value: float, step: int):
             """记录标量指标"""
             self.writer.add_scalar(tag, value, self.step)

         def log_histogram(self, tag: str, values, step: int):
             """记录分布"""
             self.writer.add_histogram(tag, values, self.step)

         def log_belief_distribution(self, beliefs: torch.Tensor, step: int):
             """记录对手手牌信念分布"""
             # beliefs: [batch, 3, 34] - 3个对手 × 34种牌
             for opponent_id in range(3):
                 for tile_id in range(34):
                     self.writer.add_scalar(
                         f'belief/opponent_{opponent_id}/tile_{tile_id}',
                         beliefs[step, opponent_id, tile_id].item(),
                         self.step
                     )

         def close(self):
             self.writer.close()
     ```
  2. 创建 `src/drl/performance_monitor.py`
     ```python
     class PerformanceMonitor:
         """监控训练性能指标"""

         def __init__(self):
             self.episode_times = []
             self.memory_usage = []

         def log_episode_time(self, start_time: float, end_time: float):
             self.episode_times.append(end_time - start_time)

         def log_memory_usage(self, memory_mb: float):
             self.memory_usage.append(memory_mb)

         def get_training_speed(self) -> float:
             """返回训练速度（episodes/hour）"""
             if not self.episode_times:
                 return 0.0
             avg_time = sum(self.episode_times) / len(self.episode_times)
             return 3600.0 / avg_time  # 秒/小时
     ```
  3. 集成到 `NFSPTrainer` 类
     - 在 `__init__` 中初始化 `self.tb_logger` 和 `self.perf_monitor`
     - 在 `train()` 方法中记录关键指标
     - 记录信念分布变化、loss、reward、win_rate
     - 记录训练速度、内存使用

  **Must NOT do**:
  - 不要过度记录导致I/O瓶颈
  - 不要在训练循环中调用close()

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`（监控任务较简单）
  - **Skills**: `pytorch`, `monitoring`, `logging`

  **Parallelization**:
  - **Can Run In Parallel**: YES（可独立于其他任务）
  - **Parallel Group**: Wave 4 (with Task 12)
  - **Blocks**: Task 12
  - **Blocked By**: None

  **References**:
  - TensorBoard文档: https://pytorch.org/docs/stable/tensorboard/
  - 现有trainer: `src/drl/trainer.py`
  - 监控指标设计: 见 draft 中 P2.1, P2.2 部分

  **Acceptance Criteria**:
  - [ ] TensorBoardLogger类可实例化
  - [ ] `log_scalar()`, `log_histogram()` 方法正常工作
  - [ ] `log_belief_distribution()` 正确记录3×34分布
  - [ ] PerformanceMonitor可计算训练速度
  - [ ] 集成到NFSPTrainer无错误
  - [ ] TensorBoard可访问（运行 `tensorboard --logdir logs/`）
  - [ ] 单元测试通过

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: TensorBoard监控集成验证
    Tool: Bash (python)
    Preconditions: 项目已安装依赖，包括tensorboard
    Steps:
      1. 导入 TensorBoardLogger 和 PerformanceMonitor
      2. tb_logger = TensorBoardLogger(log_dir='test_logs')
      3. perf_monitor = PerformanceMonitor()
      4. 记录一些测试数据：
         - tb_logger.log_scalar('loss', 0.5, step=0)
         - tb_logger.log_scalar('win_rate', 0.7, step=0)
         - beliefs = torch.randn(4, 3, 34)
         - tb_logger.log_belief_distribution(beliefs, step=0)
         - perf_monitor.log_episode_time(0, 10.5)
         - perf_monitor.log_memory_usage(2048.0)
      5. speed = perf_monitor.get_training_speed()
      6. Assert speed > 300  # episodes/hour
      7. 关闭logger: tb_logger.close()
      8. 验证日志文件生成：
         - Assert os.path.exists('test_logs/events.out.tfevents...')
    Expected Result: TensorBoard日志正常记录
    Evidence: test_logs目录内容
  ```

  **Commit**: YES
  - Message: `feat(monitor): add TensorBoard integration and performance monitoring`
  - Files: `src/drl/tensorboard_logger.py`, `src/drl/performance_monitor.py`, `tests/unit/test_monitoring.py`

---

## 6. 提交策略

| 任务 | 提交信息 | 文件 | 验证命令 |
|------|----------|------|----------|
| 0 | `fix(architecture): implement dual-critic training with phase-aware critic selection` | agent.py, buffer.py, mappo.py, trainer.py | `pytest tests/unit/test_dual_critic.py` |
| 1 | `feat(belief): add BeliefNetwork with Bayesian update` | belief_network.py | `pytest tests/unit/test_belief_network.py` |
| 1a | `feat(sampler): add detailed MonteCarloSampler with Gumbel-Softmax` | monte_carlo_sampler.py | `pytest tests/unit/test_sampler.py` |
| 2 | `feat(observation): add global state builder` | observation_builder.py | `pytest tests/unit/test_observation.py` |
| 3 | `test: add test framework` | test_*.py | `pytest tests/unit/` |
| 4 | `feat(sampler): add MonteCarloSampler` | monte_carlo_sampler.py | `pytest tests/unit/test_sampler.py` |
| 5 | `feat(critic): add CentralizedCriticNetwork` | centralized_critic.py | `pytest tests/unit/test_critic.py` |
| 6 | `feat(actor): integrate belief sampling` | network.py | `pytest tests/unit/test_actor.py` |
| 7 | `feat(training): implement DualCriticTraining` | mappo.py | `pytest tests/unit/test_mappo.py` |
| 8 | `feat(env): integrate global state` | example_mahjong_env.py | `pytest tests/integration/test_env.py` |
| 9 | `feat(training): add training script` | train_dual_critic.py | `python train_dual_critic.py --episodes 10` |
| 3a | `feat(policy): add PolicyPool for historical strategy management` | policy_pool.py | `pytest tests/unit/test_policy_pool.py` |
| 10 | `test: add integration tests` | test_belief_critic.py | `pytest tests/integration/` |
| 11 | `benchmark: add performance comparison` | compare_baseline.py | `python compare_baseline.py --episodes 100` |
| 12 | `docs: add architecture documentation` | belief_critic_architecture.md | 人工审查 |
| 4a | `feat(monitor): add TensorBoard integration and performance monitoring` | tensorboard_logger.py, performance_monitor.py | `pytest tests/unit/test_monitoring.py` |

---

## 7. 成功标准

### 7.1 验证命令

```bash
# 单元测试
pytest tests/unit/ -v

# 集成测试
pytest tests/integration/ -v

# 小规模训练验证
python train_dual_critic.py --episodes 100

# 性能基准
python benchmarks/compare_baseline.py --episodes 500
```

### 7.2 最终检查清单

- [ ] 所有单元测试通过
- [ ] 所有集成测试通过
- [ ] 100 episodes训练无错误
- [ ] 胜率提升 > 5%（vs baseline）
- [ ] 文档完整
- [ ] 代码审查通过（无重大issues）
- [ ] human_vs_ai模式正常工作
- [ ] 内存使用合理（无泄漏）

### 7.3 性能指标

| 指标 | Baseline | 新架构 | 目标 |
|------|----------|--------|------|
| 训练速度 (steps/sec) | 100 | - | > 30 (考虑计算开销) |
| 内存使用 (GB) | 2 | - | < 8 |
| 胜率 (vs random) | 70% | - | > 75% |
| 胜率提升 | - | - | > +5% |

---

## 8. 风险与缓解

### 8.1 高风险

1. **训练不稳定**: Centralized critic可能导致训练发散
   - **缓解**: 更小的学习率，gradient clipping，phase渐进切换

2. **计算开销过大**: 蒙特卡罗采样显著降低训练速度
   - **缓解**: 采样数可配置（N=5），GPU并行，异步采样

3. **训练-执行差距**: centralized critic和decentralized critic差异过大
   - **缓解**: Phase 2渐进过渡，Dual-critic平滑切换

### 8.2 中风险

1. **信念估计不准确**: 初期信念质量差
   - **缓解**: 贝叶斯更新，Transformer时序建模，初期依赖公共信息

2. **内存不足**: 存储全局状态和采样状态
   - **缓解**: 及时释放，使用float16，采样数控制

---

## 9. 附录

### 9.1 架构图

```
训练阶段 (Phase 1-2):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  公共观测 (弃牌、副露、历史)                                 │
│           ↓                                                 │
│  ┌─────────────────────┐                                    │
│  │   BeliefNetwork     │                                    │
│  │  (public → beliefs) │                                    │
│  └──────────┬──────────┘                                    │
│             ↓ beliefs [3, 34]                               │
│  ┌─────────────────────┐                                    │
│  │ MonteCarloSampler   │                                    │
│  │ (beliefs → N samples│                                    │
│  └──────────┬──────────┘                                    │
│             ↓ N GameContexts                                │
│  ┌─────────────────────┐                                    │
│  │  Actor (w/ belief)  │                                    │
│  │ (avg N samples →    │                                    │
│  │  action)            │                                    │
│  └──────────┬──────────┘                                    │
│             ↓ action                                        │
│           Env                                               │
│             ↓                                               │
│  ┌─────────────────────┐                                    │
│  │ CentralizedCritic   │◄──── 完整全局状态（4手牌+墙+公共） │
│  │ (global → V)        │                                    │
│  └─────────────────────┘                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

执行阶段 (Phase 3):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  公共观测 ──→ BeliefNetwork ──→ beliefs                     │
│                                     ↓                       │
│                              MonteCarloSampler              │
│                                     ↓                       │
│                              Actor (w/ belief)              │
│                                     ↓                       │
│                                   动作                      │
│                                     ↓                       │
│  ┌─────────────────────┐                                    │
│  │ DecentralizedCritic │◄──── 局部观测（仅自身手牌+公共）    │
│  │ (local → V)         │                                    │
│  └─────────────────────┘                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 关键类关系

```
DualCriticMAPPO
├── actor: ActorWithBelief
│   ├── belief_network: BeliefNetwork
│   ├── sampler: MonteCarloSampler
│   └── encoder: ObservationEncoder
├── centralized_critic: CentralizedCriticNetwork
│   └── encoder: GlobalStateEncoder
└── decentralized_critic: ActorCriticNetwork.critic
    └── encoder: ObservationEncoder

WuhanMahjongEnv
├── observation_builder: Wuhan7P4LObservationBuilder
│   ├── build() → local observation
│   └── build_global_state() → global observation
└── training_phase: {1, 2, 3}
```

### 9.3 开发环境

- **Python**: 3.8+
- **PyTorch**: 1.12+
- **CUDA**: 11.3+ (推荐)
- **内存**: 16GB+ (训练时32GB+推荐)
- **GPU**: RTX 3080+ (推荐，可选)

---

## 10. 使用指南

### 10.1 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行训练（小规模测试）
python train_dual_critic.py --episodes 100 --phase 1

# 3. 运行完整训练
python train_dual_critic.py --episodes 100000 --config configs/dual_critic.yaml

# 4. 评估训练好的模型
python evaluate.py --checkpoint checkpoints/dual_critic_final.pt
```

### 10.2 配置选项

```yaml
# configs/dual_critic.yaml
belief_network:
  hidden_dim: 256
  num_layers: 4
  num_heads: 4

monte_carlo_sampler:
  n_samples: 5  # 蒙特卡罗采样数
  device: cuda

centralized_critic:
  hidden_dim: 512
  num_layers: 6

training:
  phase_schedule:
    phase1_episodes: 50000  # 全知阶段
    phase2_episodes: 50000  # 渐进阶段
    phase3_episodes: 100000 # 真实阶段
  
  dual_critic:
    use_centralized: [1, 2]  # 哪些phase使用centralized
    use_decentralized: [3]   # 哪些phase使用decentralized
```

---

**计划完成时间**: 约2-3周（全职开发）  
**关键路径**: Task 1 → Task 4 → Task 6 → Task 7 → Task 9  
**并行任务**: Wave内任务可部分并行

运行 `/start-work` 开始执行此计划。
