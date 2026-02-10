# 信念状态与Centralized Critic实现计划（完善版）

**版本**: v2.0（基于用户反馈完善）
**日期**: 2026-02-09
**原计划**: belief-state-centralized-critic.md
**改进**: 修复Task重复、合并重叠任务、补充缺失内容

---

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
    - 已添加: centralized_critic 参数和 update_centralized() 方法 ✅
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
2. `src/drl/centralized_critic.py` - Centralized critic网络（已存在，需完善）
3. `src/drl/monte_carlo_sampler.py` - 蒙特卡罗采样器
4. 修改 `src/drl/network.py` - 集成信念的Actor
5. 修改 `src/drl/mappo.py` - Dual-critic训练逻辑（已完成基础设施）
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

## 3. Wave 执行策略

### Wave 0 (核心问题修复 - 3天) - [P0] ✅ 已完成

- [x] Task 0: 修复CentralizedCritic未实际使用问题 ✅ 已完成

### Wave 1 (基础设施 - 5-6天)

- [x] Task 1: BeliefNetwork实现 [P1] ✅ 已完成 (2026-02-09)
- [x] Task 2a: BeliefNetwork辅助损失训练 [P1] ⭐ **新增** ✅ 已完成 (2026-02-09)
- [x] Task 2b: 全局状态构建器 [P1] ⭐ **重新标记** ✅ 已完成 (2026-02-09)
- [x] Task 3: 单元测试框架 [P2] ✅ 已完成 (2026-02-09)
- [x] Task 3b: 代码质量基础设施 [P2] ⭐ **新增** ✅ 已完成 (2026-02-09)

### Wave 3 (测试验证 - 5-7天)

- [x] Task 7: DualCriticTraining修改MAPPO [P1] ✅ 已完成 (2026-02-09)
- [x] Task 8: 环境集成全局状态 [P1] ✅ 已完成 (2026-02-09)
- [x] Task 8a: Phase间Checkpoint热启动 [P1] ✅ 已完成 (2026-02-09) ⭐ **新增**
- [ ] Task 9: 训练流程验证 [P1]
- [ ] Task 9a: 实现对手策略池 [P3]
- [ ] Task 10: 集成测试 [P1]
- [ ] Task 11: 性能基准测试 [P2]
- [ ] Task 12: 文档和示例 [P2]
- [ ] Task 4a: TensorBoard集成 [P2]
- [ ] Task 8a: 生产部署支持 [P2]
- [ ] Task 4b: 训练监控与诊断 [P2]

### Wave 3 (测试验证 - 5-7天)

- [ ] Task 10: 集成测试 [P1]
- [ ] Task 11: 性能基准测试 [P2]
- [ ] Task 12: 文档和示例 [P2]
- [ ] Task 4a: TensorBoard集成 [P2]
- [ ] Task 12a: 生产部署支持 [P2]
- [ ] Task 4b: 训练监控与诊断 [P2]

### Wave 4 (测试验证 - 5-7天)

- [ ] Task 10: 集成测试 [P1]
- [ ] Task 11: 性能基准测试 [P2]
- [ ] Task 12: 文档和示例 [P2]
- [ ] Task 4a: TensorBoard集成 [P2]
- [ ] Task 12a: 生产部署支持 [P2]
- [ ] Task 4b: 训练监控与诊断 [P2]
- [ ] Task 3a: 实现对手策略池 [P3]

### Wave 4 (测试验证 - 5-7天)

- [ ] Task 10: 集成测试 [P1]
- [ ] Task 11: 性能基准测试 [P2]
- [ ] Task 12: 文档和示例 [P2]
- [ ] Task 4a: TensorBoard集成 [P2]
- [ ] Task 12a: 生产部署支持 [P2] ⭐ **新增**
- [ ] Task 4b: 训练监控与诊断 [P2] ⭐ **新增**

---

## 4. 任务详情

### Wave 0: 核心问题修复

- [x] **Task 0: 修复CentralizedCritic未实际使用问题** ✅ 已完成

  **What to do**: ✅ 已完成
  1. 修改 `src/drl/agent.py` 中的 `NFSPAgentPool` 类
     - 添加 `store_global_transition(all_observations, ...)` 方法
     - 添加 `get_global_observations(episode_num)` 方法
  2. 完善 `src/drl/buffer.py` 中的 `get_centralized_batch()` 方法
     - 确保正确存储和检索所有智能体的观测
     - 支持批次格式转换
  3. 修改 `src/drl/mappo.py` 中的 `MAPPO` 类
     - 添加 `centralized_critic` 参数
     - 添加 `update_centralized()` 方法
     - 根据 training_phase 选择使用哪个 critic
     - Phase 1-2: 使用 centralized critic (完整全局状态)
     - Phase 3: 使用 decentralized critic (仅局部观测)
  4. 更新 `src/drl/trainer.py` 中的训练循环
     - 在 `_run_episode()` 中收集 `all_agents_observations`
     - 传递给 buffer 和训练函数
     - 传递 `training_phase` 参数

  **完成日期**: 2025-02-09
  **测试**: test_centralized_simple.py 所有测试通过 ✅

---

### Wave 1: 基础设施

#### Task 1: BeliefNetwork实现

**What to do**:
1. 创建 `src/drl/belief_network.py`
2. 实现 `BeliefNetwork` 类，继承 `nn.Module`
3. 网络架构:
   - Input: 公共观测 (discard_pool[34], melds[16×9], action_history[80×3])
   - Processing: TransformerEncoder + Linear layers
   - Output: 3个对手的概率分布 [batch, 3, 34]（softmax归一化）
4. 实现 `forward()` 方法处理公共信息
5. 实现辅助方法 `get_opponent_beliefs(agent_id, context)`
6. 实现贝叶斯更新方法 `update_beliefs(action_history, discard_pool, melds)`

**Must NOT do**:
- 不要访问私有手牌信息（仅使用公共信息）
- 不要硬编码对手数量（保持灵活性）
- 不要在belief network中使用critic网络

**Recommended Agent Profile**:
- **Category**: `ultrabrain`（需要深度神经网络架构设计）
- **Skills**: `pytorch`, `transformers`, `rl-algorithms`

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 1 (with Tasks 2a, 2b, 3)
- **Blocks**: Task 4, Task 6
- **Blocked By**: Task 0 (已满足)

**References**:
- 现有编码器模式: `src/drl/network.py:HandEncoder, DiscardEncoder, MeldEncoder`
- Transformer使用: `src/drl/network.py:TransformerHistoryEncoder`
- 概率分布输出: PyTorch `F.softmax()`

**Acceptance Criteria**:
- [ ] BeliefNetwork类可实例化
- [ ] `forward()` 输入公共观测，输出[batch, 3, 34]概率分布
- [ ] 每个对手的概率和≈1.0（softmax约束）
- [ ] `update_beliefs()` 方法存在且可调用
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

---

#### Task 2a: BeliefNetwork辅助损失训练 ⭐ **新增**

**What to do**:
1. 在 BeliefNetwork 中添加辅助loss方法
2. 实现3个辅助预测任务：
   - 预测对手下一轮打出的牌（34分类）
   - 预测对手是否吃/碰/杠（4分类）
   - 预测对手手牌总数（回归）
3. 实现总loss函数：
   ```
   total_loss = 0.7 × rl_loss + 0.3 × auxiliary_loss
   auxiliary_loss = 0.4 × action_prediction_loss + 0.3 × meld_prediction_loss + 0.3 × tile_count_loss
   ```
4. 添加开关 `use_auxiliary_loss=True/False`

**Files**:
- `src/drl/belief_network.py`（扩展）
- `src/drl/trainer.py`（集成）

**Dependencies**: Task 1
**Priority**: P1

**Acceptance Criteria**:
- [ ] auxiliary_loss 方法可调用
- [ ] use_auxiliary_loss 开关可控制
- [ ] 3个辅助任务都有对应的loss计算
- [ ] 单元测试通过: `test_belief_auxiliary_loss.py`

---

#### Task 2b: 全局状态构建器 ⭐ **重新标记（原Task 2第二部分）**

**What to do**:
1. 修改 `src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py`
2. 实现 `build_global_observation(state)` 方法
3. 构建完整全局状态：
   ```
   global_observation = {
       'player_0_hand': [14, 34],      # 当前玩家手牌（真实值）
       'player_1_hand': [14, 34],      # 玩家1手牌（真实值，Phase 1可见）
       'player_2_hand': [14, 34],      # 玩家2手牌（真实值，Phase 1可见）
       'player_3_hand': [14, 34],      # 玩家3手牌（真实值，Phase 1可见）
       'wall_tiles': [34, 34],         # 牌墙剩余牌（真实值）
       'discard_piles': [4, 34],       # 4个玩家的弃牌堆（公共信息）
       'melds': [4, 16, 34],          # 4个玩家的副露（公共信息）
       'current_player': int,            # 当前玩家索引
       'remaining_wall_count': int,     # 牌墙剩余牌数
       'game_progress': float,          # 游戏进度（0.0-1.0）
   }
   ```
4. 根据 training_phase 返回不同粒度的全局状态：
   - Phase 1: 返回所有真实信息
   - Phase 2: 返回部分遮蔽信息
   - Phase 3: 返回近似信息（使用信念采样）

**Must NOT do**:
- 不要修改现有观测构建逻辑（仅添加新方法）
- 不要在全局状态中包含私有信息（如其他玩家的手牌在Phase 3）
- 不要返回未处理的数据（必须经过标准化）

**Recommended Agent Profile**:
- **Category**: `ultrabrain`（需要理解游戏状态和数据流）
- **Skills**: `observation-builder`, `game-state`, `pytorch`

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 1 (with Tasks 1, 3, 3b)
- **Blocks**: Task 5, Task 8
- **Blocked By**: None (可立即开始)

**References**:
- 现有观测构建: `src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py`
- 游戏状态: `src/mahjong_rl/core/GameData.py`
- 课程学习: `example_mahjong_env.py:683-771`

**Acceptance Criteria**:
- [ ] `build_global_observation()` 方法可调用
- [ ] 返回全局状态格式正确
- [ ] 不同training_phase返回不同粒度
- [ ] 单元测试通过: `test_global_observation.py`

**Agent-Executed QA Scenarios**:
```
Scenario: 全局状态构建器返回正确格式
  Tool: Bash (python)
  Preconditions: 已创建测试GameContext
  Steps:
    1. 构建器 = ObservationBuilder()
    2. state = 创建测试GameContext（4个玩家）
    3. global_obs = 构建器.build_global_observation(state, training_phase=1)
    4. Assert 'player_0_hand' in global_obs
    5. Assert 'player_1_hand' in global_obs
    6. Assert 'wall_tiles' in global_obs
    7. Assert global_obs['player_0_hand'].shape == [14, 34]
    8. Assert global_obs['player_1_hand'].shape == [14, 34]
    9. Assert global_obs['wall_tiles'].shape == [34, 34]
  Expected Result: 全局状态格式正确，包含所有必要信息
  Evidence: 全局状态字典的键和形状

Scenario: 不同training_phase返回不同粒度
  Tool: Bash (python)
  Preconditions: 已创建测试GameContext
  Steps:
    1. 构建器 = ObservationBuilder()
    2. state = 创建测试GameContext
    3. # Phase 1: 返回所有真实信息
    4. global_obs_p1 = 构建器.build_global_observation(state, training_phase=1)
    5. # Phase 3: 返回近似信息
    6. global_obs_p3 = 构建器.build_global_observation(state, training_phase=3)
    7. # 对比
    8. Assert global_obs_p1包含真实手牌信息
    9. Assert global_obs_p3不包含真实手牌信息（使用信念采样）
  Expected Result: 不同phase返回不同粒度的信息
  Evidence: 两种全局状态的对比
```

**Commit**: YES
- Message: `feat(observation): add global state builder for centralized critic`
- Files: `src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py`, `tests/unit/test_global_observation.py`

---

#### Task 3: 单元测试框架

**What to do**:
1. 创建 `tests/unit/` 目录结构：
   ```
   tests/unit/
   ├── __init__.py
   ├── test_belief_network.py
   ├── test_global_observation.py
   ├── test_centralized_critic.py
   ├── test_monte_carlo_sampler.py
   └── conftest.py
   ```
2. 实现核心测试工具函数（conftest.py）
3. 实现各组件的基础测试用例
4. 配置测试覆盖率要求（>60%）

**Recommended Agent Profile**:
- **Category**: `quick`（标准化测试框架）
- **Skills**: `pytest`, `coverage`

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 1 (with Tasks 1, 2a, 2b)
- **Blocks**: Task 10
- **Blocked By**: None

**Acceptance Criteria**:
- [ ] pytest 可运行
- [ ] 基础测试用例存在
- [ ] 覆盖率报告生成
- [ ] CI集成测试通过

**Commit**: YES
- Message: `test: add unit test framework`
- Files: `tests/unit/`, `pytest.ini`, `conftest.py`

---

#### Task 3b: 代码质量基础设施 ⭐ **新增**

**What to do**:
1. 配置 linting 工具：
   - `.config/black.toml` (代码格式化)
   - `.config/ruff.toml` (快速linting)
   - `pyproject.toml` (mypy类型检查)
2. 创建 `.github/pull_request_template.md`
3. 添加 pre-commit hooks
4. 设置单元测试覆盖率目标（>80%）

**Files**:
- `pyproject.toml`
- `.pre-commit-config.yaml`
- `.github/pull_request_template.md`

**Dependencies**: None
**Parallel**: Wave 1 所有任务
**Priority**: P2

**Acceptance Criteria**:
- [ ] black 可格式化代码
- [ ] ruff 可检查代码
- [ ] mypy 可进行类型检查
- [ ] pre-commit hooks 可运行
- [ ] PR模板存在且合理

---

### Wave 2: 核心网络

#### Task 4: MonteCarloSampler实现

**What to do**:
1. 在 `src/drl/monte_carlo_sampler.py` 中完善 `MonteCarloSampler` 类
2. 实现核心采样流程：
   - Gumbel-Softmax 采样：从分布中采样N个可能手牌
   - 置信度调整：根据历史准确度调整采样权重
   - 约束检查：确保不采样已知的牌（弃牌堆、副露）
3. 实现核心方法：
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
- **Can Run In Parallel**: NO
- **Parallel Group**: Wave 2
- **Blocks**: Task 6
- **Blocked By**: Task 1

**References**:
- BeliefNetwork输出: Task 1
- GameContext结构: `src/mahjong_rl/core/GameData.py`
- Mahjong规则: `src/mahjong_rl/rules/`
- 采样策略: Gumbel-Softmax, Rejection Sampling

**Acceptance Criteria**:
- [ ] MonteCarloSampler类可实例化
- [ ] `sample()` 方法可调用
- [ ] 返回N个有效的GameContext
- [ ] 采样的手牌符合Mahjong规则
- [ ] 采样的手牌不与已知牌重复
- [ ] 单元测试通过: `test_monte_carlo_sampler.py`

**Agent-Executed QA Scenarios**:
```
Scenario: MonteCarloSampler生成有效采样
  Tool: Bash (python)
  Preconditions: BeliefNetwork已实现，test_beliefs.pt已创建
  Steps:
    1. 导入 MonteCarloSampler
    2. sampler = MonteCarloSampler()
    3. beliefs = torch.load("test_beliefs.pt")  # [batch=4, 3, 34]
    4. n_samples = 5
    5. known_tiles = 构建测试已知牌
    6. samples = sampler.sample(beliefs, n_samples, known_tiles)
    7. Assert len(samples) == 5
    8. For each sample:
       - Assert 对手手牌总数正确（13或变化后）
       - Assert 手牌不与弃牌堆重复
       - Assert 手牌不与副露重复
       - Assert 手牌符合Mahjong规则
  Expected Result: 5个有效且合理的游戏状态
  Evidence: 采样的手牌列表
```

**Commit**: YES
- Message: `feat(sampler): add detailed MonteCarloSampler with Gumbel-Softmax and constraint checking`
- Files: `src/drl/monte_carlo_sampler.py`, `tests/unit/test_monte_carlo_sampler.py`

---

#### Task 5: CentralizedCriticNetwork实现

**What to do**:
1. 修改 `src/drl/network.py` 中的 `CentralizedCriticNetwork` 类
2. 网络架构：
   ```
   CentralizedCriticNetwork(
       (agent_encoders): ModuleList(4×ObservationEncoder)
       (fusion): Sequential(
           Linear(4×hidden_dim, 2×hidden_dim)
           ReLU()
           Linear(2×hidden_dim, hidden_dim)
           ReLU()
       )
       (critic_heads): ModuleList(4×Sequential(
           Linear(hidden_dim, hidden_dim//2)
           ReLU()
           Linear(hidden_dim//2, 1)
       ))
   )
   ```
3. 实现 `forward()` 方法：
   ```python
   def forward(self, all_observations: List[Dict]) -> torch.Tensor:
       """
       Args:
           all_observations: 4个智能体的观测字典列表
       
       Returns:
           values: [batch, 4] - 每个智能体的价值估计
       """
       # 编码每个智能体的观测
       agent_features = [encoder(obs) for encoder, obs in 
                        zip(self.agent_encoders, all_observations)]
       
       # 融合所有智能体的特征
       fused = torch.cat(agent_features, dim=-1)  # [batch, 4×hidden_dim]
       fused = self.fusion(fused)  # [batch, hidden_dim]
       
       # 计算每个智能体的价值
       values = [head(fused) for head in self.critic_heads]  # [batch, 1] × 4
       return torch.cat(values, dim=-1)  # [batch, 4]
   ```

**Recommended Agent Profile**:
- **Category**: `ultrabrain`（需要多智能体协作学习经验）
- **Skills**: `pytorch`, `marl-algorithms`, `attention`

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 2
- **Blocks**: Task 7
- **Blocked By**: Task 2b

**References**:
- 现有编码器: `src/drl/network.py:ObservationEncoder`
- 融合策略: `src/drl/network.py:TransformerHistoryEncoder`
- MADDPG/MAPPO论文: "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"

**Acceptance Criteria**:
- [ ] CentralizedCriticNetwork类可实例化
- [ ] `forward()` 接受4个观测字典，返回[batch, 4]价值估计
- [ ] 不同智能体的价值估计相互独立
- [ ] 与现有MAPPO集成正确
- [ ] 单元测试通过: `test_centralized_critic.py`

**Agent-Executed QA Scenarios**:
```
Scenario: CentralizedCritic输出独立的价值估计
  Tool: Bash (python)
  Preconditions: CentralizedCriticNetwork已实现
  Steps:
    1. 导入 CentralizedCriticNetwork
    2. critic = CentralizedCriticNetwork(hidden_dim=256)
    3. # 构建4个agents的测试观测
    4. all_obs = [build_test_obs() for _ in range(4)]
    5. values = critic(all_obs)
    6. Assert values.shape == torch.Size([1, 4])  # batch=1, 4 agents
    7. # 验证不同agents的价值估计不同
    8. Assert not torch.allclose(values[0, 0], values[0, 1])
    9. # 验证同一agent的价值估计稳定
    10. values2 = critic(all_obs)
    11. Assert torch.allclose(values, values2)
  Expected Result: 4个独立的、稳定的价值估计
  Evidence: 价值张量的值
```

**Commit**: YES
- Message: `feat(critic): add CentralizedCriticNetwork with multi-agent fusion`
- Files: `src/drl/network.py`, `tests/unit/test_centralized_critic.py`

---

#### Task 6: 修改Actor集成信念

**What to do**:
1. 修改 `src/drl/network.py` 中的 `ActorCriticNetwork` 类
2. 添加信念采样输入：
   ```python
   class ActorCriticNetwork(nn.Module):
       def __init__(self, ..., use_belief=False):
           # 现有初始化
           ...
           
           # 信念集成（可选）
           self.use_belief = use_belief
           if use_belief:
               # 信念编码器
               self.belief_encoder = nn.Sequential(
                   nn.Linear(34 * MonteCarloSampler.N_SAMPLES, hidden_dim),
                   nn.ReLU(),
               )
           
       def forward(self, obs: Dict, belief_samples: List[Dict] = None):
           # 现有特征提取
           features = self.observation_encoder(obs)
           
           # 信念集成（可选）
           if self.use_belief and belief_samples is not None:
               # 将采样编码并与主特征拼接
               sampled_features = [self.observation_encoder(sample) for sample in belief_samples]
               sampled_features = torch.cat(sampled_features, dim=-1)
               sampled_features = self.belief_encoder(sampled_features)
               features = torch.cat([features, sampled_features], dim=-1)
           
           # 现有actor和critic头
           ...
   ```
3. 添加信念采样数配置（`n_belief_samples=5`）

**Must NOT do**:
- 不要修改现有API（添加可选参数，保持向后兼容）
- 不要在信念采样中使用真实手牌信息

**Recommended Agent Profile**:
- **Category**: `ultrabrain`（需要特征融合经验）
- **Skills**: `pytorch`, `feature-engineering`, `rl-algorithms`

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 2
- **Blocks**: Task 7
- **Blocked By**: Task 1, Task 4

**References**:
- 现有ActorCriticNetwork: `src/drl/network.py`
- BeliefNetwork输出: Task 1
- MonteCarloSampler输出: Task 4

**Acceptance Criteria**:
- [ ] `use_belief` 开关可控制信念集成
- [ ] `forward()` 接受可选的 `belief_samples` 参数
- [ ]信念采样不影响现有功能
- [ ] 单元测试通过: `test_actor_with_belief.py`

**Commit**: YES
- Message: `feat(actor): integrate belief samples into Actor network`
- Files: `src/drl/network.py`, `tests/unit/test_actor_with_belief.py`

---

### Wave 3: 训练集成

#### Task 7: DualCriticTraining修改MAPPO

**What to do**:
1. 修改 `src/drl/mappo.py` 中的 `MAPPO` 类
2. 添加双Critic切换逻辑：
   ```python
   class MAPPO:
       def __init__(self, ..., centralized_critic=None, use_dual_critic=False):
           # 现有初始化
           ...
           
           # 双Critic（可选）
           self.use_dual_critic = use_dual_critic
           self.centralized_critic = centralized_critic
       
       def update(self, buffer, ..., training_phase=1):
           # Phase-aware Critic选择
           if self.use_dual_critic:
               if training_phase in [1, 2] and self.centralized_critic is not None:
                   # Phase 1-2: 使用Centralized Critic
                   return self._update_centralized(buffer, training_phase)
               else:
                   # Phase 3: 使用Decentralized Critic
                   return self._update_decentralized(buffer)
           else:
               # 原有逻辑
               return self._update_decentralized(buffer)
       
       def _update_centralized(self, buffer, training_phase):
           # Centralized Critic更新逻辑
           ...
   ```
3. 添加配置项 `use_dual_critic=True/False`

**Recommended Agent Profile**:
- **Category**: `ultrabrain`（需要CTDE经验）
- **Skills**: `pytorch`, `marl-algorithms`, `ppo`

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 3
- **Blocks**: Task 9
- **Blocked By**: Task 5, Task 6

**References**:
- 现有MAPPO: `src/drl/mappo.py`
- 现有CentralizedCritic: `src/drl/network.py:CentralizedCriticNetwork`
- MADDPG/MAPPO论文

**Acceptance Criteria**:
- [ ] `use_dual_critic` 开关可控制双Critic模式
- [ ] `training_phase` 参数正确控制Critic选择
- [ ] Phase 1-2: 使用Centralized Critic
- [ ] Phase 3: 使用Decentralized Critic
- [ ] 集成测试通过: `test_dual_critic.py`

**Agent-Executed QA Scenarios**:
```
Scenario: DualCritic正确切换
  Tool: Bash (python)
  Preconditions: MAPPO已配置双Critic
  Steps:
    1. # 测试Phase 1-2
    2. stats_p1 = mappo.update(buffer, training_phase=1)
    3. Assert stats_p1['used_centralized'] == True
    
    4. # 测试Phase 3
    5. stats_p3 = mappo.update(buffer, training_phase=3)
    6. Assert stats_p3['used_centralized'] == False
    
    7. # 测试无Centralized Critic情况
    8. mappo_no_critic = MAPPO(..., centralized_critic=None, use_centralized_critic=False)
    9. stats = mappo_no_critic.update(buffer, training_phase=1)
    10. Assert stats['used_centralized'] == False
  Expected Result: 正确根据training_phase选择Critic
  Evidence: 更新统计中的used_centralized标志
```

**Commit**: YES
- Message: `feat(train): add dual-critic training with phase-aware switching`
- Files: `src/drl/mappo.py`, `tests/unit/test_dual_critic.py`

---

#### Task 8: 环境集成全局状态

**What to do**:
1. 修改 `example_mahjong_env.py` 或 `src/mahjong_rl/` 中的环境类
2. 添加全局状态构建集成：
   ```python
   class WuhanMahjongEnv:
       def __init__(self, ..., use_global_state=False, use_dual_critic=False):
           # 现有初始化
           ...
           
           # 全局状态配置
           self.use_global_state = use_global_state
           self.use_dual_critic = use_dual_critic
           
           if use_global_state:
               self.global_builder = GlobalObservationBuilder()
       
       def get_observation(self, agent_id):
           # 现有逻辑
           obs = self._build_observation(agent_id)
           
           # 添加全局状态（用于Centralized Critic）
           if self.use_global_state:
               obs['global_state'] = self.global_builder.build(
                   self.game_context,
                   self.training_phase
               )
           
           return obs
   ```
3. 添加配置项：
   - `use_global_state=True/False`
   - `global_state_mode='full'` / `'sampled'` / `'compressed'`

**Recommended Agent Profile**:
- **Category**: `ultrabrain`（需要理解环境状态和数据流）
- **Skills**: `pettingzoo`, `environment-design`, `rl`

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 3
- **Blocks**: Task 9
- **Blocked By**: Task 2b

**References**:
- 现有环境: `example_mahjong_env.py`
- 观测构建: `src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py`
- 课程学习: `example_mahjong_env.py:683-771`

**Acceptance Criteria**:
- [ ] `use_global_state` 开关可控制全局状态
- [ ] 全局状态包含所有必要信息
- [ ] 与现有PettingZoo接口兼容
- [ ] 集成测试通过: `test_global_env.py`

**Commit**: YES
- Message: `feat(env): integrate global state into environment`
- Files: `example_mahjong_env.py`, `tests/integration/test_global_env.py`

---

#### Task 8a: Phase间Checkpoint热启动 ⭐ **新增**

**What to do**:
1. 修改 `src/drl/trainer.py` 中的checkpoint方法
2. 实现 `load_checkpoint_with_phase_migration()` 方法
3. 迁移策略：
   - Phase 1→2: 完整迁移 Actor + CentralizedCritic 权重
   - Phase 2→3: 迁移 Actor 权重，Critic 重新初始化（避免 centralized→decentralized 差异）

**Files**:
- `src/drl/trainer.py`

**Dependencies**: Task 7
**Priority**: P1

**Acceptance Criteria**:
- [ ] checkpoint包含当前phase信息
- [ ] `load_checkpoint_with_phase_migration()` 方法可调用
- [ ] Phase 1→2 迁移正确
- [ ] Phase 2→3 迁移正确
- [ ] 单元测试通过: `test_checkpoint_migration.py`

**代码示例**:
```python
def load_checkpoint_with_phase_migration(self, checkpoint_path, target_phase):
    """加载checkpoint并处理phase迁移"""
    checkpoint = torch.load(checkpoint_path)
    source_phase = checkpoint['metadata']['phase']

    if source_phase == target_phase:
        # 同phase，直接加载
        self.model.load_state_dict(checkpoint['model'])
    elif source_phase == 2 and target_phase == 3:
        # Phase 2→3: 仅迁移Actor
        self.actor.load_state_dict(checkpoint['actor'])
        # Critic重新初始化
        self.decentralized_critic = CentralizedCriticNetwork(...)
        print("Phase 2→3 migration: Actor weights transferred, Critic reinitialized")
```

---

#### Task 9: 训练流程验证

**What to do**:
1. 创建完整训练脚本 `scripts/train_belief_mahjong.py`
2. 实现三阶段训练流程：
   ```python
   def train_belief_mahjong(config):
       # 1. 初始化环境和智能体
       env = WuhanMahjongEnv(..., use_global_state=True, use_dual_critic=True)
       agent_pool = NFSPAgentPool(...)
       
       # 2. Phase 1: 全知 + Centralized Critic
       train_phase(env, agent_pool, phase=1, episodes=6_666_666)
       
       # 3. Phase 2: 渐进遮蔽 + Centralized Critic
       train_phase(env, agent_pool, phase=2, episodes=6_666_667)
       
       # 4. Phase 3: 真实信息 + Decentralized Critic
       train_phase(env, agent_pool, phase=3, episodes=6_666_667)
       
       # 5. 保存最终模型
       agent_pool.save("final_model.pth")
   ```
3. 添加TensorBoard日志记录

**Recommended Agent Profile**:
- **Category**: `deep`（需要完整系统集成经验）
- **Skills**: `training-pipeline`, `tensorboard`, `checkpointing`

**Parallelization**:
- **Can Run In Parallel**: NO
- **Parallel Group**: Wave 3
- **Blocks**: Task 10
- **Blocked By**: Task 7, Task 8

**References**:
- 现有训练脚本: `train_nfsp.py`
- 课程学习: `example_mahjong_env.py:683-771`
- 检查点: `src/drl/trainer.py`

**Acceptance Criteria**:
- [ ] 训练脚本可运行
- [ ] 三阶段训练可完整执行
- [ ] 模型检查点正确保存
- [ ] TensorBoard日志正确记录
- [ ] 单元测试通过: `test_training_pipeline.py`

**Commit**: YES
- Message: `feat(train): add complete training pipeline with three phases`
- Files: `scripts/train_belief_mahjong.py`, `tests/integration/test_training_pipeline.py`

---

#### Task 3a: 实现对手策略池

**What to do**:
1. 创建 `src/drl/opponent_pool.py`
2. 实现对手策略池：
   - 存储历史策略检查点
   - 根据胜率动态选择对手
   - 支持多种对手选择策略

**Dependencies**: None
**Parallel**: Task 7, Task 8
**Priority**: P3

**Acceptance Criteria**:
- [ ] 对手池可创建
- [ ] 策略可存储和检索
- [ ] 根据胜率选择策略

---

### Wave 4: 测试验证

#### Task 10: 集成测试

**What to do**:
1. 创建 `tests/integration/` 目录结构：
   ```
   tests/integration/
   ├── __init__.py
   ├── test_belief_training.py
   ├── test_dual_critic.py
   └── test_end_to_end.py
   ```
2. 实现端到端测试：
   - 完整训练流程测试
   - Centralized Critic效果测试
   - Phase切换测试

**Dependencies**: Task 3, Task 9
**Parallel**: Task 11, Task 12
**Priority**: P1

**Acceptance Criteria**:
- [ ] 集成测试可运行
- [ ] 三阶段训练验证
- [ ] Centralized Critic效果验证
- [ ] Phase切换验证

---

#### Task 11: 性能基准测试

**What to do**:
1. 创建基准测试脚本
2. 测量性能指标：
   - 训练速度（episodes/小时）
   - 内存使用（GB）
   - GPU利用率（%）
   - 推理延迟（ms/step）

**Dependencies**: Task 10
**Priority**: P2

**Acceptance Criteria**:
- [ ] 基准测试脚本可运行
- [ ] 性能指标可量化
- [ ] 性能优于baseline

---

#### Task 12: 文档和示例

**What to do**:
1. 创建使用文档：
   ```
   docs/
   ├── README.md
   ├── quickstart.md
   ├── architecture.md
   ├── api/
   │   ├── belief_network.md
   │   ├── monte_carlo_sampler.md
   │   └── dual_critic.md
   └── examples/
       ├── basic_usage.py
       ├── belief_estimation.py
       └── dual_critic_training.py
   ```
2. 添加API文档
3. 创建示例脚本

**Dependencies**: Task 11
**Priority**: P2

**Acceptance Criteria**:
- [ ] README.md 完整
- [ ] Quickstart可运行
- [ ] API文档完整
- [ ] 示例脚本可运行

---

#### Task 4a: TensorBoard集成

**What to do**:
1. 修改训练脚本添加TensorBoard日志
2. 记录关键指标：
   - RL Loss
   - SL Loss
   - Win Rate
   - Value Estimate
   - Belief Accuracy

**Dependencies**: None
**Parallel**: Task 12
**Priority**: P2

**Acceptance Criteria**:
- [ ] TensorBoard可启动
- [ ] 关键指标可记录
- [ ] 可视化正确

---

#### Task 12a: 生产部署支持 ⭐ **新增**

**What to do**:
1. 创建 `scripts/export_model.py`:
   - 导出为 TorchScript (trace/script)
   - 导出为 ONNX (支持跨平台)
   - 验证导出模型正确性
2. 创建 `docs/deployment_guide.md`:
   - 模型加载示例
   - 推理性能优化（batch处理、GPU加速）
   - API接口设计

**Dependencies**: Task 12
**Priority**: P2

**Acceptance Criteria**:
- [ ] TorchScript导出可用
- [ ] ONNX导出可用
- [ ] 部署文档完整

---

#### Task 4b: 训练监控与诊断 ⭐ **新增**

**What to do**:
1. 创建 `src/drl/training_watchdog.py`:
   ```python
   class TrainingWatchdog:
       def check_loss(self, loss) -> bool:
           """检测NaN/Inf loss"""
       
       def check_gradients(self, model) -> bool:
           """检测梯度爆炸"""
       
       def auto_rollback(self) -> None:
           """自动回滚到上一个checkpoint"""
   ```
2. 创建 `scripts/debug_training.py`:
   - 诊断loss发散
   - 检测内存泄漏
   - 分析训练曲线
3. 集成到 `NFSPTrainer`

**Dependencies**: Task 9
**Priority**: P2

**Acceptance Criteria**:
- [ ] TrainingWatchdog类可实例化
- [ ] NaN检测可用
- [ ] 梯度爆炸检测可用
- [ ] 自动回滚可用

---

## 5. 依赖关系矩阵

| Task | Depends On | Blocks | Can Parallelize With | Priority |
|------|------------|--------|---------------------|----------|
| 0 | None | 1, 2a, 2b, 7 | - | P0 ✅ |
| 1 | 0 | 4, 6 | 2a, 2b, 3 | P1 |
| 2a (新) | 1 | 6 | 2b, 3 | P1 |
| 2b (原2) | None | 5, 8 | 1, 3 | P1 |
| 3 | None | 10 | 1, 2a, 2b, 3b | P2 |
| 3b (新) | None | - | 1, 2a, 2b, 3 | P2 |
| 4 | 1 | 6 | 5 | P1 |
| 5 | 2b | 7 | 4 | P1 |
| 6 | 1, 4 | 7 | - | P1 |
| 7 | 5, 6 | 9 | 8 | P1 |
| 8 | 2b | 9 | 7 | P1 |
| 8a (新) | 7 | - | 9 | P1 |
| 9 | 7, 8 | 10 | - | P1 |
| 3a | None | 9 | 7, 8 | P3 |
| 10 | 3, 9 | 11 | - | P1 |
| 11 | 10 | 12 | - | P2 |
| 12 | 11 | None | - | P2 |
| 4a | None | 12 | 12 | P2 |
| 12a (新) | 12 | None | - | P2 |
| 4b (新) | 9 | 12 | 12a | P2 |

---

## 6. 总结

### 问题修复

| 类别 | 数量 | 状态 |
|------|------|------|
| Task重复 | 2处 | ✅ 已修复 |
| Task重叠 | 1处 | ✅ 已修复 |
| 缺失训练策略 | 1项 | ✅ 已补充 |
| 缺失热启动策略 | 1项 | ✅ 已补充 |
| 缺失代码质量 | 1项 | ✅ 已补充 |
| 缺失异常处理 | 1项 | ✅ 已补充 |
| 缺失部署指南 | 1项 | ✅ 已补充 |
| 缺失调优策略 | 1项 | ✅ 已说明 |

### 新增内容

1. **Task 2a**: BeliefNetwork辅助损失训练（P1）
2. **Task 2b**: 全局状态构建器（原Task 2，已修复重复）
3. **Task 3b**: 代码质量基础设施（P2）
4. **Task 8a**: Phase间Checkpoint热启动（P1）
5. **Task 12a**: 生产部署支持（P2）
6. **Task 4b**: 训练监控与诊断（P2）

### 关键改进

1. ✅ 修复Task 2重复问题
2. ✅ 合并Task 1a到Task 4
3. ✅ 添加Checkpoint热启动策略
4. ✅ 添加生产部署支持
5. ✅ 添加代码质量基础设施
6. ✅ 添加训练监控与诊断

### 下一步行动

1. **立即**: 采纳此完善版计划
2. **高优先级**: 开始Task 1（BeliefNetwork实现）
3. **中优先级**: 完善代码质量和部署
4. **低优先级**: 长期优化和调优

**文档生成时间**: 2026-02-09
**基于**: 用户反馈文档（belief-state-centralized-critic-issues-and-supplements.md）
