# 信念状态与Centralized Critic计划 - 问题分析与补充建议

**分析日期**: 2026-02-09
**分析目标**: 审查 `.sisyphus/plans/belief-state-centralized-critic.md` 并识别缺失/问题
**当前实现状态**: Task 0 已完成（基础设施），核心功能尚未实现

---

## 一、现有文档问题

### 1.1 Task 2 重复内容（严重）

**位置**: 行483-541 和 行543-601

**问题**: Task 2 的内容被重复了两次，都是关于 BeliefNetwork 的实现

**原文**:
```
行483: - [ ] **Task 2: 全局状态构建器**
行484-541: 实际内容是 BeliefNetwork 实现（与Task 1重复）

行543: - [ ] **Task 2: 全局状态构建器**
行544-601: 全局状态构建器的实际内容
```

**影响**: 造成混淆，不清楚实际需要实现什么

**建议修正**:
```
Task 2a: BeliefNetwork辅助损失训练（新增）
Task 2b: 全局状态构建器（保持行543-601的内容）
```

---

### 1.2 Task 1a 和 Task 4 内容重叠（中等）

**问题**: 两个任务都涉及 MonteCarloSampler 实现

- **Task 1a** (行388-481): "MonteCarlo采样具体实现" - 详细实现
- **Task 4** (行639-709): "MonteCarloSampler实现" - 基础实现

**影响**: 任务边界模糊，可能造成重复工作

**建议合并**:
```
Task 4: MonteCarloSampler实现（保留行639-709作为基础实现）
- 删除 Task 1a 的独立条目
- 将 Task 1a 中的详细需求（Gumbel-Softmax、约束检查）整合到 Task 4 中
```

---

## 二、缺失的关键内容

### 2.1 BeliefNetwork 训练策略（高优先级）

**问题**: 计划中 BeliefNetwork 只有架构设计，缺少训练策略

**缺失内容**:
1. BeliefNetwork 如何获得训练信号？
2. 是否需要预训练？
3. loss 函数是什么？

**建议补充 - Task 2a**:

```markdown
- [ ] **Task 2a: BeliefNetwork辅助损失训练（P1优先级）**

  **What to do**:
  1. 在 BeliefNetwork 中添加辅助loss方法
  2. 实现3个辅助预测任务：
     - 预测对手下一轮打出的牌（34分类）
     - 预测对手是否吃/碰/杠（4分类）
     - 预测对手手牌总数（回归）
  3. 总loss = 0.7 × 主任务loss（RL） + 0.3 × 辅助loss
  4. 添加开关 use_auxiliary_loss=True/False

  **文件**: src/drl/belief_network.py

  **依赖**: Task 1 (BeliefNetwork基础实现)
  **并行**: Task 2b, Task 3
```

---

### 2.2 Checkpoint 热启动策略（高优先级）

**问题**: 不同 phase 之间如何迁移模型权重未明确

**当前状态**: NFSPTrainer 已支持 checkpoint 保存/恢复，但未说明 phase 间迁移策略

**建议补充 - Wave 3 新任务**:

```markdown
- [ ] **Task 8a: Phase间Checkpoint热启动（P1优先级）**

  **What to do**:
  1. 修改 NFSPTrainer.checkpoint() 保存当前 phase 信息
  2. 实现 load_checkpoint_with_phase_migration() 方法
  3. 迁移策略：
     - Phase 1→2: 完整迁移 Actor + CentralizedCritic 权重
     - Phase 2→3: 迁移 Actor 权重，Critic 重新初始化（避免 centralized→decentralized 差异）
  4. 添加迁移验证：检查权重形状匹配

  **文件**: src/drl/trainer.py

  **依赖**: Task 7 (DualCriticTraining)
  **优先级**: P1（影响训练连续性）
```

**关键代码示例**:
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

### 2.3 生产部署指南（中优先级）

**问题**: 计划中没有模型导出和部署相关内容

**建议补充 - Wave 4 新任务**:

```markdown
- [ ] **Task 12a: 生产部署支持（P2优先级）**

  **What to do**:
  1. 创建 scripts/export_model.py:
     - 导出为 TorchScript (trace/script)
     - 导出为 ONNX (支持跨平台)
     - 验证导出模型正确性
  2. 创建 docs/deployment_guide.md:
     - 模型加载示例
     - 推理性能优化（batch处理、GPU加速）
     - API接口设计
  3. 添加推理基准测试

  **文件**:
  - scripts/export_model.py
  - docs/deployment_guide.md

  **依赖**: Task 12 (基础文档)
  **优先级**: P2（训练后需要）
```

---

### 2.4 代码质量流程（中优先级）

**问题**: 计划中没有代码质量保障内容

**建议补充 - Wave 1 新任务**:

```markdown
- [ ] **Task 3b: 代码质量基础设施（P2优先级）**

  **What to do**:
  1. 配置 linting 工具:
     - .config/black.toml (代码格式化)
     - .config/ruff.toml (快速linting)
     - pyproject.toml (mypy类型检查)
  2. 创建 .github/pull_request_template.md
  3. 添加 pre-commit hooks
  4. 设置单元测试覆盖率目标 (>80%)

  **文件**:
  - pyproject.toml
  - .pre-commit-config.yaml
  - .github/pull_request_template.md

  **依赖**: 无
  **并行**: Wave 1 所有任务
  **优先级**: P2
```

---

### 2.5 异常处理与诊断（中优先级）

**问题**: 训练异常情况处理未覆盖

**建议补充 - Wave 4 新任务**:

```markdown
- [ ] **Task 4b: 训练监控与诊断（P2优先级）**

  **What to do**:
  1. 创建 src/drl/training_watchdog.py:
     ```python
     class TrainingWatchdog:
         def check_loss(self, loss) -> bool:
             """检测NaN/Inf loss"""
         def check_gradients(self, model) -> bool:
             """检测梯度爆炸"""
         def auto_rollback(self) -> None:
             """自动回滚到上一个checkpoint"""
     ```
  2. 创建 scripts/debug_training.py:
     - 诊断loss发散
     - 检测内存泄漏
     - 分析训练曲线
  3. 集成到 NFSPTrainer

  **文件**:
  - src/drl/training_watchdog.py
  - scripts/debug_training.py

  **依赖**: Task 9 (训练流程验证)
  **优先级**: P2
```

---

### 2.6 超参数调优策略（低优先级）

**问题**: 计划中没有超参数调优内容

**建议补充 - Wave 4 说明**:

```markdown
### 超参数调优策略

**策略**: 渐进式调优

1. **第一轮**: 使用默认值跑通完整流程
   - 验证所有功能正常工作
   - 记录baseline性能

2. **第二轮**: 根据结果调整关键参数
   - learning rate: [1e-4, 3e-4, 1e-3]
   - n_samples (MonteCarlo): [3, 5, 10]
   - auxiliary_loss_weight: [0.1, 0.3, 0.5]

3. **可选第三轮**: 使用 Optuna 自动搜索
   - 仅在第二轮效果不佳时使用
```

---

## 三、文档结构问题

### 3.1 Wave 0 缺少 Acceptance Criteria

**问题**: Task 0 已完成，但文档中的 Acceptance Criteria 与实际完成内容不完全匹配

**当前状态**:
- 计划: `test_dual_critic.py` 单元测试
- 实际: `test_centralized_simple.py` 集成测试

**建议**: 更新文档以反映实际完成状态

---

### 3.2 缺少任务优先级标记

**问题**: 计划中有 P0/P1/P2/P3 优先级，但未在 Wave 概览中体现

**建议**: 在 Wave 执行策略中添加优先级说明

```markdown
Wave 0 (核心问题修复 - 3天) - [P0]
├── Task 0: 修复CentralizedCritic未实际使用问题 [P0] ✅ 已完成

Wave 1 (基础设施 - 5-6天):
├── Task 1: BeliefNetwork实现 [P1]
├── Task 2a: BeliefNetwork辅助损失 [P1] (新增)
├── Task 2b: 全局状态构建器 [P1]
├── Task 3: 单元测试框架 [P2]
├── Task 3b: 代码质量基础设施 [P2] (新增)

Wave 2 (核心网络 - 5-7天):
├── Task 4: MonteCarloSampler实现 [P1]
├── Task 5: CentralizedCriticNetwork实现 [P1]
├── Task 6: 修改Actor集成信念 [P1]

Wave 3 (训练集成 - 6-8天):
├── Task 7: DualCriticTraining修改MAPPO [P1]
├── Task 8: 环境集成全局状态 [P1]
├── Task 8a: Phase间Checkpoint热启动 [P1] (新增)
├── Task 9: 训练流程验证 [P1]
├── Task 3a: 实现对手策略池 [P3]

Wave 4 (测试验证 - 5-7天):
├── Task 10: 集成测试 [P1]
├── Task 11: 性能基准测试 [P2]
├── Task 12: 文档和示例 [P2]
├── Task 4a: TensorBoard集成 [P2]
├── Task 12a: 生产部署支持 [P2] (新增)
├── Task 4b: 训练监控与诊断 [P2] (新增)
```

---

## 四、依赖关系更新

### 原依赖矩阵问题

**问题**: 依赖矩阵未包含新增任务

**更新后的依赖矩阵**:

| Task | Depends On | Blocks | Can Parallelize With | Priority |
|------|------------|--------|---------------------|----------|
| 0 | None | 1, 1a, 2, 7 | - | P0 ✅ |
| 1 | 0 | 4, 6 | 2a, 2b, 3 | P1 |
| 2a (新) | 1 | 6 | 2b, 3 | P1 |
| 2b | None | 5, 8 | 1, 3 | P1 |
| 3 | None | All tests | 1, 2a, 2b, 3b | P2 |
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

## 五、建议的修订顺序

### 第一阶段：修复文档问题
1. 删除行483-541的重复Task 2内容
2. 将Task 1a整合到Task 4
3. 添加Task 2a（BeliefNetwork辅助损失）
4. 更新Wave概览和依赖矩阵

### 第二阶段：补充高优先级内容
1. 添加Task 8a（Checkpoint热启动）
2. 完善Task 2a的详细内容
3. 添加超参数调优策略说明

### 第三阶段：补充中低优先级内容
1. 添加Task 3b（代码质量）
2. 添加Task 4b（训练监控）
3. 添加Task 12a（生产部署）

---

## 六、总结

### 问题统计

| 类别 | 数量 | 严重程度 |
|------|------|----------|
| 重复内容 | 2处 | 高 |
| 缺失训练策略 | 1项 | 高 |
| 缺失热启动策略 | 1项 | 高 |
| 缺失代码质量 | 1项 | 中 |
| 缺失异常处理 | 1项 | 中 |
| 缺失部署指南 | 1项 | 中 |
| 缺失调优策略 | 1项 | 低 |

### 建议行动

1. **立即**: 修复Task 2重复问题
2. **高优先级**: 添加Task 2a和Task 8a
3. **中优先级**: 添加代码质量和监控任务
4. **低优先级**: 完善部署和调优文档

---

**文档生成时间**: 2026-02-09
**分析工具**: Claude Code
**下一步**: 等待用户确认后更新原计划文档
