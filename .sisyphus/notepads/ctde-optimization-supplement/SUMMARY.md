# CTDE 优化补充计划 - 总结文档

**日期**: 2025-02-09
**用户**: 汪呜呜
**状态**: ✅ 补充计划已完成

---

## 📋 用户确认的决策

### Q1: CentralizedCritic 输入方式
**选择**: 选项A（激进）- 完整全局状态作为输入
- 训练时 centralized critic 接收完整全局状态（4玩家手牌+牌墙+公共信息）
- Phase 1-2 使用 centralized critic
- Phase 3 使用 decentralized critic（仅局部观测）
- **优点**: 最大化利用全局信息
- **缺点**: 训练-执行差距大、状态空间爆炸（>1500维）

### Q2: 信念状态表示方式
**选择**: 选项C - 采样表示（N个可能状态）
- 从概率分布采样N个可能对手手牌状态
- 平均处理采样结果输入到 Actor
- **优点**: 直接、计算密集、精确
- **缺点**: 计算开销大
- **采样数配置**: N=5-10（可配置）

### Q3: 训练策略
**选择**: 结合两者优势
- Dual-Critic：训练时用 centralized，执行时用 decentralized
- 结合三阶段课程学习（Phase 1-2 centralized，Phase 3 decentralized）
- 保留现有课程学习基础，增加 dual-critic 切换
- **优点**: 平衡、平滑过渡、最佳实践

---

## ✅ 已完成的补充内容

### 1. 新增Wave和任务

#### Wave 0: 核心问题修复（新增）
- **Task 0**: 修复CentralizedCritic未实际使用问题（P0优先级）
  - 修改 NFSPAgentPool 收集全局观测
  - 完善 CentralizedRolloutBuffer
  - 修改 MAPPO 支持 dual-critic
  - 实现 phase-aware critic 切换

#### Wave 1: 基础设施（增强）
- **Task 1**: BeliefNetwork实现（增强，包含贝叶斯更新）
  - 新增贝叶斯更新公式
  - 新增对手动作响应更新逻辑
- **Task 1a**: MonteCarlo采样具体实现（新增，P1优先级）
  - Gumbel-Softmax 采样
  - 约束检查（不采样已知牌）
  - 置信度调整
- **Task 2**: 全局状态构建器
- **Task 3**: 单元测试框架

#### Wave 2: 核心网络（保持不变）
- **Task 4**: MonteCarloSampler实现
- **Task 5**: CentralizedCriticNetwork实现
- **Task 6**: 修改Actor集成信念

#### Wave 3: 训练集成（增强）
- **Task 7**: DualCriticTraining修改MAPPO
- **Task 8**: 环境集成全局状态
- **Task 9**: 训练流程验证
- **Task 3a**: 实现对手策略池（新增，P3优先级）
  - PolicyPool 类实现
  - 策略添加、采样、检索
  - 基于性能的加权采样

#### Wave 4: 测试验证（增强）
- **Task 10**: 集成测试
- **Task 11**: 性能基准测试
- **Task 12**: 文档和示例
- **Task 4a**: TensorBoard集成和性能监控（新增，P2优先级）
  - TensorBoardLogger 类实现
  - PerformanceMonitor 类实现
  - 信念分布可视化
  - 训练速度和内存监控

---

### 2. 详细实现细节

#### P1.1: 贝叶斯更新公式（补充到Task 1）

**先验更新公式**:
```
P(t|E) ∝ P(E|t) × L(E|t)
```

**具体更新规则**:
```python
# 打出牌 d
for opponent_id in range(3):
    if action_type[opponent_id] == DISCARD and action_param[opponent_id] == d:
        beliefs[opponent_id, d] *= 0.1

# 碰牌 p
for opponent_id in range(3):
    if action_type[opponent_id] == PONG and action_param[opponent_id] == p:
        beliefs[opponent_id, p] *= 1.5

# 杠牌 k
for opponent_id in range(3):
    if action_type[opponent_id] in [KONG_EXPOSED, KONG_CONCEALED]:
        beliefs[opponent_id, action_param[opponent_id]] *= 2.0

# 贝叶斯归一化
sum_beliefs = beliefs.sum(dim=-1, keepdim=True)
normalized_beliefs = beliefs / sum_beliefs
```

#### P1.2: 蒙特卡罗采样具体实现（补充到Task 1a）

**Gumbel-Softmax 采样流程**:
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

#### P2.1: TensorBoard 集成（补充到Task 4a）

**核心类实现**:
```python
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

class TensorBoardLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量指标"""
        self.writer.add_scalar(tag, value, self.step)

    def log_belief_distribution(self, beliefs: torch.Tensor, step: int):
        """记录对手手牌信念分布"""
        # beliefs: [batch, 3, 34] - 3个对手 × 34种牌
        for opponent_id in range(3):
            for tile_id in range(34):
                self.writer.add_scalar(
                    f'belief/opponent_{opponent_id}/tile_{tile_id}',
                    beliefs[step, opponent_id, tile_id].item(),
                    step
                )

    def close(self):
        self.writer.close()
```

#### P2.2: 性能监控实现（补充到Task 4a）

**核心类实现**:
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

#### P3.1: 对手策略池实现（补充到Task 3a）

**核心类实现**:
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
        if len(self.policies) >= self.capacity:
            self.policies.pop(0)

        policy_id = self.next_id
        self.next_id += 1

        policy_data = {
            'id': policy_id,
            'state_dict': policy['state_dict'],
            'samples_used': samples,
            'added_at': datetime.now().isoformat()
        }

        self.policies.append(policy_data)
        return policy_id

    def sample_policy(self, k: int = 1, weights: Optional[List[float]] = None) -> Dict:
        """从池中采样策略"""
        if not self.policies:
            raise ValueError("Policy pool is empty")

        # 确保使用次数最少的策略
        candidates = sorted(self.policies, key=lambda p: p['samples_used'])
        selected = candidates[:k]

        # 如果提供权重，使用加权采样
        if weights is not None:
            # 使用 softmax 归一化
            total_samples = sum(p['samples_used'] for p in selected)
            probs = [w / total_samples for w in weights]
            selected_idx = np.random.choice(len(selected), p=probs)
            return selected[selected_idx]
        else:
            return np.random.choice(selected)

    def get_policy(self, policy_id: int) -> Dict:
        """获取指定策略"""
        for policy in self.policies:
            if policy['id'] == policy_id:
                return policy
        raise ValueError(f"Policy {policy_id} not found")
```

---

### 3. 更新的依赖矩阵

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 0 (Fix CentralizedCritic) | None | 1, 1a, 2, 7 | - |
| 1 (BeliefNetwork) | 0 | 4, 6 | 2, 3 |
| 1a (MonteCarloDetails) | 1 | 6 | 2, 3 |
| 2 (GlobalStateBuilder) | 0 | 5, 8 | 1, 3 |
| 3 (Test Framework) | None | All tests | 1, 2 |
| 4 (MonteCarloSampler) | 1, 1a | 6 | 5 |
| 5 (CentralizedCritic) | 2 | 7 | 4 |
| 6 (Modified Actor) | 1, 4 | 7 | - |
| 7 (DualCriticTraining) | 0, 5, 6 | 9 | 8 |
| 8 (Env Integration) | 2 | 9 | 7 |
| 9 (Training Validation) | 7, 8 | 10 | - |
| 3a (PolicyPool) | None | 9 | 4, 5 |
| 10 (Integration Tests) | 3, 9 | 11 | - |
| 11 (Benchmark) | 10 | 12 | - |
| 12 (Documentation) | 11 | None | - |
| 4a (TensorBoard & Monitor) | None | 12 | - |

---

### 4. 更新的提交策略表

| 任务 | 提交信息 | 文件 | 验证命令 |
|------|----------|------|----------|
| 0 | `fix(architecture): implement dual-critic training` | agent.py, buffer.py, mappo.py, trainer.py | `pytest tests/unit/test_dual_critic.py` |
| 1 | `feat(belief): add BeliefNetwork with Bayesian update` | belief_network.py | `pytest tests/unit/test_belief_network.py` |
| 1a | `feat(sampler): add detailed MonteCarloSampler` | monte_carlo_sampler.py | `pytest tests/unit/test_sampler.py` |
| 2 | `feat(observation): add global state builder` | observation_builder.py | `pytest tests/unit/test_observation.py` |
| 3 | `test: add test framework` | test_*.py | `pytest tests/unit/` |
| 4 | `feat(sampler): add MonteCarloSampler` | monte_carlo_sampler.py | `pytest tests/unit/test_sampler.py` |
| 5 | `feat(critic): add CentralizedCriticNetwork` | centralized_critic.py | `pytest tests/unit/test_critic.py` |
| 6 | `feat(actor): integrate belief sampling` | network.py | `pytest tests/unit/test_actor.py` |
| 7 | `feat(training): implement DualCriticTraining` | mappo.py | `pytest tests/unit/test_mappo.py` |
| 8 | `feat(env): integrate global state` | example_mahjong_env.py | `pytest tests/integration/test_env.py` |
| 9 | `feat(training): add training script` | train_dual_critic.py | `python train_dual_critic.py --episodes 10` |
| 3a | `feat(policy): add PolicyPool for self-play` | policy_pool.py | `pytest tests/unit/test_policy_pool.py` |
| 10 | `test: add integration tests` | test_belief_critic.py | `pytest tests/integration/` |
| 11 | `benchmark: add performance comparison` | compare_baseline.py | `python compare_baseline.py --episodes 100` |
| 12 | `docs: add architecture documentation` | belief_critic_architecture.md | 人工审查 |
| 4a | `feat(monitor): add TensorBoard and monitoring` | tensorboard_logger.py, performance_monitor.py | `pytest tests/unit/test_monitoring.py` |

---

## 📊 工作量统计

### 原计划
- Wave 1: 3-5天（3个任务）
- Wave 2: 5-7天（3个任务）
- Wave 3: 4-6天（3个任务）
- Wave 4: 3-4天（3个任务）
- **原总计**: 15-22天（12个任务）

### 补充内容
- Wave 0: 3天（1个任务，P0）
- Wave 1 增强: +1天（Task 1a，P1）
- Wave 3 增强: +1天（Task 3a，P3）
- Wave 4 增强: +3.5天（Task 4a，P2）
- **补充总计**: 8.5天（4个补充任务）

### 更新后总工作量
- **Wave 0**: 3天（1个任务）
- **Wave 1**: 6-7天（4个任务）
- **Wave 2**: 5-7天（3个任务）
- **Wave 3**: 7-9天（4个任务）
- **Wave 4**: 6.5-9.5天（4个任务）
- **总合计**: 27.5-35.5天（16个任务）

---

## 🎯 关键路径

### 最长依赖链
**Task 0** → **Task 1** → **Task 4** → **Task 6** → **Task 7** → **Task 9** → **Task 10** → **Task 11** → **Task 12**

这条路径涉及9个任务，每个任务之间的依赖都必须完成。

### 关键路径耗时估算
- Task 0: 3天
- Task 1: 2天
- Task 4: 1.5天
- Task 6: 2天
- Task 7: 3天
- Task 9: 2天
- Task 10: 2天
- Task 11: 2天
- Task 12: 1天
- **关键路径总计**: 约18.5天（按顺序执行）

### 并行加速潜力
通过Wave内并行执行，可以节省约30%的时间：
- Wave 1中: Task 1, 1a, 2, 3可以部分并行（5-6天而非8-10天）
- Wave 2中: Task 4, 5可以并行（6-7天而非8-10天）
- Wave 3中: Task 7, 8, 3a可以部分并行（7-8天而非10-13天）
- Wave 4中: Task 4a可以独立并行执行

**最终估算**: 约19-24天（全职开发）

---

## ⚠️ 风险与缓解

### 高风险（需要特别关注）

1. **训练不稳定**（Task 0, Task 7）
   - **风险**: Centralized critic 可能导致训练发散
   - **缓解**: 更小的学习率（3e-4 → 1e-4），gradient clipping（max_grad_norm=0.5）
   - **缓解**: Phase 2 渐进过渡，避免突然切换

2. **计算开销过大**（Task 1a, Task 4a）
   - **风险**: 蒙特卡罗采样 + dual-critic + TensorBoard 显著增加训练时间
   - **缓解**: 采样数可配置（N=5-10），GPU 批处理
   - **缓解**: 异步采样，与训练并行
   - **缓解**: TensorBoard记录频率降低（每100步而非每步）

3. **训练-执行差距**（Task 0, Task 7）
   - **风险**: centralized critic 和 decentralized critic 差异过大
   - **缓解**: Phase 2 渐进式掩码，平滑过渡
   - **缓解**: Dual-critic 切换时使用混合权重

### 中风险

1. **信念估计不准确**（Task 1）
   - **风险**: 初期信念质量差，影响采样质量
   - **缓解**: 贝叶斯更新，Transformer 时序建模
   - **缓解**: 初期依赖公共信息（Phase 1）

2. **内存不足**（Task 1a, Task 4a）
   - **风险**: 存储全局状态、采样状态、TensorBoard 日志
   - **缓解**: 及时释放，使用 float16，采样数控制
   - **缓解**: 检查点间隔增加

3. **策略池管理复杂**（Task 3a）
   - **风险**: 策略池可能引入策略不稳定性
   - **缓解**: 最小样本数控制（min_samples=100）
   - **缓解**: 加权采样优先使用稳定策略

---

## ✅ 完成检查清单

### 文件更新
- [x] 计划文件已更新（.sisyphus/plans/belief-state-centralized-critic.md）
- [x] Draft文件已创建（.sisyphus/drafts/ctde-optimization-supplement.md）
- [x] 补充总结文档已创建（本文件）

### 内容完整性
- [x] Wave 0已添加（核心问题修复）
- [x] Task 0已添加（修复CentralizedCritic）
- [x] Task 1已增强（包含贝叶斯更新）
- [x] Task 1a已添加（蒙特卡罗采样细节）
- [x] Task 3a已添加（策略池实现）
- [x] Task 4a已添加（TensorBoard和性能监控）
- [x] 依赖矩阵已更新
- [x] 提交策略表已更新
- [x] 每个补充任务都有详细的Agent-Executed QA Scenarios

### 详细实现
- [x] 贝叶斯更新公式已补充
- [x] Gumbel-Softmax 采样流程已补充
- [x] TensorBoardLogger 类设计已补充
- [x] PerformanceMonitor 类设计已补充
- [x] PolicyPool 类设计已补充

---

## 🚀 下一步行动

### 立即可开始

1. **Wave 0**（核心问题修复）
   - Task 0: 修复CentralizedCritic未实际使用问题（3天）
   - 这是最高优先级，必须最先完成

2. **Wave 1**（基础设施）
   - Task 1: BeliefNetwork实现（包含贝叶斯更新）
   - Task 1a: MonteCarlo采样具体实现
   - Task 2: 全局状态构建器
   - Task 3: 单元测试框架

### 用户需要确认

汪呜呜，补充计划已完成。请确认：

1. ✅ 所有补充内容是否符合你的期望？
2. ✅ 是否需要调整任何任务的优先级或工作量？
3. ✅ 是否准备好开始执行（运行 `/start-work`）？

确认后，可以立即开始实施。

---

## 📝 文档引用

### 计划文件
- 主计划: `.sisyphus/plans/belief-state-centralized-critic.md`（已更新）
- 补充草案: `.sisyphus/drafts/ctde-optimization-supplement.md`（新建）
- 补充总结: 本文件（新建）

### 问题分析
- CentralizedCritic问题: `.sisyphus/notepads/nfsp_mappo_curriculum_implementation/centralized_critic_issue.md`
- NFSP完成报告: `.sisyphus/notepads/nfsp_mappo_curriculum_implementation/FINAL_COMPLETION_REPORT.md`

### 代码参考
- 现有网络: `src/drl/network.py`
- 现有MAPPO: `src/drl/mappo.py`
- 现有Trainer: `src/drl/trainer.py`
- 现有AgentPool: `src/drl/agent.py`
- 现有Buffer: `src/drl/buffer.py`

---

**补充计划状态**: ✅ 完成
**准备开始执行**: 是
**建议执行命令**: `/start-work`
