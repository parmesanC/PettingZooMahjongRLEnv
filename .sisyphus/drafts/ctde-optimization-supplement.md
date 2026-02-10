# Draft: CTDE 优化补充计划

## 用户确认的决策

**日期**: 2025-02-09
**用户**: 汪呜呜

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

## 需要补充的内容

### P0（高优先级）- 核心问题修复

#### P0.1: 修复 CentralizedCritic 未实际使用问题

**问题诊断**（来自 centralized_critic_issue.md）：
```python
# 当前问题链
env.last() → 全局观测 ✅
  ↓
agent.choose_action(obs) → NFSP.select_action(obs) ✅
  ↓
agent.store_transition(obs) → NFSP.store_transition(obs) ⚠️ 只存当前agent观测
  ↓
buffer.append(obs) → 只有当前agent观测 ⚠️
  ↓
mappo.train_step() → _prepare_obs_batch() ⚠️ 只用当前agent观测
  ↓
critic.forward() → 只看到当前agent ⚠️
```

**解决方案**：
1. 修改 `NFSPAgentPool` 添加收集全局观测的方法
2. 修改 `CentralizedRolloutBuffer` 存储全局观测
3. 修改 `MAPPO.train_step()` 传递全局观测给 centralized critic
4. 创建训练阶段判断逻辑（Phase 1-2 用 centralized，Phase 3 用 decentralized）

**预计工作量**: 3 天

---

### P1（高优先级）- 信念系统实现细节

#### P1.1: 贝叶斯更新公式

**先验更新公式**：
```
P(t|E) ∝ P(E|t) × L(E|t)
```

其中：
- `P(t)`: 对手手牌的先验分布
- `E`: 观测证据（对手动作）
- `L(E|t)`: 似然函数

**具体更新规则**：

1. **打出牌 d**：对手打出牌 d
   ```python
   # 降低该对手持有牌 d 的概率
   for opponent_id in range(3):
       if action_type[opponent_id] == DISCARD and action_param[opponent_id] == d:
           beliefs[opponent_id, d] *= 0.1
   ```

2. **碰牌 p**：对手碰某张牌
   ```python
   # 推断该对手可能持有该牌
   for opponent_id in range(3):
       if action_type[opponent_id] == PONG and action_param[opponent_id] == p:
           # 提高该对手持有牌 p 的概率
           beliefs[opponent_id, p] *= 1.5
   ```

3. **杠牌 k**：对手杠某张牌
   ```python
   # 推断该对手可能拥有该牌
   for opponent_id in range(3):
       if action_type[opponent_id] in [KONG_EXPOSED, KONG_CONCEALED]:
           beliefs[opponent_id, action_param[opponent_id]] *= 2.0
   ```

4. **贝叶斯归一化**：
   ```python
   sum_beliefs = beliefs.sum(dim=-1, keepdim=True)
   normalized_beliefs = beliefs / sum_beliefs
   ```

**预计工作量**: 2 天

#### P1.2: 蒙特卡罗采样具体实现

**采样策略**：
1. **Gumbel-Softmax 采样**：从分布中采样N个可能手牌
2. **置信度调整**：根据历史准确度调整采样权重
3. **约束检查**：确保不采样已知的牌（弃牌堆、副露）

**采样流程**：
```python
def sample(beliefs: torch.Tensor, n_samples: int, known_tiles: torch.Tensor) -> List[GameContext]:
    """
    Args:
        beliefs: [batch, 3, 34] - 3个对手的概率分布
        n_samples: 采样数量
        known_tiles: [batch, 34] - 已知的牌（弃牌堆+副露）
    Returns:
        N个采样的GameContext，每个包含采样的对手手牌
    """
    samples = []
    for _ in range(n_samples):
        # 1. Gumbel-Softmax 采样
        gumbel = -torch.log(-torch.log(torch.rand_like(beliefs)))
        sampled_indices = torch.argmax(beliefs + gumbel, dim=-1)

        # 2. 掩码已知的牌
        sampled_indices = sampled_indices * (1 - known_tiles.int())

        # 3. 构建采样的GameContext
        sampled_context = self._build_sampled_context(sampled_indices)

        # 4. 约束检查（手牌数、规则符合性）
        if self._validate_sample(sampled_context):
            samples.append(sampled_context)

    return samples
```

**预计工作量**: 1 天

---

### P2（中优先级）- 监控系统

#### P2.1: TensorBoard 集成

**实现文件**: `src/drl/tensorboard_logger.py`

**核心类**：
```python
class TensorBoardLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_belief_distribution(self, beliefs: torch.Tensor, step: int):
        """记录对手手牌信念分布"""
        for opponent_id in range(3):
            for tile_id in range(34):
                self.writer.add_scalar(
                    f'belief/opponent_{opponent_id}/tile_{tile_id}',
                    beliefs[step, opponent_id, tile_id].item(),
                    step
                )
```

**集成到 NFSPTrainer**：
```python
class NFSPTrainer:
    def __init__(self, ...):
        # ... 现有代码
        self.tb_logger = TensorBoardLogger(self.log_dir)

    def train(self):
        # ... 训练循环
        self.tb_logger.log_belief_distribution(
            beliefs=latest_beliefs,
            step=self.episode_count
        )
```

**预计工作量**: 2 天

#### P2.2: 性能监控实现

**监控指标**：
- 训练速度（episodes/hour）
- 内存使用（GB）
- GPU 利用率（%）
- 信念收敛速度
- 价值估计方差

**实现文件**: `src/drl/performance_monitor.py`

**预计工作量**: 1.5 天

---

### P3（低优先级）- 对手策略池

#### P3.1: 策略池具体实现

**实现文件**: `src/drl/policy_pool.py`

**核心类**：
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

**预计工作量**: 1 天

---

## 集成点确认

### 1. NFSPAgentPool 修改点
- 文件: `src/drl/agent.py`
- 方法: `store_transition()` 需要改为存储全局观测
- 新增: `store_global_transition(all_observations, ...)`

### 2. CentralizedRolloutBuffer 修改点
- 文件: `src/drl/buffer.py`
- 现有: `get_centralized_batch()` 方法已存在但未完全实现
- 修改: 完善全局观测存储和检索逻辑

### 3. MAPPO 修改点
- 文件: `src/drl/mappo.py`
- 方法: `train_step()` 需要支持 dual-critic
- 新增: `dual_critic_update(centralized_obs, decentralized_obs, phase)` 方法

### 4. WuhanMahjongEnv 修改点
- 文件: `example_mahjong_env.py`
- 方法: `reset()` 和 `step()` 需要返回 `global_observation` 在 info 字典中
- 新增: `build_global_observation(player_id)` 辅助方法

---

## 风险评估

### 高风险
1. **训练不稳定**: Centralized critic 可能导致训练发散
   - **缓解**: 更小的学习率（3e-4 → 1e-4），gradient clipping（max_grad_norm=0.5）
   - **缓解**: Phase 2 渐进过渡，避免突然切换

2. **计算开销过大**: 蒙特卡罗采样 + dual-critic 显著增加训练时间
   - **缓解**: 采样数可配置（N=5-10），GPU 批处理
   - **缓解**: 异步采样，与训练并行

3. **训练-执行差距**: centralized critic 和 decentralized critic 差异过大
   - **缓解**: Phase 2 渐进式掩码，平滑过渡
   - **缓解**: Dual-critic 切换时使用混合权重

### 中风险
1. **信念估计不准确**: 初期信念质量差
   - **缓解**: 贝叶斯更新，Transformer 时序建模
   - **缓解**: 初期依赖公共信息（Phase 1）

2. **内存不足**: 存储全局状态和采样状态
   - **缓解**: 及时释放，使用 float16，采样数控制
   - **缓解**: 检查点间隔增加

---

## 总工作量预估

| 任务类别 | 任务数 | 工作量（天） | 优先级 |
|---------|--------|-------------|--------|
| P0: 核心问题修复 | 1 | 3.0 | 🔴 高 |
| P1: 信念系统细节 | 2 | 3.0 | 🔴 高 |
| P2: 监控系统 | 2 | 3.5 | 🟡 中 |
| P3: 对手策略池 | 1 | 1.0 | 🟢 低 |
| **总计** | **6** | **10.5** | - |

**总预计工作量**: 10.5 天（全职开发）

---

## 下一步行动

1. ✅ 用户已确认所有决策
2. 🔄 创建补充任务 TODOs
3. 🔄 插入到现有计划文件
4. 🔄 更新依赖矩阵
5. 🔄 更新 Wave 执行顺序
6. 🔄 完成计划审查

**当前状态**: 等待补充 TODOs 创建
