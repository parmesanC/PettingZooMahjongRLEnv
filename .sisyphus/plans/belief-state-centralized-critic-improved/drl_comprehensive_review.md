# DRL 目录全面代码审查报告

**日期**: 2026-02-10
**审查范围**: `src/drl/` 目录所有文件
**审查类型**: 架构一致性、属性访问、数据流、接口匹配、类型问题

---

## 执行摘要

共发现 **10 个关键问题**，分为 5 大类：

| 类别 | 数量 | 严重程度 |
|------|------|----------|
| 属性访问错误 | 1 | P0 |
| 架构不一致 | 3 | P0-P1 |
| 数据流问题 | 3 | P0-P1 |
| 接口不匹配 | 3 | P0 |
| 类型问题 | 1 | P1 |

---

## 问题分类详解

### 1. 属性访问错误 (P0)

#### Issue #1: MixedBuffer 没有 centralized_buffer 属性

**文件**: `src/drl/trainer.py:329`

**错误代码**:
```python
self.agent_pool.shared_nfsp.buffer.centralized_buffer.add_multi_agent(
```

**问题分析**:
```
NFSP
├── buffer = MixedBuffer  # 只有 rl_buffer, sl_buffer
└── centralized_buffer = CentralizedRolloutBuffer  # 独立属性
```

`MixedBuffer` 类只有：
- `self.rl_buffer` (RolloutBuffer)
- `self.sl_buffer` (ReservoirBuffer)

**修复方案**:
```python
# 方案1: 使用 NFSP 的 centralized_buffer
self.agent_pool.shared_nfsp.centralized_buffer.add_multi_agent(...)

# 方案2: 使用 NFSPAgentPool 的 centralized_buffer
self.agent_pool.centralized_buffer.add_multi_agent(...)
```

---

### 2. 架构不一致 (P0-P1)

#### Issue #2: NFSPAgentPool 和 NFSP 各有自己的 centralized_buffer

**文件**: `src/drl/agent.py:273`, `src/drl/nfsp.py:71`

**问题**:
```python
# NFSPAgentPool 创建了自己的 buffer
self.centralized_buffer = CentralizedRolloutBuffer(capacity=rl_capacity)

# NFSP 也创建了自己的 buffer
self.centralized_buffer = CentralizedRolloutBuffer(capacity=config.nfsp.rl_buffer_size)
```

**影响**: 两个独立的 buffer 实例，数据填充到哪个不清楚

**修复方案**:
- 统一使用一个 buffer
- 明确文档说明每个 buffer 的用途
- 确保 trainer 填充的 buffer 就是 train_step 使用的 buffer

---

#### Issue #3: finish_episode() 永远不被调用

**文件**: `src/drl/trainer.py`

**问题**:
- `CentralizedRolloutBuffer` 在 episode 期间积累数据到 `current_*` 列表
- `finish_episode()` 方法负责将数据打包到 `self.episodes` 并重置 `current_*`
- 但 trainer 从未调用 `finish_episode()`
- 导致数据永远无法通过 `get_centralized_batch()` 访问

**修复方案**:
```python
# 在 _run_episode() 结束时调用
if self.current_phase in [1, 2]:
    self.agent_pool.centralized_buffer.finish_episode()
```

---

#### Issue #4: Buffer 索引模式错误

**文件**: `src/drl/buffer.py:624-633`

**错误代码**:
```python
# 假设 episode["observations"] 是 [num_steps, 4]
agent_obs_list = [
    episode["observations"][step_idx][agent_idx]
    for step_idx in range(num_steps)
]
```

**实际结构** (buffer.py:533-535):
```python
"observations": [
    list(obs_list) for obs_list in self.current_obs
],  # 这是 [4, num_steps] 结构
```

**修复方案**:
```python
# 应该是:
agent_obs_list = [
    episode["observations"][agent_idx][step_idx]
    for step_idx in range(num_steps)
]
```

---

### 3. 数据流问题 (P0-P1)

#### Issue #5: values 数据未被收集和存储

**文件**: `src/drl/buffer.py:488-521`, `src/drl/trainer.py:322`

**问题**:
1. Trainer 收集了 `all_values` (trainer.py:322)
2. 但 `add_multi_agent()` 没有这个参数
3. 即使添加参数，也没有代码存储它

**修复方案**:
```python
# 添加参数
def add_multi_agent(
    self,
    all_observations: List[Dict[str, np.ndarray]],
    action_masks: List[np.ndarray],
    actions_type: List[int],
    actions_param: List[int],
    log_probs: List[float],
    rewards: List[float],
    values: List[float] = None,  # 新增
    done: bool = False,
):
    # 存储数据
    if values is not None:
        self.current_values[agent_idx].append(values[agent_idx])
```

---

#### Issue #6: get_centralized_batch 返回结构混乱

**文件**: `src/drl/buffer.py:586-719`

**问题**:
- 文档说明返回 `[batch_size, num_steps, 4]`
- 但实际代码创建的是 `[batch_size, 4, num_steps]`
- 转置逻辑混乱

**修复方案**:
- 统一数据结构
- 修正转置逻辑
- 更新文档

---

### 4. 接口不匹配 (P0)

#### Issue #7: NFSPAgent.train_step() 缺少参数

**文件**: `src/drl/agent.py:157-167`

**错误代码**:
```python
def train_step(self) -> Dict:
    return self.nfsp.train_step()  # 缺少参数!
```

**NFSP.train_step() 签名** (nfsp.py:212):
```python
def train_step(self, training_phase: int = 1, centralized_buffer=None) -> Dict:
```

**修复方案**:
```python
def train_step(self, training_phase: int = 1, centralized_buffer=None) -> Dict:
    if not self.is_training:
        return {}
    return self.nfsp.train_step(
        training_phase=training_phase,
        centralized_buffer=centralized_buffer
    )
```

---

#### Issue #8: NFSPAgentWrapper.train_step() 缺少参数

**文件**: `src/drl/agent.py:393-395`

**错误代码**:
```python
def train_step(self):
    return self.nfsp.train_step()
```

**修复方案**: 同 Issue #7

---

### 5. 类型问题 (P1)

#### Issue #9: add_multi_agent 假设错误的参数类型

**文件**: `src/drl/buffer.py:488-521`

**问题**:
- `actions_type: List[int]` - 但应该是每个 agent 每步的值
- 当前实现假设每个 agent 只有一个值
- 实际应该是每个 agent 每个时间步都有值

---

## 优先修复顺序

### P0 (立即修复 - 阻塞 Phase 1-2 训练)

1. **trainer.py:329** - 修复 centralized_buffer 访问路径
2. **agent.py:157-167** - 为 NFSPAgent.train_step() 添加参数
3. **agent.py:393-395** - 为 NFSPAgentWrapper.train_step() 添加参数
4. **buffer.py** - 添加 `values` 参数到 `add_multi_agent`
5. **trainer.py** - 调用 `finish_episode()` 结束 episode

### P1 (尽快修复 - 导致运行时错误)

6. **buffer.py:624-633** - 修复 buffer 索引模式
7. **agent.py:273 + nfsp.py:71** - 统一 centralized_buffer 使用
8. **buffer.py:586-719** - 修复 get_centralized_batch 返回结构

### P2 (可以延后 - 代码质量)

9. 添加类型提示
10. 添加数据流文档
11. 添加 buffer 访问验证

---

## 数据流图 (当前状态 vs 修复后)

### 当前 (损坏的数据流)

```
Episode 收集数据
    ↓
_populate_centralized_buffer_from_steps()
    ↓
self.agent_pool.shared_nfsp.buffer.centralized_buffer  ❌ 不存在!
    ↓
训练失败
```

### 修复后

```
Episode 收集数据
    ↓
_populate_centralized_buffer_from_steps()
    ↓
self.agent_pool.centralized_buffer.add_multi_agent()  ✅
    ↓
finish_episode() 打包数据
    ↓
train_all() 传递 centralized_buffer
    ↓
train_step() 使用数据训练
    ✅ 正常工作
```

---

## 文件修改清单

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `trainer.py:329` | 修复 centralized_buffer 访问 | P0 |
| `trainer.py` | 添加 finish_episode() 调用 | P0 |
| `agent.py:157-167` | 添加 train_step 参数 | P0 |
| `agent.py:393-395` | 添加 train_step 参数 | P0 |
| `buffer.py:488-521` | 添加 values 参数 | P0 |
| `buffer.py:624-633` | 修复索引模式 | P1 |
| `buffer.py:586-719` | 修复返回结构 | P1 |

---

**生成时间**: 2026-02-10
**Agent**: Explore (a71c9c6)
**审查范围**: src/drl/ 目录所有 Python 文件
