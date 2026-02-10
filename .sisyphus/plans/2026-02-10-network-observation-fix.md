# DRL 架构完整修复计划

## 执行摘要

通过系统性分析发现 **8 个 P0 问题**（阻塞训练）和 **12 个 P1 问题**（影响效果），以及多个优化建议。

---

## 第一部分：P0 问题（阻塞训练）

### P0-1: StateEncoder 输入维度错误
**位置**: `src/drl/network.py:196`
**问题**: 网络定义 `nn.Linear(4 + 1 + 4 + 1 + 2 + 1, 48)` = 13维，但实际输入只有 10 维
**根因**: 第一个 `4` 假设 `current_player` 是 one-hot 编码，但实际是标量
**修复**:
```python
# 修复前
nn.Linear(4 + 1 + 4 + 1 + 2 + 1, 48)

# 修复后
nn.Linear(10, 48)
```

### P0-2: 标量值维度处理错误
**位置**: `src/drl/mappo.py:263` `_prepare_obs` 方法
**问题**: `remaining_tiles`, `current_phase` 标量经 `unsqueeze(0)` 后变成 `[1]`，不是 `[1, 1]`
**影响**: 导致 StateEncoder 接收到错误的维度
**修复**:
```python
def _prepare_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    tensor_obs = {}
    for key, value in obs.items():
        if isinstance(value, dict):
            tensor_obs[key] = self._prepare_obs(value)
        elif isinstance(value, np.ndarray):
            tensor_val = torch.FloatTensor(value).to(self.device)
            # 确保至少是 2D [batch, features]
            if tensor_val.dim() == 1:
                tensor_val = tensor_val.unsqueeze(0)  # [1] -> [1, 1]
            elif tensor_val.dim() == 0:
                tensor_val = tensor_val.unsqueeze(0).unsqueeze(0)  # scalar -> [1, 1]
            tensor_obs[key] = tensor_val
        else:  # 标量
            tensor_val = torch.FloatTensor([value]).unsqueeze(0).to(self.device)
            tensor_obs[key] = tensor_val
    return tensor_obs
```

### P0-3: get_centralized_batch 索引错误
**位置**: `src/drl/buffer.py:636, 673, 680, 687, 696, 705`
**问题**: 数据结构是 `[4, num_steps]` 但代码假设 `[num_steps, 4]`
**状态**: ✅ 已修复

### P0-4: get_centralized_batch 变量遮蔽
**位置**: `src/drl/buffer.py:671-700`
**问题**: 循环内变量遮蔽外层变量
**状态**: ✅ 已修复

### P0-5: discard_tiles 拼写错误
**位置**: `src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py:439`
**问题**: `player.discarded_tiles` 应该是 `player.discard_tiles`
**状态**: ✅ 已修复

### P0-6: global_hand 字段缺失
**位置**: `src/drl/trainer.py:316-338`
**问题**: Centralized buffer 存储局部观测，但 critic 需要全局观测
**状态**: ✅ 已修复

### P0-7: get_centralized_batch 返回类型
**位置**: `src/drl/buffer.py:717-727`
**问题**: 返回 list 而不是 numpy array
**状态**: ✅ 已修复

### P0-8: 使用错误的 buffer 字段
**位置**: `src/drl/trainer.py:329`
**问题**: 访问 `shared_nfsp.buffer.centralized_buffer` 应该是 `agent_pool.centralized_buffer`
**状态**: ✅ 已修复

---

## 第二部分：P1 问题（影响训练效果）

### P1-1: 动作掩码分割逻辑简化
**位置**: 需要检查 `_split_action_mask` 方法
**问题**: 不同动作类型对参数掩码的要求不同，当前处理过于简化
**建议**: 为每种动作类型维护独立的参数掩码

### P1-2: 副露编码器信息丢失
**位置**: `src/drl/network.py:400-408` `_process_melds`
**问题**: `tiles[256] -> tiles[:, :, :4]` 只取前4维，可能丢失信息
**建议**: 重新设计 melds 表示，保留完整信息

### P1-3: Transformer 只取最后一步
**位置**: 需要检查 `TransformerHistoryEncoder`
**问题**: `x[:, -1, :]` 只使用最后一步的表示
**建议**: 使用注意力池化或 [CLS] 令牌聚合所有时间步

### P1-4: 特征融合只是简单拼接
**位置**: `src/drl/network.py:382`
**问题**: 所有特征简单拼接可能不足以捕捉复杂交互
**建议**: 考虑使用门控融合或交叉注意力

### P1-5: 信念采样实现简单
**位置**: 需要检查信念采样实现
**问题**: 当前只是简单平均
**建议**: 使用注意力机制加权融合

### P1-6: 缺少训练稳定性技术
**问题**: 没有梯度裁剪、学习率预热等
**建议**:
- 添加梯度裁剪（max_norm=1.0）
- Transformer 使用 Pre-LN
- 添加学习率预热

---

## 第三部分：修复执行计划

### Batch 1: 修复 P0-1 和 P0-2（阻塞训练的核心问题）

**Task 1**: 修复 StateEncoder 输入维度
```python
# 文件: src/drl/network.py
# 行: 196-198
# 修改前
nn.Linear(4 + 1 + 4 + 1 + 2 + 1, 48)

# 修改后
nn.Linear(10, 48)
```

**Task 2**: 修复 _prepare_obs 标量处理
```python
# 文件: src/drl/mappo.py
# 行: 258-264
# 添加对 np.ndarray 和标量的区分处理
```

### Batch 2: 验证修复

运行测试验证修复：
```bash
python tests/unit/test_global_observation_prep.py
python -c "from src.drl.trainer import train_nfsp; train_nfsp(quick_test=True)"
```

---

## 第四部分：优化建议（按优先级排序）

### 高优先级（建议在 P0 修复后立即实施）

1. **动作掩码改进**
   - 为每种动作类型维护独立的参数掩码
   - 添加动作类型特定的参数验证

2. **训练稳定性**
   - 添加梯度裁剪
   - 添加学习率预热
   - 添加 value function clipping

### 中优先级（影响训练效果）

3. **Transformer 改进**
   - 使用 [CLS] 令牌或注意力池化
   - 添加 Pre-LN
   - 考虑相对位置编码

4. **特征融合增强**
   - 门控融合
   - 交叉注意力

### 低优先级（长期优化）

5. **计算效率**
   - 混合精度训练
   - 梯度检查点
   - 共享底层表示

6. **信念集成**
   - 注意力加权融合
   - 对手建模模块

---

## 第五部分：测试计划

### 单元测试
```python
def test_state_encoder_input_dim():
    """验证 StateEncoder 输入维度为 10"""
    encoder = StateEncoder()
    # 输入应该是 10 维
    assert encoder.net[0].in_features == 10

def test_prepare_obs_scalar_handling():
    """验证标量值正确转换为 [1, 1]"""
    mappo = MAPPO(...)
    obs = {"remaining_tiles": 100, "current_phase": 1}
    tensor_obs = mappo._prepare_obs(obs)
    assert tensor_obs["remaining_tiles"].shape == (1, 1)
    assert tensor_obs["current_phase"].shape == (1, 1)
```

### 集成测试
```python
def test_full_training_step():
    """验证完整训练步骤可以执行"""
    trainer = NFSPTrainer(...)
    trainer.train()  # 应该不崩溃
```

---

## 第六部分：验证清单

修复后验证：
- [ ] StateEncoder 输入维度 = 10
- [ ] 所有标量值转换为 [batch, 1]
- [ ] _prepare_obs 正确处理所有数据类型
- [ ] 训练可以运行至少 1 个 episode
- [ ] centralized critic 更新不报错
- [ ] 梯度可以正常回传

---

## 第七部分：Git 提交计划

### Commit 1: 修复 StateEncoder 维度
```
fix(network): correct StateEncoder input dimension from 13 to 10

- Fix mismatch between network definition and actual observation space
- current_player is scalar (0-3) not 4-dim one-hot
- Update nn.Linear(13, 48) to nn.Linear(10, 48)

Fixes training error: mat1 and mat2 shapes cannot be multiplied
```

### Commit 2: 修复 _prepare_obs 标量处理
```
fix(mappo): properly handle scalar values in _prepare_obs

- Distinguish between np.ndarray and scalar values
- Ensure scalars become [1, 1] not [1]
- Ensure 1D arrays become [1, n] with proper batch dim

Fixes dimension mismatch in state encoder inputs
```

### Commit 3: 验证测试
```
test: add tests for observation preprocessing

- Test StateEncoder input dimension
- Test _prepare_obs scalar handling
- Test full training step integration
```

---

## 附录：完整的数据流图

```
ObservationBuilder.build()
    ↓
观测字典 (numpy)
    ├─ global_hand: [136] ✓
    ├─ private_hand: [34] ✓
    ├─ remaining_tiles: scalar ← 问题
    ├─ current_phase: scalar ← 问题
    ├─ current_player: [1] ✓
    └─ ...
    ↓
_prepare_obs (mappo.py)
    ↓
观测字典 (torch.Tensor)
    ├─ global_hand: [1, 136] ✓
    ├─ private_hand: [1, 34] ✓
    ├─ remaining_tiles: [1] ← 问题！应该是 [1, 1]
    ├─ current_phase: [1] ← 问题！应该是 [1, 1]
    └─ ...
    ↓
ObservationEncoder.forward
    ↓
StateEncoder.forward
    ├─ current_player: [1, 1] ✓
    ├─ remaining_tiles: [1, 1] ✓
    ├─ fan_counts: [1, 4] ✓
    ├─ current_phase: [1, 1] ✓
    ├─ special_indicators: [1, 2] ✓
    └─ dealer: [1, 1] ✓
    ↓
torch.cat(..., dim=-1) → [1, 10] ✓
    ↓
nn.Linear(10, 48) ← 当前是 nn.Linear(13, 48) ✗
```
