# Belief State & Centralized Critic - Wave 1 Learnings

## 2026-02-09 Wave 1 执行总结

### Task 1: BeliefNetwork实现 ✅

**文件**: `src/drl/belief_network.py`

**关键设计决策**:
- 使用 Transformer 建模出牌历史
- 支持可配置对手数量 (默认3个)
- 使用 softmax 归一化每个对手的概率分布
- 实现贝叶斯更新方法用于动态调整

**复用的现有编码器**:
- DiscardEncoder: 编码弃牌池 [34] -> [32]
- MeldEncoder: 编码副露 [16, 9] -> [64]
- TransformerHistoryEncoder: 编码出牌历史

**遇到的挑战**:
- 需要处理 melds 字典格式转换为张量
- 贝叶斯更新需要处理不同维度的输入

**验证结果**:
- 导入成功: `from src.drl.belief_network import BeliefNetwork`
- 实例化成功: `BeliefNetwork(hidden_dim=256)`

---

### Task 2b: 全局状态构建器 ✅

**文件**: `src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py`

**添加的方法**:
- `build_global_observation(context, training_phase=1)`: 构建全局状态
- `_build_hand_onehot(hand_tiles)`: 手牌 one-hot 编码 [14, 34]
- `_build_wall_counts(wall)`: 牌墙 count 编码 [34, 34]
- `_count_tiles(tiles)`: 统计牌数量 [34]
- `_build_melds_onehot(melds)`: 副露 one-hot 编码 [16, 34]
- `_apply_progressive_masking(global_obs, context)`: Phase 2 渐进遮蔽
- `_apply_belief_sampling(global_obs, context)`: Phase 3 信念采样（暂返回零）

**Phase 差异**:
- Phase 1 (全知): 返回所有真实信息
- Phase 2 (渐进): 随机遮蔽 30%-70% 的手牌
- Phase 3 (采样): 返回零掩码（待Task 1完成集成）

**数据格式**:
```
{
  'player_0_hand': [14, 34],  # 4玩家手牌
  'player_1_hand': [14, 34],
  'player_2_hand': [14, 34],
  'player_3_hand': [14, 34],
  'wall_tiles': [34, 34],     # 牌墙
  'discard_piles': [4, 34],    # 弃牌堆
  'melds': [4, 16, 34],       # 副露
  'current_player': int,
  'remaining_wall_count': int,
  'game_progress': float
}
```

---

### Task 3: 单元测试框架 ✅

**创建的文件**:
- `tests/unit/__init__.py`
- `tests/unit/conftest.py`: pytest fixtures
- `tests/unit/test_belief_network.py`: BeliefNetwork 测试
- `tests/unit/test_global_observation.py`: 全局观测构建测试
- `tests/integration/__init__.py`

**提供的 fixtures**:
- `sample_observation`: 测试用的观测样本
- `batch_size`: 批次大小 (默认4)
- `hidden_dim`: 隐藏层维度 (默认256)

---

### Task 3b: 代码质量基础设施 ✅

**创建的配置文件**:
- `.config/black.toml`: Black 代码格式化配置
- `.config/ruff.toml`: Ruff 快速 linting 配置
- `.pre-commit-config.yaml`: Pre-commit hooks 配置
- `.github/pull_request_template.md`: PR 模板

**配置内容**:
- Black: line-length=100, target-version=py38
- Ruff: 兼容 pytest, 基本忽略配置
- Pre-commit: black + ruff 自动修复

---

## 技术债务和待解决问题

### LSP 错误（预存在，不影响代码运行）
以下错误是项目中预存在的，不影响 Wave 1 实现的代码：
- `src/drl/network.py`: torch 导入无法解析（配置问题）
- `src/mahjong_rl/core/GameData.py`: 类型注解问题

### Phase 3 信念采样待实现
当前 Phase 3 返回零掩码，需要在 Task 1 完成后集成 BeliefNetwork 进行采样。

---

## 下一步：Task 2a

**任务**: BeliefNetwork 辅助损失训练

**需要添加的内容**:
1. 3个辅助预测头：
   - 预测对手下一轮打出的牌（34分类）
   - 预测对手是否吃/碰/杠（4分类）
   - 预测对手手牌总数（回归）
2. 总 loss 函数：
   ```
   total_loss = 0.7 × rl_loss + 0.3 × auxiliary_loss
   auxiliary_loss = 0.4 × action_prediction_loss + 0.3 × meld_prediction_loss + 0.3 × tile_count_loss
   ```
3. 开关 `use_auxiliary_loss=True/False`

**依赖**: Task 1 已完成 ✅

---



