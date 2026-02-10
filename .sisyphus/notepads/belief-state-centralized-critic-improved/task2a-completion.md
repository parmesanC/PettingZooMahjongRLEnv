---

## 2026-02-09 Task 2a: BeliefNetwork辅助损失训练 ✅

**文件**: `src/drl/belief_network.py`

**添加的方法**:
- `compute_auxiliary_loss(beliefs, next_actions, next_melds, next_tile_counts, use_auxiliary=True)`: 计算辅助损失
- `_predict_action_from_belief(beliefs)`: 从信念预测对手动作类型（4分类）
- `_predict_tile_count_from_belief(beliefs)`: 从信念预测手牌数量（回归）

**3个辅助预测任务**:
1. **动作预测损失**（34分类）: 预测对手下一轮动作类型（出牌/吃/碰/杠）
   - 基于信念熵进行映射：低熵→出牌，高熵→杠/胡
2. **副露预测损失**（4分类）: 预测对手是否吃/碰/杠
   - 与动作预测使用相同逻辑
3. **手牌数量预测损失**（回归）: 预测对手手牌总数
   - 基于信念总概率 × 13（标准手牌数）

**总 loss 函数**:
```
total_loss = 0.7 × rl_loss + 0.3 × auxiliary_loss
auxiliary_loss = 0.4 × action_prediction_loss + 0.3 × meld_prediction_loss + 0.3 × tile_count_loss
```

**配置开关**:
- `use_auxiliary_loss=True/False`: 控制是否启用辅助损失

**验证结果**:
- 辅助损失方法可调用 ✅
- use_auxiliary_loss 开关可控制 ✅
- 禁用时返回 0.0 ✅
- 启用时正确计算3项损失 ✅

---

