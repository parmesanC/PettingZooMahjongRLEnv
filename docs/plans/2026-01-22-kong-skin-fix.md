# KONG_SKIN 动作掩码一致性修复

## 1. 问题分析

### 1.1 问题现象
在 MELD_DECISION 状态下，当玩家选择 KONG_SKIN（皮子杠）动作时，出现错误：
```
IndexError: list.remove(x): x not in list
```

### 1.2 调试发现
通过运行 `test_kong_lazy_debug.py`，发现以下异常情况：

```
[DEBUG ActionValidator]
玩家手牌: [2, 4, 5, 12, 15, 15, 16, 21, 27, 30, 32]
context.skin_tile: [6, 5]
检测到的可用动作: [('KONG_SKIN', 5), ...]  # 正确：手牌中有牌5

[DEBUG GongState]
玩家动作: (7, 6)  # 错误：参数变成了6，但手牌中没有牌6
牌 6 不在手牌中！将会报错！
```

### 1.3 根本原因
**动作检测与掩码构建不一致：**

```
┌─────────────────────────────────────────────────────────────┐
│ ActionValidator.detect_available_actions_after_draw()       │
│ → 检测手牌中存在的皮子                                      │
│ → 返回 ('KONG_SKIN', 5)  ✅ 手牌中有牌5                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ ObservationBuilder._build_decision_mask()                   │
│ → 遍历 context.skin_tile = [6, 5]                          │
│ → mask[208 + 0*34 + 6] = 1  (索引214) ❌ 手牌中无牌6        │
│ → mask[208 + 1*34 + 5] = 1  (索引247) ✅ 手牌中有牌5        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ RandomStrategy.choose_action()                              │
│ → 可能选择索引214（对应牌6）                                 │
│ → 返回 (7, 6)  ❌ 错误的参数                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ GongState._handle_special_kong()                            │
│ → 尝试从手牌中移除牌6                                       │
│ → list.remove(6)  ❌ 牌6不在手牌中                          │
│ → 报错！                                                    │
└─────────────────────────────────────────────────────────────┘
```

## 2. 设计方案

### 2.1 观测空间优化
**问题：** KONG_SKIN 使用 68 维（208-275），假设两个皮子可能不同

**分析：** 武汉麻将规则中，两个皮子不会相同

**解决方案：** 将 KONG_SKIN 降至 34 维（208-241）

### 2.2 统一检测与掩码逻辑
**原则：** ObservationBuilder 只标记 ActionValidator 实际检测到的动作

**修改：** 使用 `action.parameter` 而非遍历 `context.skin_tile`

## 3. 完整修改清单

| 文件 | 修改内容 | 行号 |
|------|----------|------|
| `wuhan_7p4l_observation_builder.py` | ACTION_MASK_RANGES 常量 | 12-24 |
| `wuhan_7p4l_observation_builder.py` | 掩码数组初始化 278→244 | 83 |
| `wuhan_7p4l_observation_builder.py` | _build_decision_mask KONG_SKIN 逻辑 | 159-163 |
| `wuhan_7p4l_observation_builder.py` | _build_decision_mask WIN 索引 276→242 | 165 |
| `wuhan_7p4l_observation_builder.py` | _build_meld_decision_mask KONG_SKIN 逻辑 | 208-212 |
| `wuhan_7p4l_observation_builder.py` | _build_response_mask WIN 索引 276→242 | 251 |
| `wuhan_7p4l_observation_builder.py` | _build_response_mask PASS 索引 277→243 | 254 |
| `example_mahjong_env.py` | 观测空间定义 278→244 | 98 |
| `random_strategy.py` | RANGES 常量 | 35-47 |
| `random_strategy.py` | KONG_SKIN 处理逻辑 | 82-90 |
| `random_strategy.py` | 注释 278→244 | 28-29 |
| `test_kong_lazy_debug.py` | 打印范围 276→242 | 69 |

## 4. 修改后索引范围

| 动作类型 | 索引范围 | 维度 |
|----------|----------|------|
| DISCARD | 0-33 | 34 |
| CHOW | 34-36 | 3 |
| PONG | 37 | 1 |
| KONG_EXPOSED | 38-71 | 34 |
| KONG_SUPPLEMENT | 72-105 | 34 |
| KONG_CONCEALED | 106-139 | 34 |
| KONG_RED | 140-173 | 34 |
| KONG_LAZY | 174-207 | 34 |
| KONG_SKIN | 208-241 | 34 |
| WIN | 242 | 1 |
| PASS | 243 | 1 |
| **总计** | **0-243** | **244** |

## 5. 关键代码修改

### 5.1 ACTION_MASK_RANGES 常量

```python
# 修改前
ACTION_MASK_RANGES = {
    'KONG_SKIN': (208, 276),     # 68维
    'WIN': (276, 277),
    'PASS': (277, 278),
}

# 修改后
ACTION_MASK_RANGES = {
    'KONG_SKIN': (208, 242),     # 34维
    'WIN': (242, 243),
    'PASS': (243, 244),
}
```

### 5.2 KONG_SKIN 掩码构建逻辑

```python
# 修改前
elif action_type == ActionType.KONG_SKIN.value:
    for i, skin_tile in enumerate(context.skin_tile):
        if skin_tile != -1:
            mask[208 + i * 34 + skin_tile] = 1

# 修改后
elif action_type == ActionType.KONG_SKIN.value:
    # 只标记 ActionValidator 检测到的实际可用的皮子杠动作
    mask[208 + action.parameter] = 1
```

### 5.3 RandomStrategy KONG_SKIN 处理

```python
# 修改前
elif action_type == 'KONG_SKIN':
    for i in range(2):
        skin_segment = segment[i * 34:(i + 1) * 34]
        valid_tiles = np.where(skin_segment > 0)[0]
        if len(valid_tiles) > 0:
            param = int(np.random.choice(valid_tiles))
            available_actions.append((action_type_value, param))
            break

# 修改后
elif action_type == 'KONG_SKIN':
    valid_tiles = np.where(segment > 0)[0]
    if len(valid_tiles) > 0:
        param = int(np.random.choice(valid_tiles))
        available_actions.append((action_type_value, param))
```

## 6. 验证测试

运行以下测试验证修复：
```bash
python test_kong_lazy_debug.py
python test_four_ai.py
```

预期结果：
- action_mask 形状为 (244,)
- KONG_SKIN 动作参数与手牌中实际存在的皮子牌一致
- 不再出现 `list.remove(x): x not in list` 错误
