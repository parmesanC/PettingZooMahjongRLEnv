# Action Mask 重构设计文档（211位 → 145位）

## 概述

### 文档定位
本文档记录了武汉麻将 RL 环境 action mask 的重大重构。此次重构将 action mask 从 211 位压缩至 145 位，并实施了"排他性设计"（Plan C）：特殊牌（赖子、皮子、红中）只能通过杠动作处理，不能通过 DISCARD 打出。

**文档受众：混合受众（开发者 + RL 研究者）**

### 核心变更
- 索引 107：KONG_RED（1位，全场只有一张红中）
- 索引 108：KONG_LAZY（1位，全场只有一张赖子）
- 索引 109-142：KONG_SKIN（34位，两张皮子独立）
- 索引 0-33：DISCARD（**不再包含特殊牌**）
- 索引 143：WIN（原 209）
- 索引 144：PASS（原 210）

### 设计动机
原设计中，特殊牌既可以通过 DISCARD 打出，也可以通过杠动作处理。这导致了状态转移的不确定性（同一状态下同一动作可能转移至不同状态），违反了 MDP 的确定性要求，影响 RL 训练的稳定性。排他性设计确保了"状态 + 动作 → 下一状态"的唯一确定性。

---

## 使用指南

### Action Mask 快速参考表

| 动作类型 | 索引范围 | 参数 | 说明 |
|---------|---------|------|------|
| DISCARD | 0-33 | 牌ID (0-33) | 打出普通牌，**不包含特殊牌** |
| CHOW | 34-36 | 0=左吃, 1=中吃, 2=右吃 | 吃牌 |
| PONG | 37 | 0（忽略） | 碰牌 |
| KONG_EXPOSED | 38 | 0（忽略） | 明杠 |
| KONG_SUPPLEMENT | 39-72 | 牌ID | 补杠 |
| KONG_CONCEALED | 73-106 | 牌ID | 暗杠 |
| **KONG_RED** | **107** | **0** | **红中杠（1位）** |
| **KONG_LAZY** | **108** | **0** | **赖子杠（1位）** |
| **KONG_SKIN** | **109-142** | **牌ID** | **皮子杠（34位）** |
| WIN | 143 | -1 | 胡牌 |
| PASS | 144 | -1 | 过牌 |

### 关键设计说明

#### 1. 特殊牌的排他性处理
特殊牌（赖子、皮子、红中）不能通过 DISCARD 动作打出，必须通过对应的杠动作处理：
- 手中有红中 → 只能执行 KONG_RED（索引107，参数0）
- 手中有赖子 → 只能执行 KONG_LAZY（索引108，参数0）
- 手中有皮子 → 只能执行 KONG_SKIN（索引109-142，参数为皮子牌ID）

#### 2. 为什么 KONG_RED 和 KONG_LAZY 只有 1 位
- 全场只有一张红中和一张赖子
- 无需参数指定具体是哪张牌
- 从 context.red_dragon 和 context.lazy_tile 获取

#### 3. 为什么 KONG_SKIN 需要 34 位
- 有两张独立的皮子牌
- 玩家可能拥有其中一张或两张
- 需要通过参数指定具体杠哪张皮子

### 代码示例

#### 示例 1：从 action_mask 读取可用动作

```python
import numpy as np

def get_available_actions(action_mask):
    """解析 145 位 action_mask，返回可用动作列表"""
    actions = []

    # DISCARD (0-33)
    for tile_id in range(34):
        if action_mask[tile_id]:
            actions.append(('DISCARD', tile_id))

    # CHOW (34-36)
    for i, chow_type in enumerate(['左吃', '中吃', '右吃']):
        if action_mask[34 + i]:
            actions.append(('CHOW', i))

    # PONG (37)
    if action_mask[37]:
        actions.append(('PONG', 0))

    # KONG_EXPOSED (38)
    if action_mask[38]:
        actions.append(('KONG_EXPOSED', 0))

    # 特殊杠（重点）
    if action_mask[107]:  # KONG_RED
        actions.append(('KONG_RED', 0))
    if action_mask[108]:  # KONG_LAZY
        actions.append(('KONG_LAZY', 0))

    # KONG_SKIN (109-142)
    for tile_id in range(34):
        if action_mask[109 + tile_id]:
            actions.append(('KONG_SKIN', tile_id))

    # WIN (143) / PASS (144)
    if action_mask[143]:
        actions.append(('WIN', -1))
    if action_mask[144]:
        actions.append(('PASS', -1))

    return actions
```

#### 示例 2：RL 环境中的动作选择

```python
import gymnasium as gym
from example_mahjong_env import WuhanMahjongEnv

env = WuhanMahjongEnv(render_mode=None, training_phase=3)
obs, info = env.reset(seed=42)

# 获取 action_mask
action_mask = obs['action_mask']

# 检查是否有特殊杠可用
has_red_kong = action_mask[107] == 1
has_lazy_kong = action_mask[108] == 1
has_skin_kong = np.any(action_mask[109:143]) == 1

if has_red_kong:
    print("可以执行红中杠，动作: (6, 0)")
if has_lazy_kong:
    print("可以执行赖子杠，动作: (7, 0)")
if has_skin_kong:
    skin_tiles = [i for i in range(34) if action_mask[109 + i]]
    print(f"可以执行皮子杠，牌ID: {skin_tiles}")

# 执行动作
next_obs, reward, terminated, truncated, info = env.step((6, 0))  # 红中杠
```

### 常见问题解答（FAQ）

#### Q1: 为什么我的手牌中有红中，但 action_mask[0-33] 中没有对应的 DISCARD 位？
**A:** 这是设计的核心特性。特殊牌（红中、赖子、皮子）不能通过 DISCARD 打出，只能通过杠动作处理。如果你手中有红中（假设牌ID为31），你会发现：
- `action_mask[31]` (DISCARD位) = 0
- `action_mask[107]` (KONG_RED位) = 1

执行 `(6, 0)` 即可触发红中杠。

#### Q2: KONG_SKIN 的参数应该是多少？
**A:** KONG_SKIN 需要指定具体杠哪张皮子牌。假设皮子牌的ID分别是 4 和 5：
- 如果只想杠牌ID为4的皮子：参数为 4，动作 `(8, 4)`
- 如果只想杠牌ID为5的皮子：参数为 5，动作 `(8, 5)`
- `action_mask[109+4]` 和 `action_mask[109+5]` 会分别表示是否可用

#### Q3: 迁移旧代码时需要注意什么？
**A:** 主要变更点：
1. WIN 和 PASS 的索引从 209、210 变更为 143、144
2. KONG_RED 和 KONG_LAZY 从 34 位压缩为 1 位，参数固定为 0
3. KONG_SKIN 的起始索引从 175 变更为 109
4. DISCARD 不再包含特殊牌

#### Q4: 如何验证 action_mask 是否正确？
**A:** 运行测试脚本：
```bash
python test_obs_builder_validator.py
python test_kong_lazy_debug.py
```

---

## 技术实现

### 问题 1：状态转移的不确定性

#### 问题描述
在旧设计中，当玩家手中有特殊牌（如红中）时，可以执行两种动作：
- `(0, 31)` - DISCARD 打出红中
- `(6, 31)` - KONG_RED 红中杠

这两种动作虽然类型不同，但在武汉麻将规则下，"打出一张红中就是红中杠"。然而，这种设计导致了问题：
- 如果选择 `(0, 31)`，状态机应该转移到哪里？是当作普通出牌处理，还是当作杠牌处理？
- 同一个状态 `S`，同一个动作 `(0, 31)`，可能转移至不同状态，违反了 MDP 的确定性要求

#### 解决方案：排他性设计（Plan C）
- 将特殊牌从 DISCARD 的可用范围中完全移除
- 特殊牌只能通过对应的杠动作处理
- 确保状态转移的确定性：特殊牌只有一种合法操作方式

#### 代码变更

在 `wuhan_7p4l_observation_builder.py` 的 `_build_decision_mask` 方法中：

```python
# 修复前
if action_type == ActionType.DISCARD.value:
    for tile_id in range(34):
        if hand_counts[tile_id] > 0:
            mask[tile_id] = 1  # 所有牌都可以打出

# 修复后
if action_type == ActionType.DISCARD.value:
    special_tiles = [context.lazy_tile, context.red_dragon] + context.skin_tile
    for tile_id in range(34):
        if tile_id not in special_tiles and hand_counts[tile_id] > 0:
            mask[tile_id] = 1  # 排除特殊牌
```

### 问题 2：后备逻辑中的特殊牌遗漏

#### 问题描述
在 `_build_meld_decision_mask` 方法中，有一段后备逻辑用于确保 DISCARD 始终可用。代码注释说明"需要排除特殊牌"，但实际实现中遗漏了 `and tile not in special_tiles` 条件，导致特殊牌仍可能被错误地加入 DISCARD 掩码。

#### 解决方案
添加特殊牌排除条件，与主逻辑保持一致。

#### 代码变更

在 `wuhan_7p4l_observation_builder.py` 第 227 行：

```python
# 修复前
for tile in player.hand_tiles:
    if 0 <= tile < 34:
        mask[tile] = 1

# 修复后
special_tiles = [context.lazy_tile, context.red_dragon] + context.skin_tile
for tile in player.hand_tiles:
    if 0 <= tile < 34 and tile not in special_tiles:
        mask[tile] = 1
```

### 问题 3：GongState 特殊杠处理错误

#### 问题描述
在 `gong_state.py` 的 `_handle_special_kong` 方法中，直接使用传入的 `tile` 参数从手牌中移除牌。然而，由于我们将 KONG_RED 和 KONG_LAZY 改为 1 位设计，它们的 parameter 固定为 0，导致：
- `player.hand_tiles.remove(0)` 尝试移除牌ID为 0 的牌
- 实际应该移除的是 `context.red_dragon` 或 `context.lazy_tile`

#### 解决方案
根据 `kong_type` 从 `context` 中获取正确的牌ID。

#### 代码变更

在 `gong_state.py` 的 `_handle_special_kong` 方法中：

```python
# 修复前
def _handle_special_kong(self, context, player, tile, kong_type):
    player.hand_tiles.remove(tile)  # ❌ tile=0

# 修复后
def _handle_special_kong(self, context, player, tile, kong_type):
    # 确定要移除的牌ID
    if kong_type == 'RED':
        actual_tile = context.red_dragon
    elif kong_type == 'LAZY':
        actual_tile = context.lazy_tile
    elif kong_type == 'SKIN':
        actual_tile = tile  # 皮子杠使用传入的参数

    player.hand_tiles.remove(actual_tile)
```

### 问题 4：ActionValidator 返回错误的 parameter

#### 问题描述
在 `action_validator.py` 中，检测到 KONG_RED 和 KONG_LAZY 时，将 `parameter` 设置为实际的牌ID。这与新的 1 位设计不匹配（parameter 应固定为 0）。

#### 解决方案
将 KONG_RED 和 KONG_LAZY 的 parameter 固定为 0。

#### 代码变更

在 `action_validator.py` 第 178-180 行：

```python
# 修复前
if card == self.context.lazy_tile:
    available_actions.append(MahjongAction(ActionType.KONG_LAZY, card))
elif card == self.context.red_dragon:
    available_actions.append(MahjongAction(ActionType.KONG_RED, card))

# 修复后
if card == self.context.lazy_tile:
    available_actions.append(MahjongAction(ActionType.KONG_LAZY, 0))
elif card == self.context.red_dragon:
    available_actions.append(MahjongAction(ActionType.KONG_RED, 0))
```

### 问题 5：Action Mask 索引范围全面更新

#### 问题描述
将 action mask 从 211 位压缩到 145 位，需要同步更新所有相关文件中的索引范围定义和访问逻辑。

#### 解决方案
系统性地更新以下组件：
1. ACTION_MASK_RANGES 常量定义
2. observation_space 维度
3. RandomStrategy 的索引解析
4. CLI/Web 渲染器的按钮索引
5. 测试脚本的断言

#### 代码变更

**wuhan_7p4l_observation_builder.py - ACTION_MASK_RANGES:**
```python
ACTION_MASK_RANGES = {
    'DISCARD': (0, 34),
    'CHOW': (34, 37),
    'PONG': (37, 38),
    'KONG_EXPOSED': (38, 39),
    'KONG_SUPPLEMENT': (39, 73),
    'KONG_CONCEALED': (73, 107),
    'KONG_RED': (107, 108),      # 1位
    'KONG_LAZY': (108, 109),     # 1位
    'KONG_SKIN': (109, 143),     # 34位
    'WIN': (143, 144),
    'PASS': (144, 145),
}
```

**example_mahjong_env.py - observation_space:**
```python
'action_mask': spaces.MultiBinary(145)  # 原 211
```

**random_strategy.py - RANGES:**
```python
RANGES = {
    'DISCARD': (0, 34),
    # ... 其他范围同步更新
    'KONG_RED': (107, 108),
    'KONG_LAZY': (108, 109),
    'KONG_SKIN': (109, 143),
    'WIN': (143, 144),
    'PASS': (144, 145),
}
```

### 修改文件清单

| 文件 | 修改类型 | 关键变更 |
|------|----------|----------|
| `wuhan_7p4l_observation_builder.py` | 核心逻辑 | ACTION_MASK_RANGES、DISCARD排除特殊牌、后备逻辑修复 |
| `gong_state.py` | Bug修复 | `_handle_special_kong` 根据 kong_type 获取牌ID |
| `action_validator.py` | Bug修复 | KONG_RED/LAZY parameter 改为 0 |
| `example_mahjong_env.py` | 配置 | observation_space 211→145 |
| `random_strategy.py` | 索引同步 | RANGES 更新、KONG_SKIN 参数处理 |
| `cli_controller.py` | 索引同步 | action_ranges 更新、参数验证 |
| `cli_renderer.py` | 索引同步 | action_ranges 更新 |
| `web_renderer.py` | 索引同步 | 动作按钮索引更新 |
| `test_kong_lazy_debug.py` | 测试 | 索引断言更新 |
| `test_obs_builder_validator.py` | 测试 | 索引断言更新 |

### 验证清单

- [ ] 运行 `test_obs_builder_validator.py` 验证 action_mask 结构
- [ ] 运行 `test_kong_lazy_debug.py` 验证特殊杠流程
- [ ] 运行 `test_four_ai.py` 验证完整游戏流程
- [ ] 检查 DISCARD 掩码不包含特殊牌
- [ ] 检查 KONG_RED/LAZY parameter 为 0
- [ ] 检查 KONG_SKIN parameter 为正确的牌ID

---

## 附录：新旧索引对比

| 动作 | 旧索引 | 新索引 | 变更说明 |
|------|--------|--------|----------|
| WIN | 209 | 143 | -66 |
| PASS | 210 | 144 | -66 |
| KONG_RED | 107-140 | 107 | 34位→1位 |
| KONG_LAZY | 141-174 | 108 | 34位→1位，起始位置前移 |
| KONG_SKIN | 175-208 | 109-142 | 34位保持，起始位置前移 |

**总位数：211 → 145（减少 66 位）**

---

## 总结

本次重构通过以下改进，提升了 RL 环境的训练稳定性和代码可维护性：

1. **MDP 确定性**：特殊牌的排他性设计确保状态转移的唯一性
2. **空间优化**：action mask 从 211 位压缩至 145 位，减少 31%
3. **代码一致性**：统一了所有组件对 action_mask 的处理逻辑
4. **设计清晰性**：每种特殊牌的处理方式更加明确

这些改进为后续的 RL 训练提供了更稳定的环境基础。
