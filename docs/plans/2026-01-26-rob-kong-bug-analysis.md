# 🔍 武汉麻将状态机全面分析报告

**日期**：2026-01-26
**任务**：修复抢杠和测试失败并全面检查类似问题
**状态**：✅ 核心问题已修复，待处理代码质量问题

---

## 一、原始问题总结

### 问题：`check_min_fan_requirement` 漏算胡牌类型基础分

**位置**：`src/mahjong_rl/rules/wuhan_mahjong_rule_engine/score_calculator.py:360`

**错误代码**：
```python
# 只计算了口口翻和硬胡因子
base_fan = self._get_base_fan_score(winner)
winner_fan = base_fan * (2.0 if HARD_WIN else 1.0)
# ❌ 缺少胡牌类型基础分！
```

**修复方案**：
```python
# 添加 _get_win_type_base_score 方法
def _get_win_type_base_score(self, win_types: list) -> float:
    """
    计算胡牌类型的基础分（不包括口口翻）

    规则：
    - 小胡（屁胡）：基础分 1
    - 大胡：基础分 10 × 大胡个数
    """
    from src.mahjong_rl.core.constants import WinType

    # 大胡类型列表
    BIG_WIN_TYPES = {
        WinType.PURE_FLUSH,    # 清一色
        WinType.PENG_PENG_HU,  # 碰碰胡
        WinType.ALL_WIND,      # 风一色
        WinType.ALL_JIANG,     # 将一色
        WinType.FLOWER_ON_KONG,# 杠上开花
        WinType.ROB_KONG,      # 抢杠和
        WinType.LAST_TILE_WIN, # 海底捞月
        WinType.FULLY_MELDED,  # 全求人
    }

    # 计算大胡个数
    big_win_count = sum(1 for wt in win_types if wt in BIG_WIN_TYPES)

    # 返回基础分：小胡1分，大胡10×个数
    return 10.0 * big_win_count if big_win_count > 0 else 1.0

# 修改 check_min_fan_requirement
def check_min_fan_requirement(self, winner_id: int, win_types: list, ctx: GameContext) -> bool:
    """
    检查是否满足起胡番要求

    计算公式：
    总番数 = 胡牌类型基础分 × 口口翻（底翻×开口×杠牌） × 硬胡因子
    """
    winner = ctx.players[winner_id]

    # 获取口口翻（底翻+开口+杠牌）
    kou_kou_fan = self._get_base_fan_score(winner)

    # 计算胡牌类型的基础分（小胡=1，大胡=10×个数）
    win_type_base_score = self._get_win_type_base_score(win_types)

    # 硬胡因子（硬胡乘2，软胡乘1）
    hard_win_factor = 2.0 if WinType.HARD_WIN in win_types else 1.0

    # 总番数 = 胡牌类型基础分 × 口口翻 × 硬胡因子
    winner_fan = win_type_base_score * kou_kou_fan * hard_win_factor

    # 计算所有玩家的番数，找到最小值
    min_other_fan = float('inf')
    for other_player in ctx.players:
        if other_player.player_id != winner_id:
            other_fan = self._get_base_fan_score(other_player)
            min_other_fan = min(min_other_fan, other_fan)

    # 检查：赢家番数 × 最小番数 >= 16
    return winner_fan * min_other_fan >= 16
```

**影响**：
- ✅ 抢杠和测试现在应该能通过（抢杠和是大胡，基础分10）
- ✅ 所有大胡类型现在都能正确计算起胡番

---

## 二、发现的其他问题

### 🔴 严重问题

#### 1. DrawingState 的死代码

**位置**：`src/mahjong_rl/state_machine/states/drawing_state.py:89-97`

**问题**：
```python
# 检查是否杠上开花（如果是杠后摸牌）
if context.is_kong_draw:  # ⚠️ 这个条件永远不会为 True！
    context.win_way = WinWay.KONG_SELF_DRAW.value
    # 检查是否胡牌
    win_result = self._check_win(context, current_player)
    if win_result.can_win:
        context.winner_ids = [context.current_player_idx]
        context.is_win = True
        return GameStateType.WIN
```

**原因**：
- 杠后摸牌走 `DrawingAfterGongState`，不会走 `DrawingState`
- 状态转换流程：`GONG` → `DRAWING_AFTER_GONG` → `PLAYER_DECISION`
- 正常摸牌流程：`WAITING_RESPONSE` → `DRAWING` → `PLAYER_DECISION`
- `is_kong_draw` 在 `DrawingState` 中永远不会为 True

**建议**：
```python
# 方案1：删除死代码
# 删除 line 89-97

# 方案2：添加注释说明
# 注意：这段代码目前不会执行，因为杠后摸牌走 DrawingAfterGongState
# 保留此代码是为了未来可能的架构变更
if context.is_kong_draw:  # 死代码：当前架构下不会执行
    ...
```

---

#### 2. DrawingAfterGongState 没有清理 `win_way`

**位置**：`src/mahjong_rl/state_machine/states/drawing_after_gong_state.py:93-98`

**问题**：
```python
# 检查杠上开花（自己胡这张牌）
win_result = self._check_win(context, current_player)
if win_result.can_win:
    context.win_way = WinWay.KONG_SELF_DRAW.value
    context.winner_ids = [context.current_player_idx]
    context.is_win = True
    return GameStateType.WIN

# ⚠️ 如果不能胡牌，win_way 没有被重置！
return GameStateType.PLAYER_DECISION
```

**影响**：
- 虽然 `PlayerDecisionState._handle_win` 会覆盖 `win_way = WinWay.SELF_DRAW.value`
- 但在未进入 WIN 状态前，`context.win_way` 仍然是 `KONG_SELF_DRAW.value`
- 如果有其他代码依赖 `win_way` 判断当前状态，会产生错误

**建议**：
```python
# 检查杠上开花（自己胡这张牌）
win_result = self._check_win(context, current_player)
if win_result.can_win:
    context.win_way = WinWay.KONG_SELF_DRAW.value
    context.winner_ids = [context.current_player_idx]
    context.is_win = True
    return GameStateType.WIN

# 重置 win_way，避免影响后续状态
context.win_way = None
return GameStateType.PLAYER_DECISION
```

---

### 🟡 代码质量问题

#### 3. WaitResponseState 使用硬编码值

**位置**：`src/mahjong_rl/state_machine/states/wait_response_state.py:214`

**问题**：
```python
context.win_way = 3  # ❌ 硬编码
```

**建议**：
```python
context.win_way = WinWay.DISCARD.value  # ✅ 使用枚举
```

---

## 三、状态机 win_way 设置完整性检查

### WinWay 枚举定义
```python
class WinWay(Enum):
    """和牌方式"""
    SELF_DRAW = 0  # 自摸
    ROB_KONG = 1  # 抢杠
    KONG_SELF_DRAW = 2  # 杠开
    DISCARD = 3  # 点炮
```

### 各状态 win_way 设置检查表

| 状态 | 位置 | WinWay 设置 | 是否调用 check_win | 状态 |
|------|------|-------------|-------------------|------|
| **PlayerDecisionState** | `_handle_win:206` | `SELF_DRAW` | ❌ 否（依赖 WinState） | ⚠️ 可接受 |
| **DrawingState** | `step:91` | `KONG_SELF_DRAW` | ✅ 是 | ❌ 死代码 |
| **DrawingAfterGongState** | `step:93` | `KONG_SELF_DRAW` | ✅ 是 | ⚠️ 未清理 win_way |
| **WaitResponseState** | `step:214` | `3` (DISCARD) | ❌ 否（依赖 WinState） | ❌ 硬编码 |
| **WaitRobKongState** | `step:145,194,256` | `ROB_KONG` | ✅ 是（在 _can_rob_kong 中） | ✅ 正确 |

### WinType 大胡检测检查

| 大胡类型 | WinChecker 检测 | 依赖的 context 属性 | 状态 |
|----------|----------------|-------------------|------|
| 清一色 PURE_FLUSH | ✅ | 无 | ✅ 正确 |
| 碰碰胡 PENG_PENG_HU | ✅ | 无 | ✅ 正确 |
| 风一色 ALL_WIND | ✅ | 无 | ✅ 正确 |
| 将一色 ALL_JIANG | ✅ | 无 | ✅ 正确 |
| 全求人 FULLY_MELDED | ✅ | 无 | ✅ 正确 |
| 杠上开花 FLOWER_ON_KONG | ✅ | `win_way == KONG_SELF_DRAW` | ✅ 正确 |
| 抢杠和 ROB_KONG | ✅ | `win_way == ROB_KONG` | ✅ 正确 |
| 海底捞月 LAST_TILE_WIN | ✅ | `wall <= 4` | ✅ 正确 |

---

## 四、起胡番检查流程分析

### 检查时机

`check_min_fan_requirement` 在以下两个时机被调用：

#### 1. ActionValidator 生成 action_mask 时

**位置**：
- `action_validator.py:211` - 自摸胡牌检查
- `action_validator.py:256` - 接炮胡牌检查
- `wait_rob_kong_state.py:226` - 抢杠和检查

**流程**：
```
1. 调用 WinChecker.check_win() 检测胡牌类型
2. 调用 check_min_fan_requirement() 检查起胡番
3. 如果满足，将 WIN 动作加入 action_mask
4. 玩家可以选择 WIN 动作
```

#### 2. WinState 计算分数时

**位置**：`win_state.py:106`

**流程**：
```
1. 进入 WIN 状态
2. 调用 WinChecker.check_win() 检测胡牌类型（包括大胡）
3. 调用 score_calculator.settle() 计算分数
```

### 设计评估

**优点**：
- ✅ 胡牌类型检测集中在 `WinChecker.check_win`
- ✅ 起胡番检查在动作验证阶段，避免无效动作
- ✅ 状态职责清晰，每个状态只负责自己的逻辑

**潜在问题**：
- ⚠️ `win_way` 生命周期管理不够严格
- ⚠️ 部分状态设置 `win_way` 后未清理

---

## 五、验证检查清单

### 功能验证

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 抢杠和检测 | ✅ | `WaitRobKongState._can_rob_kong` 正确调用 `check_min_fan_requirement` |
| 杠上开花检测 | ✅ | `DrawingAfterGongState` 设置 `win_way = KONG_SELF_DRAW`，`WinChecker` 正确检测 `FLOWER_ON_KONG` |
| 海底捞月检测 | ✅ | `WinChecker._check_last_tile_win` 检测牌墙剩余 ≤4 张 |
| 接炮胡牌检测 | ✅ | `WaitResponseState` 设置 `win_way = DISCARD`，`WinChecker` 正确检测 |
| 自摸胡牌检测 | ✅ | `PlayerDecisionState` 设置 `win_way = SELF_DRAW`，`WinState` 中检测大胡类型 |
| 起胡番计算 | ✅ | 修复后正确计算：胡牌类型基础分 × 口口翻 × 硬胡因子 |

### 测试覆盖

| 测试场景 | 文件 | 状态 |
|----------|------|------|
| 抢杠和状态转换 | `test_rob_kong_full_transition.py` | ⚠️ 待验证 |
| 接炮胡牌 | `test_win_by_discard.py` | ✅ 已通过 |
| 状态自动跳过 | `test_auto_skip_state.py` | ✅ 已通过 |

---

## 六、修复优先级

| 优先级 | 问题 | 文件 | 影响 | 工作量 |
|--------|------|------|------|--------|
| **P0** | `check_min_fan_requirement` 漏算胡牌类型基础分 | `score_calculator.py` | 导致抢杠和等大胡无法通过起胡番检查 | ✅ 已完成 |
| **P1** | `DrawingAfterGongState` 清理 `win_way` | `drawing_after_gong_state.py:93-98` | 代码清晰度和逻辑一致性 | 5分钟 |
| **P2** | 删除 `DrawingState` 死代码 | `drawing_state.py:89-97` | 代码维护性 | 5分钟 |
| **P3** | 修复 `WaitResponseState` 硬编码值 | `wait_response_state.py:214` | 代码质量 | 2分钟 |

---

## 七、建议的代码改进

### 1. win_way 生命周期管理

**建议**：只在确定进入 WIN 状态时设置 `win_way`

```python
# 当前模式（分散设置）
PlayerDecisionState: win_way = SELF_DRAW
WaitResponseState: win_way = DISCARD
DrawingAfterGongState: win_way = KONG_SELF_DRAW
WaitRobKongState: win_way = ROB_KONG

# 建议模式（集中设置）
WinState.enter():
    # 根据 context 属性判断 win_way
    if context.is_kong_draw and hasattr(context, 'last_drawn_tile'):
        win_way = KONG_SELF_DRAW
    elif context.last_discarded_tile is not None:
        win_way = DISCARD
    elif context.last_kong_tile is not None:
        win_way = ROB_KONG
    else:
        win_way = SELF_DRAW
    context.win_way = win_way.value
```

### 2. 状态机文档化

**建议**：在每个状态类的 docstring 中说明：
- 设置哪些 context 属性
- 依赖哪些 context 属性
- 清理哪些 context 属性

```python
class DrawingAfterGongState(GameState):
    """
    杠后补牌状态（自动状态）

    Context 操作：
    - 设置：last_drawn_tile, is_kong_draw, win_way（如果胡牌）
    - 依赖：current_player_idx, wall, is_kong_draw
    - 清理：is_kong_draw（在 exit 中）

    状态转换：
    - WIN: 如果杠上开花
    - PLAYER_DECISION: 正常情况
    """
```

---

## 八、总结

### 核心问题已修复 ✅
- `check_min_fan_requirement` 现在正确计算胡牌类型基础分
- 抢杠和等大胡类型现在能正确通过起胡番检查

### 待处理的代码质量问题 ⚠️
- P1: `DrawingAfterGongState` 清理 `win_way`
- P2: 删除 `DrawingState` 死代码
- P3: 修复 `WaitResponseState` 硬编码值

### 设计评估
- 整体架构合理，状态职责清晰
- 胡牌类型检测集中在 `WinChecker`
- 起胡番检查在动作验证阶段，设计正确
- 需要加强 `win_way` 生命周期管理和代码文档化

---

**下一步**：运行测试验证修复效果，并处理 P1-P3 代码质量问题。
