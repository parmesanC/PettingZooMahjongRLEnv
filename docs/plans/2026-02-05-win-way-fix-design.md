# WinWay 设置修复设计文档

## 问题背景

### WinWay 的正确含义

`win_way` 表示"**当前处于哪种可能的胡牌场景**"，有四种类型：

| WinWay | 值 | 场景描述 | 入口状态 |
|--------|---|----------|----------|
| SELF_DRAW | 0 | 自摸：摸牌阶段可以胡牌 | `DRAWING` → `PLAYER_DECISION` |
| ROB_KONG | 1 | 抢杠：其他玩家补杠阶段可以胡牌 | `GONG` → `WAIT_ROB_KONG` |
| KONG_SELF_DRAW | 2 | 杠开：杠后补牌阶段可以胡牌 | `DRAWING_AFTER_GONG` → `PLAYER_DECISION` |
| DISCARD | 3 | 点炮：其他玩家弃牌后响应阶段可以胡牌 | `DISCARDING` → `WAITING_RESPONSE` |

**关键理解**：
- `win_way` 应该在**进入检测胡牌的状态之前**就设置
- 它是场景标识，不是"玩家已经选择胡牌"的标志
- 这样 `check_win()` 中的各种检测方法（如 `_check_flower_on_kong()`）才能正确判断当前场景

## 当前问题分析

### 问题1：四种场景都没有正确设置 win_way

| 场景 | 当前代码位置 | 状态 | 问题 |
|------|--------------|------|------|
| **SELF_DRAW** | `drawing_state.py:99` | ❌ 未设置 | 摸牌后直接返回 `PLAYER_DECISION`，未设置 `win_way` |
| **KONG_SELF_DRAW** | `drawing_after_gong_state.py:93` | ❌ 位置错误 | 在检测到胡牌后才设置 `win_way`，应在摸牌后立即设置 |
| **DISCARD** | `discarding_state.py:119` | ❌ 未设置 | 出牌后直接返回 `WAITING_RESPONSE`，未设置 `win_way` |
| **ROB_KONG** | `gong_state.py:133` | ❌ 未设置 | 补杠时直接返回 `WAIT_ROB_KONG`，未设置 `win_way` |

### 问题2：PlayerDecisionState 错误地设置 win_way

**文件**: `src/mahjong_rl/state_machine/states/player_decision_state.py` (line 269)

```python
context.win_way = WinWay.SELF_DRAW.value
```

**问题**：
- 这里是在玩家选择 WIN 后才设置 `win_way`
- 但此时 `check_win()` 已经调用过了，设置 `win_way` 已经太晚
- 应该在进入 `PLAYER_DECISION` 状态之前就设置好 `win_way`

### 问题3：DrawingAfterGongState 自动胡牌

**文件**: `src/mahjong_rl/state_machine/states/drawing_after_gong_state.py` (lines 90-96)

```python
# 检查杠上开花（自己胡这张牌）
win_result = self._check_win(context, current_player)
if win_result.can_win:
    context.win_way = WinWay.KONG_SELF_DRAW.value
    context.winner_ids = [context.current_player_idx]
    context.is_win = True
    return GameStateType.WIN  # 直接跳到 WIN，玩家无法选择
```

**问题**：
- 自动检测胡牌并直接跳到 WIN 状态
- 剥夺了玩家选择是否胡牌的权利
- 应该让玩家在 `PLAYER_DECISION` 状态选择

## 修复方案设计

### 原则

1. **提前设置**：`win_way` 应在进入检测胡牌的状态**之前**设置
2. **统一位置**：在状态转换处设置，而不是在状态内部
3. **清理遗留**：移除 `PlayerDecisionState._handle_win()` 中的 `win_way` 设置代码

### 修改清单

#### 1. DRAWING 状态 - 设置 SELF_DRAW

**文件**: `src/mahjong_rl/state_machine/states/drawing_state.py`

**修改位置**: `step()` 方法末尾

**修改前** (line 99):
```python
return GameStateType.PLAYER_DECISION
```

**修改后**:
```python
# 设置胡牌场景为自摸
context.win_way = WinWay.SELF_DRAW.value
return GameStateType.PLAYER_DECISION
```

#### 2. DRAWING_AFTER_GONG 状态 - 设置 KONG_SELF_DRAW 并移除自动胡牌

**文件**: `src/mahjong_rl/state_machine/states/drawing_after_gong_state.py`

**修改位置**: `step()` 方法 (lines 82-98)

**修改前**:
```python
# 摸牌
draw_tile = context.wall.pop()
current_player = context.players[context.current_player_idx]
current_player.hand_tiles.append(draw_tile)

# 存储摸到的牌供PLAYER_DECISION状态使用
context.last_drawn_tile = draw_tile

# 检查杠上开花（自己胡这张牌）
win_result = self._check_win(context, current_player)
if win_result.can_win:
    context.win_way = WinWay.KONG_SELF_DRAW.value
    context.winner_ids = [context.current_player_idx]
    context.is_win = True
    return GameStateType.WIN

return GameStateType.PLAYER_DECISION
```

**修改后**:
```python
# 摸牌
draw_tile = context.wall.pop()
current_player = context.players[context.current_player_idx]
current_player.hand_tiles.append(draw_tile)

# 存储摸到的牌供PLAYER_DECISION状态使用
context.last_drawn_tile = draw_tile

# 设置胡牌场景为杠上开花
context.win_way = WinWay.KONG_SELF_DRAW.value

# 不自动检测胡牌，让玩家在PLAYER_DECISION状态选择
return GameStateType.PLAYER_DECISION
```

#### 3. DISCARDING 状态 - 设置 DISCARD

**文件**: `src/mahjong_rl/state_machine/states/discarding_state.py`

**修改位置**: `step()` 方法末尾

**修改前** (line 119):
```python
return GameStateType.WAITING_RESPONSE
```

**修改后**:
```python
# 设置胡牌场景为点炮
context.win_way = WinWay.DISCARD.value
return GameStateType.WAITING_RESPONSE
```

#### 4. GONG 状态 - 设置 ROB_KONG

**文件**: `src/mahjong_rl/state_machine/states/gong_state.py`

**修改位置**: 补杠时返回 `WAIT_ROB_KONG` 之前

**修改前** (lines 120-133):
```python
# 对于补杠，需要先检查是否可以抢杠和
if kong_type == ActionType.KONG_SUPPLEMENT:
    # 设置被抢杠的牌
    context.rob_kong_tile = kong_tile
    context.kong_player_idx = player_id

    # 保存杠牌动作，供后续使用
    context.saved_kong_action = kong_action

    # 清理临时变量
    context.pending_kong_action = None

    # 转到等待抢杠和状态
    return GameStateType.WAIT_ROB_KONG
```

**修改后**:
```python
# 对于补杠，需要先检查是否可以抢杠和
if kong_type == ActionType.KONG_SUPPLEMENT:
    # 设置被抢杠的牌
    context.rob_kong_tile = kong_tile
    context.kong_player_idx = player_id

    # 保存杠牌动作，供后续使用
    context.saved_kong_action = kong_action

    # 清理临时变量
    context.pending_kong_action = None

    # 设置胡牌场景为抢杠和
    context.win_way = WinWay.ROB_KONG.value

    # 转到等待抢杠和状态
    return GameStateType.WAIT_ROB_KONG
```

#### 5. PlayerDecisionState - 移除 win_way 设置

**文件**: `src/mahjong_rl/state_machine/states/player_decision_state.py`

**修改位置**: `_handle_win()` 方法 (lines 266-271)

**修改前**:
```python
# 设置游戏状态为和牌
context.is_win = True
context.winner_ids = [context.current_player_idx]
context.win_way = WinWay.SELF_DRAW.value

return GameStateType.WIN
```

**修改后**:
```python
# 设置游戏状态为和牌
context.is_win = True
context.winner_ids = [context.current_player_idx]
# win_way 已在进入状态前设置，这里不需要再设置

return GameStateType.WIN
```

### 状态转换图（修复后）

```
DRAWING (摸牌)
  └─> win_way = SELF_DRAW
  └─> PLAYER_DECISION

DRAWING_AFTER_GONG (杠后补牌)
  └─> win_way = KONG_SELF_DRAW
  └─> PLAYER_DECISION

DISCARDING (出牌)
  └─> win_way = DISCARD
  └─> WAITING_RESPONSE

GONG (补杠)
  └─> win_way = ROB_KONG
  └─> WAIT_ROB_KONG
```

## 测试计划

### 单元测试

1. **DRAWING 状态**：
   - 验证摸牌后设置 `win_way = WinWay.SELF_DRAW.value`
   - 验证转到 `PLAYER_DECISION` 状态

2. **DRAWING_AFTER_GONG 状态**：
   - 验证补牌后设置 `win_way = WinWay.KONG_SELF_DRAW.value`
   - 验证不再自动检测胡牌
   - 验证转到 `PLAYER_DECISION` 状态

3. **DISCARDING 状态**：
   - 验证出牌后设置 `win_way = WinWay.DISCARD.value`
   - 验证转到 `WAITING_RESPONSE` 状态

4. **GONG 状态**：
   - 验证补杠时设置 `win_way = WinWay.ROB_KONG.value`
   - 验证转到 `WAIT_ROB_KONG` 状态

5. **PlayerDecisionState._handle_win()**：
   - 验证不再设置 `win_way`
   - 验证使用之前设置的 `win_way` 值

### 集成测试

1. **自摸场景**：
   - 摸牌后能胡牌
   - 验证 `win_way = SELF_DRAW`
   - 验证 `_check_flower_on_kong()` 返回 False

2. **杠上开花场景**：
   - 杠后补牌能胡牌
   - 验证 `win_way = KONG_SELF_DRAW`
   - 验证 `_check_flower_on_kong()` 返回 True
   - 验证玩家可以选择 WIN 或 DISCARD

3. **点炮场景**：
   - 出牌后其他玩家能胡牌
   - 验证 `win_way = DISCARD`
   - 验证 `_check_rob_kong()` 返回 False

4. **抢杠场景**：
   - 补杠时其他玩家能胡牌
   - 验证 `win_way = ROB_KONG`
   - 验证 `_check_rob_kong()` 返回 True

## 实现步骤

1. 修改 `DRAWING.step()` - 添加 `win_way = SELF_DRAW.value`
2. 修改 `DRAWING_AFTER_GONG.step()` - 移除自动胡牌，添加 `win_way = KONG_SELF_DRAW.value`
3. 修改 `DISCARDING.step()` - 添加 `win_way = DISCARD.value`
4. 修改 `GONG.step()` - 补杠时添加 `win_way = ROB_KONG.value`
5. 修改 `PlayerDecisionState._handle_win()` - 移除 `win_way` 设置
6. 运行测试验证
7. 提交代码

## 注意事项

1. **向后兼容**：使用 `WinWay.SELF_DRAW.value` 等常量，而不是硬编码数字
2. **清理标志**：确保 `is_kong_draw` 等临时标志在适当的位置清理
3. **测试覆盖**：确保四种场景都有对应的测试用例
