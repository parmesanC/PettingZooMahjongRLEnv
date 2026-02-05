# 杠上开花玩家选择问题分析

## 问题描述

### 当前行为
在 `DrawingAfterGongState`（杠后补牌状态）中，系统会自动检测玩家是否胡牌（杠上开花），如果胡牌则直接跳转到 `WIN` 状态，玩家没有选择的机会。

### 期望行为
杠后补牌后，应该让玩家在 `PLAYER_DECISION` 状态选择是否胡牌，就像普通摸牌一样。玩家可以选择：
- `WIN` - 胡牌（杠上开花）
- `DISCARD` - 打牌

## 代码缺陷分析

### 缺陷1：DrawingAfterGongState 自动胡牌检测

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

**问题**：玩家杠后摸牌，如果刚好能胡，系统自动判定胡牌并跳转到 WIN 状态，玩家没有选择打牌的机会。

### 缺陷2：循环依赖问题

**文件**: `src/mahjong_rl/rules/wuhan_mahjong_rule_engine/win_detector.py` (line 372)

```python
def _check_flower_on_kong(self, hand_tiles: List[int]) -> Tuple[bool, int]:
    """检测杠上开花"""
    if not self.game_context.win_way == WinWay.KONG_SELF_DRAW.value:
        return False, -1
    # ...
```

**循环依赖链**：
1. `PlayerDecisionState` 需要检测 WIN 动作是否合法
2. 调用 `check_win()` 来检测
3. `check_win()` 调用 `detect_big_wins()`
4. `detect_big_wins()` 调用 `_check_flower_on_kong()`
5. `_check_flower_on_kong()` 检查 `context.win_way == WinWay.KONG_SELF_DRAW.value`
6. **但 `win_way` 只有在玩家选择 WIN 后才在 `PlayerDecisionState._handle_win()` 中设置**
7. **矛盾**：检测 WIN 动作是否合法时，`win_way` 还没设置！

**后果**：如果直接移除 `DrawingAfterGongState` 的自动胡牌检测，转到 `PLAYER_DECISION` 让玩家选择，会导致：
- 玩家能胡牌时，`check_win()` 返回 `can_win=False`（因为 `win_way` 没设置）
- WIN 动作不会被添加到可用动作列表
- 玩家无法选择胡牌

### 缺陷3：ScoreCalculator 缺少 KONG_SELF_DRAW 自摸番数

**文件**: `src/mahjong_rl/rules/wuhan_mahjong_rule_engine/score_calculator.py` (lines 125-129)

```python
if ctx.win_way == WinWay.SELF_DRAW.value:
    if self._is_big_hu(win_result):
        fan_components.append(1.5)  # 大胡自摸乘1.5
    else:
        fan_components.append(2.0)  # 小胡自摸乘2
```

**问题**：只检查了 `WinWay.SELF_DRAW`，没有检查 `WinWay.KONG_SELF_DRAW`。杠上开花（杠上自摸）应该也享受自摸番数加成。

## 参考：抢杠和为什么没问题？

抢杠和的 `_check_rob_kong()` 也依赖 `win_way`：

```python
def _check_rob_kong(self, hand_tiles: List[int]) -> Tuple[bool, int]:
    """检测抢杠胡"""
    if not self.game_context.win_way == WinWay.ROB_KONG.value:
        return False, -1
    # ...
```

**为什么抢杠和没有循环依赖问题？**

抢杠和有**专门的状态** `WaitRobKongState` 处理：
1. `WaitRobKongState._can_rob_kong()` 调用 `check_win()` 检测玩家能否胡牌
2. 此时 `win_way` 还没设置，所以 `_check_rob_kong()` 返回 `(False, -1)`
3. 但只要玩家能普通胡牌，`check_win()` 就返回 `can_win=True`，不影响判断
4. 等到玩家选择 WIN 后，在 `WaitRobKongState.step()` 中设置 `win_way = WinWay.ROB_KONG.value`

**关键区别**：
- 抢杠和：在**专门的状态** (`WaitRobKongState`) 中处理，不需要在 `PlayerDecisionState` 中检测
- 杠上开花：需要在 `PlayerDecisionState` 中检测 WIN 动作是否合法，导致循环依赖

## 设计问题总结

1. **违反单一职责原则**：`DrawingAfterGongState` 既负责补牌又负责胡牌检测
2. **违反依赖倒置原则**：`_check_flower_on_kong()` 依赖 `win_way`（一个由其他状态设置的后期变量）
3. **信息隐藏不当**：使用 `win_way` 既表示"胡牌方式"又作为"检测标志"，职责混乱
4. **生命周期不清晰**：`win_way` 在 `PlayerDecisionState` 检测时需要，但在之后才设置

## 影响

1. **游戏体验**：玩家杠后摸牌时，如果手牌能胡，系统自动胡牌，玩家没有选择打牌继续游戏的机会
2. **规则正确性**：根据武汉麻将规则，杠上开花是玩家可以主动选择的大胡类型，不应该自动触发
3. **代码可维护性**：循环依赖使得修改任一相关状态都可能影响其他状态
