# 武汉麻将和牌场景测试 - 设计文档

## 文档信息

**创建日期**: 2026-02-02
**项目**: PettingZooRLENVMahjong
**目标**: 为所有和牌类型设计全面的测试场景

---

## 1. 场景测试框架使用方法

### 1.1 基本结构

```python
from tests.scenario.builder import ScenarioBuilder
from src.mahjong_rl.core.constants import GameStateType, ActionType

result = (
    ScenarioBuilder("场景名称")
    .with_initial_state({
        'dealer_idx': 0,           # 庄家位置
        'current_player_idx': 0,  # 当前玩家
        'hands': {                 # 玩家手牌
            0: [...],  # 庄家14张
            1: [...],  # 其他玩家13张
            2: [...],
            3: [...],
        },
        'wall': [...],             # 牌墙（预先计算）
        'special_tiles': {         # 特殊牌
            'lazy': 15,             # 赖子ID
            'skins': [14, 13],      # 皮子ID
        },
    })
    .step(1, "步骤描述")
        .action(0, ActionType.DISCARD, 26)
    .run()
)
```

### 1.2 步骤类型

#### 动作步骤（action）
```python
.step(1, "玩家0出牌")
    .action(0, ActionType.DISCARD, 26)
    .expect_state(GameStateType.WAITING_RESPONSE)
```

#### 自动步骤（auto_advance）
```python
.step(2, "自动摸牌")
    .auto_advance()
    .expect_state(GameStateType.PLAYER_DECISION)
```

#### 验证步骤
```python
.step(3, "验证状态")
    .action(0, ActionType.PASS)
    .verify("验证描述", lambda ctx: len(ctx.players[0].hand_tiles) == 14)
```

### 1.3 牌墙计算方法

```python
def calculate_wall():
    # 初始手牌
    p0 = [...]  # 14张（庄家）
    p1 = [...]  # 13张
    p2 = [...]  # 13张
    p3 = [...]  # 13张

    # 游戏流程中摸走的牌（按顺序）
    drawn_tiles = [8, 9, 2, 25]  # p1摸8, p2摸9, p3摸2, p0摸25

    # 计算牌墙
    all_used = p0 + p1 + p2 + p3 + drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    wall = drawn_tiles + unused_wall
    return wall
```

**注意**：`drawn_tiles` 列表必须放在牌墙前面，确保按顺序摸牌。

### 1.4 牌值对照

```
万子：0-8  (1万=0, 2万=1, ..., 9万=8)
条子：9-17 (1条=9, 2条=10, ..., 9条=17)
筒子：18-26 (1筒=18, 2筒=19, ..., 9筒=26)
风牌：27-33 (东=27, 南=28, 西=29, 北=30, 中=31, 发=32, 白=33)
```

---

## 2. 状态机自动跳过机制

### 2.1 发现过程

在测试中，我们尝试让每个玩家都执行PASS动作：

```python
.step(1, "玩家3出牌")
    .action(3, ActionType.DISCARD, 2)
.step(2, "玩家0 PASS")
    .action(0, ActionType.PASS)
.step(3, "玩家1 PASS")
    .action(1, ActionType.PASS)  # ❌ 失败：action_mask中没有PASS
.step(4, "玩家2 PASS")
    .action(2, ActionType.PASS)
```

**结果**：步骤3失败，因为当前玩家不是玩家1。

### 2.2 原理分析

通过阅读状态机代码，发现了**自动跳过机制**：

```python
# machine.py: transition_to()
def transition_to(self, next_state_type: GameStateType, context: GameContext) -> None:
    self.current_state = self.states[next_state_type]
    self.current_state.enter(context)

    # 检查是否需要自动跳过
    if self.current_state.should_auto_skip(context):
        self._auto_skip_state(context)
```

**WAITING_RESPONSE 状态的自动跳过逻辑**：

```python
# wait_response_state.py: enter()
def enter(self, context: GameContext) -> None:
    # 初始化响应收集器
    context.response_collector = ResponseCollector()

    # 构建响应者列表
    context.active_responders = []
    for responder_id in context.response_order:
        if not self._can_only_pass(context, responder_id):
            context.active_responders.append(responder_id)
        else:
            # 只能 PASS，自动添加响应
            context.response_collector.add_response(
                responder_id, ActionType.PASS,
                ResponsePriority.PASS, -1
            )
```

```python
# wait_response_state.py: should_auto_skip()
def should_auto_skip(self, context: GameContext) -> bool:
    """如果所有玩家都只能 PASS，则自动跳过"""
    return len(context.active_responders) == 0
```

### 2.3 状态转换流程

当所有玩家都只能PASS时：

```
WAITING_RESPONSE.enter()
  ↓
构建 active_responders（如果都只能PASS，则为空）
  ↓
transition_to() 检测到 should_auto_skip() == True
  ↓
_auto_skip_state() 调用 step(context, 'auto')
  ↓
step() 检测到 active_responders 为空
  ↓
_select_best_response() 返回 DRAWING
  ↓
transition_to(DRAWING)
  ↓
DRAWING.enter()
  ↓
transition_to() 检测到 should_auto_skip() == True（DRAWING是自动状态）
  ↓
_auto_skip_state() 调用 step(context, 'auto')
  ↓
DRAWING.step() 从牌墙摸牌
  ↓
返回 PLAYER_DECISION
  ↓
transition_to(PLAYER_DECISION)
```

**关键**：所有这些转换都在**一次 env.step() 调用**中完成！

### 2.4 env.step() 的自动推进循环

```python
# example_mahjong_env.py: step()
def step(self, action):
    # 执行状态机step
    next_state_type = self.state_machine.step(self.context, mahjong_action)

    # 自动推进循环
    while not self.state_machine.is_terminal():
        current_state = self.state_machine.current_state_type

        # 需要agent动作的状态 - 停止自动推进
        if current_state in [
            GameStateType.PLAYER_DECISION,
            GameStateType.MELD_DECISION,
            GameStateType.WAITING_RESPONSE,  # ← 包含 WAITING_RESPONSE
            GameStateType.WAIT_ROB_KONG
        ]:
            break

        # 其他状态都是自动状态，使用'auto'推进
        next_state_type = self.state_machine.step(self.context, 'auto')
```

**重要发现**：WAITING_RESPONSE 被列为需要agent动作的状态！

这意味着：
- 如果 active_responders 不为空，循环会停止在 WAITING_RESPONSE
- 如果 active_responders 为空（都只能PASS），自动跳过机制会处理所有玩家的PASS，然后进入 DRAWING

---

## 3. 测试设计关键发现

### 3.1 不需要显式处理所有玩家的PASS

**错误设计**：
```python
# ❌ 错误：为每个玩家都添加PASS步骤
.step(1, "玩家3出牌")
    .action(3, ActionType.DISCARD, 2)
.step(2, "玩家0 PASS")
    .action(0, ActionType.PASS)
.step(3, "玩家1 PASS")  # ❌ 失败
    .action(1, ActionType.PASS)
.step(4, "玩家2 PASS")  # ❌ 失败
    .action(2, ActionType.PASS)
```

**正确设计**：
```python
# ✅ 正确：只让一个玩家执行PASS
.step(1, "玩家3出牌")
    .action(3, ActionType.DISCARD, 2)
.step(2, "玩家0 PASS")  # 触发自动跳过，所有玩家PASS
    .action(0, ActionType.PASS)
# env.step() 会自动处理 WAITING_RESPONSE → DRAWING → PLAYER_DECISION
```

### 3.2 WAITING_RESPONSE 状态的响应收集

**场景1**：所有玩家都只能PASS
- active_responders 为空
- should_auto_skip() 返回 True
- 自动跳过到 DRAWING
- 下一个玩家摸牌

**场景2**：有玩家可以鸣牌
- active_responders 包含可以鸣牌的玩家
- 需要依次为每个玩家执行 step()
- 最后调用 _select_best_response() 选择最佳响应

### 3.3 测试流程分析

**硬胡自摸场景（部分）**：

```python
# 步骤21：玩家3出2
.step(21, "玩家3出2")
    .action(3, ActionType.DISCARD, 2)
    # → DISCARDING → WAITING_RESPONSE

# 步骤22：玩家0 PASS（触发自动跳过）
.step(22, "玩家0 PASS")
    .action(0, ActionType.PASS)
    # → WAITING_RESPONSE step()
    #   → active_responders 为空（都只能PASS）
    #   → _auto_skip_state()
    #   → DRAWING step()（玩家0摸牌）
    #   → PLAYER_DECISION

# 步骤23：玩家0自摸硬胡
.step(23, "玩家0自摸硬胡")
    .action(0, ActionType.WIN, -1)
    # → WIN
```

---

## 4. 当前问题与解决方案

### 4.1 当前问题

**硬胡自摸场景测试失败**：
```
步骤22后: 状态=PLAYER_DECISION, 玩家0手牌=14, wall_count=79
手牌=[0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 25]

步骤23: 玩家0 WIN(-1)
预期: WIN 状态
实际: PLAYER_DECISION 状态
```

**牌型分析**：
- `[0, 0, 0]` - 1万x3
- `[1, 2, 3]` - 2万,3万,4万
- `[4, 5, 6]` - 5万,6万,7万
- `[24, 24, 24]` - 8筒x3
- `[25, 25]` - 9筒x2

**目标牌型**：`111 234 567 888 99` - 完整胡牌牌型 ✓

### 4.2 可能的原因

1. **玩家0没有开口**（melds为空或exposure_count=0）
   - 武汉麻将规则：胡牌必须开口（吃/碰/明杠/补杠）

2. **胡牌检测失败**
   - 可能是牌型验证问题
   - 可能是赖子还原逻辑问题

3. **WIN动作不在action_mask中**
   - 可能是available_actions检测问题

### 4.3 解决方案方向

1. **添加调试信息**：
   ```python
   .verify("检查开口", lambda ctx: print(f"melds={ctx.players[0].melds}, exposure_count={ctx.players[0].exposure_count}") or True)
   ```

2. **检查胡牌检测逻辑**：
   - 验证牌型是否符合硬胡条件
   - 验证将牌是否为2/5/8
   - 验证赖子数量

3. **简化测试场景**：
   - 先测试简单的自摸场景（不涉及碰牌）
   - 逐步增加复杂度

---

## 5. 已完成的场景设计

### 5.1 场景1：硬胡自摸（部分完成）

**目的**：验证无赖子、将牌为2/5/8的自摸硬胡

**初始手牌**：
```python
玩家0（庄家）：[0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 26]
              = 1万x3, 2万,3万,4万, 5万,6万,7万, 8筒x3, 9筒, 9筒

特殊牌：
  赖子：15（2条）
  皮子：14, 13
```

**游戏流程**（已完成部分）：
1. 玩家0出26（9筒）- 听牌
2. 玩家1,2,3依次PASS
3. 玩家1出0（1万）
4. 玩家2,3依次PASS，玩家0 PONG 0（完成开口）
5. 玩家0出3（4万）
6. 玩家1,2,3依次PASS
7. 玩家1出8
8. 玩家2,3,0依次PASS
9. 玩家2出9
10. 玩家3,0,1依次PASS
11. 玩家3出2
12. 玩家0 PASS（触发自动跳过，玩家0摸牌）
13. ✅ 玩家0摸到25（9筒）
14. ❓ 玩家0 WIN - 待验证

**当前状态**：
- ✅ 状态机自动跳过机制验证通过
- ✅ 玩家0成功摸牌
- ✅ 牌型完整（111 234 567 888 99）
- ❌ 胡牌检测待验证

### 5.2 场景2：软胡接炮（未完成）

**目的**：验证1个赖子未还原的接炮软胡

### 5.3 场景3：杠上开花（未完成）

**目的**：验证杠后补牌自摸胡牌（大胡）

---

## 6. 设计原则总结

### 6.1 测试流程设计原则

1. **使用 with_initial_state 设置初始状态**
   - 庄家14张，其他13张
   - 精确控制每个玩家的手牌

2. **预先计算牌墙**
   - 包含游戏流程中所有摸走的牌
   - drawn_tiles 必须放在牌墙前面

3. **充分利用自动跳过机制**
   - 不需要为每个玩家都添加PASS步骤
   - WAITING_RESPONSE 会自动处理所有玩家的PASS

4. **验证关键状态**
   - 使用 verify() 添加验证条件
   - 使用 expect_state() 验证状态转换

### 6.2 状态机关键状态

**手动状态**（需要agent动作）：
- `PLAYER_DECISION` - 玩家决策（打牌、杠牌、自摸和牌）
- `MELD_DECISION` - 吃牌决策
- `WAITING_RESPONSE` - 等待响应（可自动跳过）
- `WAIT_ROB_KONG` - 等待抢杠和（可自动跳过）

**自动状态**（自动推进）：
- `DRAWING` - 摸牌
- `DISCARDING` - 出牌
- `PROCESSING_MELD` - 处理吃碰
- `GONG` - 杠牌
- `DRAWING_AFTER_GONG` - 杠后补牌

### 6.3 调试技巧

1. **添加状态验证**：
   ```python
   .verify("检查状态", lambda ctx: print(f"状态={ctx.current_state.name}, 手牌={len(ctx.players[0].hand_tiles)}") or True)
   ```

2. **打印详细信息**：
   ```python
   .verify("检查手牌", lambda ctx: print(f"手牌={sorted(ctx.players[0].hand_tiles)}") or True)
   ```

3. **查看最终快照**：
   ```python
   print(f"最终状态: {result.final_context_snapshot}")
   ```

---

## 7. 下一步工作

### 7.1 调试当前问题

1. 确认玩家0的melds是否正确记录
2. 确认exposure_count是否正确
3. 检查胡牌检测逻辑
4. 验证WIN动作是否在action_mask中

### 7.2 完成剩余场景

1. 场景1：硬胡自摸（调试中）
2. 场景2：软胡接炮
3. 场景3：杠上开花
4. 场景4：全求人
5. 场景5：清一色
6. 场景6：碰碰胡
7. 场景7：风一色
8. 场景8：将一色
9. 场景9：海底捞月
10. 场景10：抢杠和（新设计）
11. 场景11：赖子还原硬胡
12. 场景12-15：边界条件测试

### 7.3 优化测试框架

1. 添加更详细的调试输出
2. 改进错误提示信息
3. 添加场景模板和示例
4. 编写场景测试最佳实践文档

---

## 8. 参考代码

### 8.1 抢杠和场景测试（参考）

**文件位置**：`tests/integration/test_rob_kong_scenario.py`

**关键发现**：
- 使用 `with_initial_state` 自定义初始状态
- 完整模拟游戏流程
- 包含碰牌、补杠、抢杠和等复杂交互

### 8.2 场景测试框架

**文件位置**：`tests/scenario/`

- `builder.py` - 场景构建器
- `executor.py` - 测试执行器
- `context.py` - 场景上下文
- `validators.py` - 验证器

---

## 9. 附录

### 9.1 武汉麻将规则

**规则文档**：`src/mahjong_rl/rules/wuhan_mahjong_rule_engine/wuhan_mahjong_rules.md`

**关键规则**：
- 起胡番数：乘积 ≥ 16
- 必须开口：吃/碰/明杠/补杠
- 小胡赖子限制：≤1个
- 大胡赖子限制：≤2个
- 胡牌时不能有皮或红中

### 9.2 状态机流程图

详见 `CLAUDE.md` 中的 Mermaid 状态图。

### 9.3 牌型示例

**硬胡示例**：
```
手牌：[1万x3, 2万,3万,4万, 5万,6万,7万, 8筒x3, 9筒]
      = 111 234 567 888 9

碰牌后：[1万x3(碰), 2万,3万,4万, 5万,6万,7万, 8筒x3, 9筒, 9筒]
      = 111 234 567 888 99

胡牌：摸9筒 → 111 234 567 888 99 (硬胡)
```

---

**文档版本**: 1.0
**最后更新**: 2026-02-02
**作者**: Claude Code
