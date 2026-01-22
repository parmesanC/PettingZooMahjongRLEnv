# 设计文档：状态拆分方案 - 分离摸牌决策与鸣牌决策

## 文档信息

| 项目 | 内容 |
|------|------|
| **目标** | 拆分 PLAYER_DECISION 状态，解决语义混乱和 PASS 动作错误问题 |
| **日期** | 2026-01-22 |
| **状态** | 已实施 |
| **作者** | Claude Code |

---

## 1. 问题分析

### 1.1 原始问题

在实施过程中发现 `PLAYER_DECISION` 状态下出现了不应该存在的 PASS 动作（动作类型10）：

```
步骤 9：当前agent: player_2
当前状态: PLAYER_DECISION
剩余牌墙: 81张
player_2 动作: (10, 0)  # PASS 动作
```

**麻将规则**：出牌阶段必须不能出现 PASS 动作，玩家必须打出一张牌或执行杠/胡操作。

### 1.2 深入分析发现的设计缺陷

通过深入分析代码，发现了多个设计缺陷：

#### 缺陷 1：`PLAYER_DECISION` 状态语义混乱

`PLAYER_DECISION` 被用于两种不同的场景：

| 进入路径 | 场景描述 | `last_drawn_tile` 含义 |
|---------|---------|----------------------|
| DrawingState → PLAYER_DECISION | 摸牌后决策 | 刚摸到的牌 |
| DrawingAfterGongState → PLAYER_DECISION | 杠后补牌决策 | 刚补到的牌 |
| InitialState → PLAYER_DECISION | 庄家开局决策 | 庄家额外摸的牌 |
| **ProcessMeldState → PLAYER_DECISION** | **鸣牌后出牌** | **上一次摸的牌（错误！）** |

**问题**：鸣牌后没有新的摸牌，但 `last_drawn_tile` 还保留着旧值，导致语义错误。

#### 缺陷 2：`ActionValidator.detect_available_actions_after_draw()` 重复添加牌

```python
# action_validator.py 第153-154行
temp_hand = current_player.hand_tiles.copy()
temp_hand.append(draw_tile)  # ← 无条件添加，可能重复！
```

**问题**：
- `DrawingState` 已经将摸到的牌加入 `hand_tiles`（第84行）
- `ActionValidator` 又无条件添加一次，导致 `Counter` 计数错误
- 可能将 3 张牌误判为 4 张，错误地允许暗杠

#### 缺陷 3：`_build_draw_mask()` 依赖 `last_drawn_tile`

```python
# wuhan_7p4l_observation_builder.py 第102-104行
last_tile = context.last_drawn_tile
if last_tile is None and len(player.hand_tiles) > 0:
    last_tile = player.hand_tiles[-1]

if last_tile is not None:
    actions = ActionValidator(context).detect_available_actions_after_draw(...)
```

**问题**：当 `last_tile` 为 `None` 时（比如鸣牌后），返回空的 mask，导致 `RandomStrategy` 默认返回 PASS。

#### 缺陷 4：`RandomStrategy` 的后备逻辑

```python
# random_strategy.py 第97-98行
if len(available_actions) == 0:
    return (ActionType.PASS.value, 0)  # 默认返回 PASS
```

**问题**：在决策状态下不应该有 PASS 选项，这个后备逻辑违反了麻将规则。

---

## 2. 解决方案设计

### 2.1 核心设计理念

**状态职责单一化**：将 `PLAYER_DECISION` 拆分为两个语义明确的状态

| 状态 | 名称 | 职责 | 可用动作 | 不能 |
|-----|------|------|---------|------|
| **PLAYER_DECISION** | 摸牌后决策 | 摸牌后的决策 | DISCARD, 各种KONG, **WIN** | PASS |
| **MELD_DECISION** | 鸣牌后决策 | 鸣牌后的决策 | DISCARD, 各种KONG | **WIN, PASS** |

**关键区别**：
- 两者都可以杠（暗杠、补杠、红中杠、赖子杠、皮子杠）
- 两者都必须出牌（不能 PASS）
- **区别**：摸牌后可以胡，鸣牌后不能胡

### 2.2 状态转换图

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                      摸牌流程                                  │
                    ├─────────────────────────────────────────────────────────────┤
                    │                                                             │
                    │  DrawingState / DrawingAfterGongState                       │
                    │  (摸牌/杠后补牌)                                              │
                    │         ↓                                                   │
                    │  PLAYER_DECISION                                            │
                    │  (摸牌后决策)                                                │
                    │  - 可以：暗杠、补杠、红中杠、赖子杠、皮子杠、胡牌、出牌         │
                    │  - 不能：过                                                 │
                    │         ↓                                                   │
                    │    ┌────┴─────┐              ┌──────────┐                    │
                    │    │          │              │          │                    │
                    │    ↓          ↓              ↓          ↓                    │
                    │  GONG       WIN          DISCARDING   (出牌)                  │
                    │  (杠牌)     (胡牌)        (执行出牌)                          │
                    │                                        ↓                    │
                    │                                   WAITING_RESPONSE          │
                    │                                   (等待其他玩家响应)          │
                    └─────────────────────────────────────────────────────────────┘


                    ┌─────────────────────────────────────────────────────────────┐
                    │                      鸣牌流程                                  │
                    ├─────────────────────────────────────────────────────────────┤
                    │                                                             │
                    │  WAITING_RESPONSE                                           │
                    │  (收集响应)                                                  │
                    │         ↓                                                   │
                    │  ProcessMeldState                                          │
                    │  (处理吃/碰)                                                 │
                    │         ↓                                                   │
                    │  MELD_DECISION ──────────────────→ WAITING_RESPONSE         │
                    │  (鸣牌后决策)                      (出牌后等待响应)           │
                    │  - 可以：暗杠、补杠、红中杠、赖子杠、皮子杠、出牌             │
                    │  - 不能：胡牌、过                                             │
                    │         ↓                                                   │
                    │    ┌────┴─────┐                                              │
                    │    │          │                                              │
                    │    ↓          ↓                                              │
                    │  GONG      DISCARDING                                        │
                    │  (杠牌)     (执行出牌)                                        │
                    └─────────────────────────────────────────────────────────────┘
```

### 2.3 架构设计

#### 2.3.1 新状态：MeldDecisionState

**职责**：处理鸣牌（吃/碰）后的决策逻辑

**关键特性**：
- 手动状态，需要 agent 传入动作
- 支持所有杠牌操作（暗杠、补杠、红中杠、赖子杠、皮子杠）
- 支持出牌操作
- **明确拒绝** WIN 和 PASS 动作

**动作处理器映射**：
```python
self.action_handlers = {
    ActionType.DISCARD: self._handle_discard,
    ActionType.KONG_SUPPLEMENT: self._handle_supplement_kong,
    ActionType.KONG_CONCEALED: self._handle_concealed_kong,
    ActionType.KONG_RED: self._handle_red_kong,
    ActionType.KONG_SKIN: self._handle_skin_kong,
    ActionType.KONG_LAZY: self._handle_lazy_kong,
}
# 不包含 WIN 和 PASS，会在 _handle_default 中拒绝
```

#### 2.3.2 ObservationBuilder 修改

**修改点 1**：添加 `MELD_DECISION` 状态支持

```python
STATE_TO_PHASE = {
    ...
    GameStateType.MELD_DECISION: 4,  # 鸣牌后决策
    ...
}
```

**修改点 2**：重构 `_build_draw_mask` 为 `_build_decision_mask`

```python
def _build_decision_mask(self, player, context, mask):
    """
    摸牌后决策掩码（PLAYER_DECISION）
    - 不再依赖 last_drawn_tile
    - 传入 None 给 ActionValidator，让检测基于手牌进行
    """
    actions = ActionValidator(context).detect_available_actions_after_draw(
        player, None  # 关键修改：不依赖 last_drawn_tile
    )
    # ... 处理所有动作类型，包括 WIN
```

**修改点 3**：新增 `_build_meld_decision_mask`

```python
def _build_meld_decision_mask(self, player, context, mask):
    """
    鸣牌后决策掩码（MELD_DECISION）
    - 可以杠、出牌
    - 不能胡、过
    """
    actions = ActionValidator(context).detect_available_actions_after_draw(
        player, None
    )
    # 处理杠和出牌，忽略 WIN、CHOW、PONG、KONG_EXPOSED
```

#### 2.3.3 ActionValidator 修复

**关键修改**：支持 `draw_tile=None` 参数

```python
def detect_available_actions_after_draw(
    self, current_player: PlayerData, draw_tile: Optional[int]
) -> List[MahjongAction]:
    """
    Args:
        draw_tile: 摸牌编码（如果为 None，表示不指定特定牌，仅基于手牌检测）
    """
    temp_hand = current_player.hand_tiles.copy()

    # 只在 draw_tile 不为 None 且不在手牌中时添加（避免重复添加）
    if draw_tile is not None and draw_tile not in temp_hand:
        temp_hand.append(draw_tile)

    # ... 后续逻辑
```

---

## 3. 实施细节

### 3.1 文件修改清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/mahjong_rl/core/constants.py` | 修改 | 添加 `GameStateType.MELD_DECISION` |
| `src/mahjong_rl/state_machine/states/meld_decision_state.py` | 新建 | 鸣牌后决策状态 |
| `src/mahjong_rl/state_machine/states/process_meld_state.py` | 修改 | 返回 `MELD_DECISION` |
| `src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py` | 修改 | 支持新状态，重构掩码构建 |
| `src/mahjong_rl/rules/wuhan_mahjong_rule_engine/action_validator.py` | 修改 | 支持 `draw_tile=None` |
| `src/mahjong_rl/state_machine/machine.py` | 修改 | 注册新状态 |

### 3.2 代码变更摘要

#### 变更 1：constants.py

```python
class GameStateType(Enum):
    ...
    PLAYER_DECISION = auto()  # 摸牌后决策（可杠/胡/出牌）
    MELD_DECISION = auto()    # 鸣牌后决策（可杠/出牌，不能胡）
    ...
```

#### 变更 2：process_meld_state.py

```python
# 修改前：
return GameStateType.PLAYER_DECISION

# 修改后：
return GameStateType.MELD_DECISION
```

#### 变更 3：wuhan_7p4l_observation_builder.py

```python
# 修改前的 build_action_mask
if current_state in [GameStateType.PLAYER_DECISION, GameStateType.DRAWING]:
    mask = self._build_draw_mask(player, context, mask)

# 修改后：
if current_state == GameStateType.MELD_DECISION:
    mask = self._build_meld_decision_mask(player, context, mask)
elif current_state in [GameStateType.PLAYER_DECISION, GameStateType.DRAWING]:
    mask = self._build_decision_mask(player, context, mask)
```

#### 变更 4：action_validator.py

```python
# 修改前：
def detect_available_actions_after_draw(self, current_player, draw_tile: int):
    temp_hand = current_player.hand_tiles.copy()
    temp_hand.append(draw_tile)  # 无条件添加

# 修改后：
def detect_available_actions_after_draw(
    self, current_player: PlayerData, draw_tile: Optional[int]
):
    temp_hand = current_player.hand_tiles.copy()
    if draw_tile is not None and draw_tile not in temp_hand:
        temp_hand.append(draw_tile)  # 条件添加，避免重复
```

---

## 4. 测试策略

### 4.1 单元测试

```python
# test_state_split.py

def test_meld_decision_rejects_pass():
    """测试 MELD_DECISION 拒绝 PASS 动作"""
    env = WuhanMahjongEnv()
    # ... 模拟鸣牌场景进入 MELD_DECISION 状态
    with pytest.raises(ValueError, match="PASS action is not allowed"):
        env.step((ActionType.PASS.value, 0))

def test_meld_decision_rejects_win():
    """测试 MELD_DECISION 拒绝 WIN 动作"""
    with pytest.raises(ValueError, match="WIN action is not allowed"):
        env.step((ActionType.WIN.value, 0))

def test_player_decision_allows_kong():
    """测试 PLAYER_DECISION 允许杠牌"""
    # 验证暗杠、补杠等动作可用

def test_meld_decision_allows_kong():
    """测试 MELD_DECISION 允许杠牌"""
    # 验证暗杠、补杠等动作可用
```

### 4.2 集成测试

```python
def test_full_game_with_state_split():
    """测试完整游戏流程"""
    env = WuhanMahjongEnv()
    obs, info = env.reset()

    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()

        # 验证 action_mask 形状正确
        assert obs['action_mask'].shape == (278,)

        # 验证在决策状态下不能 PASS
        if not terminated:
            assert obs['action_mask'][277] == 0  # PASS 位应该为 0

        # ... 执行动作
```

---

## 5. 设计优势

| 优势 | 说明 |
|------|------|
| **语义清晰** | 每个状态职责单一，PLAYER_DECISION 用于摸牌后，MELD_DECISION 用于鸣牌后 |
| **规则正确** | 明确禁止 PASS 动作，符合麻将规则 |
| **易于维护** | 状态逻辑分离，修改一个状态不影响另一个 |
| **向后兼容** | 状态转换流程平滑过渡，不影响其他部分 |
| **修复 bug** | 同时解决了重复添加牌、依赖 last_drawn_tile 等多个问题 |

---

## 6. 后续优化建议

1. **添加更多验证**：在状态机层面验证动作合法性，提前拒绝非法动作
2. **改进日志**：记录状态转换详情，便于调试
3. **性能优化**：缓存动作检测结果，避免重复计算
4. **文档更新**：更新 QUICK_START_GUIDE.md，反映新的状态设计
5. **测试覆盖**：增加边界情况测试（如开局、杠后、鸣牌后等场景）
