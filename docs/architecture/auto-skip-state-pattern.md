# Auto-Skip State Pattern

> **架构模式**: 状态机模式扩展
> **创建日期**: 2026-01-23
> **目的**: 允许状态声明"可以被自动跳过"，避免在 `enter()` 中包含状态转换逻辑

---

## 设计意图

### 问题场景

在状态机实现中，某些状态可能不需要用户交互即可自动转换到下一个状态。例如：
- 当所有玩家都只能选择 PASS 动作时，WAITING_RESPONSE 状态应该自动跳过
- 某些自动检查状态在没有条件满足时应该直接跳过

### 传统解决方案的问题

**错误的做法**：在 `enter()` 方法中执行状态转换逻辑

```python
# ❌ 错误示例
def enter(self, context: GameContext) -> None:
    # 初始化逻辑
    context.response_collector = ResponseCollector()

    # 问题：enter() 中包含了状态转换逻辑
    if len(context.active_responders) == 0:
        self._select_best_response(context)  # 触发状态转换
```

**问题**：
1. **违反单一职责原则 (SRP)**: `enter()` 混合了初始化和业务逻辑
2. **状态转换不一致**: 逻辑上应该转换但 `current_state_type` 没有更新
3. **难以测试**: 状态转换逻辑和初始化逻辑耦合

### Auto-Skip 模式解决方案

**正确的做法**：使用 `should_auto_skip()` 声明状态可跳过

```python
# ✅ 正确示例
def enter(self, context: GameContext) -> None:
    # 只负责初始化
    context.response_collector = ResponseCollector()
    # 不执行任何状态转换逻辑

def should_auto_skip(self, context: GameContext) -> bool:
    # 声明是否可以自动跳过
    return len(context.active_responders) == 0
```

---

## 实现方式

### 1. GameState 基类扩展

**文件**: `src/mahjong_rl/state_machine/base.py`

```python
class GameState(ABC):
    @abstractmethod
    def enter(self, context: GameContext) -> None:
        """进入状态"""
        pass

    def should_auto_skip(self, context: GameContext) -> bool:
        """
        检查是否应该自动跳过此状态

        默认实现：不跳过
        子类可以重写此方法以支持自动跳过逻辑

        设计意图：
        - 允许状态声明"可以被自动跳过"
        - 由状态机在 transition_to() 中统一处理自动转换
        - 避免在 enter() 中包含状态转换逻辑

        Args:
            context: 游戏上下文

        Returns:
            True 表示应该自动跳过（使用空动作执行 step）
            False 表示需要等待 agent 输入
        """
        return False
```

### 2. 状态机自动跳过检查

**文件**: `src/mahjong_rl/state_machine/machine.py`

```python
def transition_to(self, new_state_type: GameStateType, context: GameContext):
    """
    转换到新状态

    执行状态转换的完整流程：
    1. 退出当前状态
    2. 设置新状态
    3. 保存快照（不包括终端状态）
    4. 记录转换日志
    5. 进入新状态
    6. 【新增】检查是否需要自动跳过
    """
    # 退出当前状态
    if self.current_state:
        self.current_state.exit(context)

    # 设置新状态
    old_state_type = self.current_state_type
    self.current_state_type = new_state_type
    self.current_state = self.states[new_state_type]

    # 记录快照
    if new_state_type not in [GameStateType.WIN, GameStateType.FLOW_DRAW]:
        self._save_snapshot(context)

    # 记录状态转换日志
    if self.external_logger:
        self.external_logger.log_state_transition(old_state_type, new_state_type, context)

    # 进入新状态
    self.current_state.enter(context)

    # 【关键】检查是否需要自动跳过
    if self.current_state.should_auto_skip(context):
        self._auto_skip_state(context)
```

### 3. 自动跳过执行方法

**文件**: `src/mahjong_rl/state_machine/machine.py`

```python
def _auto_skip_state(self, context: GameContext) -> None:
    """
    自动跳过当前状态

    当状态的 should_auto_skip() 返回 True 时调用，
    使用 'auto' 动作执行 step()，触发状态转换。

    设计意图：
    - 统一处理自动跳过逻辑
    - 避免 enter() 中包含状态转换代码
    - 支持递归自动跳过（跳过后的状态也可能需要跳过）

    Args:
        context: 游戏上下文
    """
    if self.external_logger:
        self.external_logger.log_info(f"Auto-skipping state {self.current_state_type.name}")

    # 执行 step()，传入 'auto' 动作
    next_state_type = self.current_state.step(context, 'auto')

    # 如果需要转换状态
    if next_state_type is not None and next_state_type != self.current_state_type:
        self.transition_to(next_state_type, context)
```

---

## 使用示例

### WaitResponseState 实现

**文件**: `src/mahjong_rl/state_machine/states/wait_response_state.py`

```python
class WaitResponseState(GameState):
    def enter(self, context: GameContext) -> None:
        """
        进入等待响应状态

        职责：仅初始化，不执行任何状态转换逻辑
        """
        context.current_state = GameStateType.WAITING_RESPONSE

        # 初始化响应收集器
        context.response_collector = ResponseCollector()

        # 设置响应顺序
        if not context.response_order:
            context.setup_response_order(context.discard_player)

        # 构建响应者列表
        context.active_responders = []
        context.active_responder_idx = 0

        for responder_id in context.response_order:
            if not self._can_only_pass(context, responder_id):
                # 需要决策的玩家
                context.active_responders.append(responder_id)
            else:
                # 只能 PASS，自动添加响应
                context.response_collector.add_response(
                    responder_id,
                    ActionType.PASS,
                    ResponsePriority.PASS,
                    -1
                )

        # 如果有需要决策的玩家，为第一个生成观测
        if context.active_responders:
            first_responder = context.active_responders[0]
            context.current_player_idx = first_responder
            self.build_observation(context)

    def should_auto_skip(self, context: GameContext) -> bool:
        """
        检查是否应该自动跳过此状态

        如果所有玩家都只能 PASS，则自动跳过，无需等待任何输入。
        这允许状态机在 transition_to() 中自动推进到下一个状态。
        """
        return len(context.active_responders) == 0

    def step(self, context: GameContext, action: Union[MahjongAction, str]) -> GameStateType:
        """
        处理一个真实响应者的响应

        如果 active_responders 为空（所有玩家都只能 PASS），
        直接调用 _select_best_response() 返回下一个状态。
        """
        # 关键修复：如果 active_responders 为空（所有玩家都只能 PASS），
        # 直接选择最佳响应，无需等待任何输入
        if not context.active_responders:
            return self._select_best_response(context)

        # 检查是否还有待处理的响应者
        if context.active_responder_idx >= len(context.active_responders):
            # 所有真实响应者都已处理
            return self._select_best_response(context)

        # ... 后续逻辑 ...
```

---

## 设计原则符合性

| 原则 | 符合性 | 说明 |
|------|--------|------|
| **SRP** | ✅✅✅✅✅ | `enter()` 只负责初始化，`should_auto_skip()` 负责跳过判断 |
| **OCP** | ✅✅✅✅✅ | 通过扩展 `should_auto_skip()` 实现自动跳过，不需要修改状态机核心 |
| **LSP** | ✅✅✅✅✅ | 所有状态都可以重写 `should_auto_skip()`，不影响其他状态 |
| **ISP** | ✅✅✅✅✅ | `should_auto_skip()` 是可选接口，默认实现返回 False |
| **DIP** | ✅✅✅✅✅ | 状态机依赖 GameState 抽象，不关心具体实现 |
| **LoD** | ✅✅✅✅✅ | `transition_to()` 只调用当前状态的方法，不深入内部 |

---

## 执行流程

### 正常流程（不跳过）

```
1. 转换到 WAITING_RESPONSE
   ↓
2. transition_to(WAITING_RESPONSE, context)
   ↓
3. WaitResponseState.enter(context)
   ↓
4. active_responders = [1, 2]  # 有玩家可以响应
   ↓
5. should_auto_skip() → False
   ↓
6. 不跳过，等待玩家输入
```

### 自动跳过流程

```
1. 转换到 WAITING_RESPONSE
   ↓
2. transition_to(WAITING_RESPONSE, context)
   ↓
3. WaitResponseState.enter(context)
   ↓
4. active_responders = []  # 所有玩家只能 PASS
   ↓
5. should_auto_skip() → True
   ↓
6. _auto_skip_state(context)
   ↓
7. step(context, 'auto')
   ↓
8. _select_best_response(context) → DRAWING
   ↓
9. transition_to(DRAWING, context)
   ↓
10. DRAWING.enter(context)
   ↓
11. context.current_player_idx = 1  # 下一个玩家
```

---

## 扩展性

### 支持递归自动跳过

```python
def _auto_skip_state(self, context: GameContext) -> None:
    next_state_type = self.current_state.step(context, 'auto')
    if next_state_type is not None and next_state_type != self.current_state_type:
        # 递归调用 transition_to，如果新状态也需要跳过，会继续跳过
        self.transition_to(next_state_type, context)
```

**示例场景**：
- WAITING_RESPONSE → 自动跳过 → DRAWING → 自动跳过 → PLAYER_DECISION

### 其他状态可复用

任何状态都可以实现 `should_auto_skip()`：

```python
class MeldDecisionState(GameState):
    def should_auto_skip(self, context: GameContext) -> bool:
        # 如果没有可用的副露动作，自动跳过
        available_actions = self._get_available_actions(context)
        return len(available_actions) == 0

class RobKongState(GameState):
    def should_auto_skip(self, context: GameContext) -> bool:
        # 如果无人可以抢杠，自动跳过
        return not self._has_valid_rob_kong(context)
```

---

## 测试

### 单元测试

**文件**: `tests/integration/test_auto_skip_state.py`

```python
def test_wait_response_auto_skip_when_all_pass():
    """测试所有玩家都只能 PASS 时自动跳过"""
    context = GameContext()
    # 设置场景...

    state_machine = MahjongStateMachine(...)
    state_machine.transition_to(GameStateType.WAITING_RESPONSE, context)

    # 验证状态已自动跳过
    assert state_machine.current_state_type == GameStateType.DRAWING
    assert context.current_player_idx == 1  # 下一个玩家

def test_wait_response_no_skip_when_has_responders():
    """测试有玩家可以响应时不跳过"""
    context = GameContext()
    # 设置场景...

    state_machine = MahjongStateMachine(...)
    state_machine.transition_to(GameStateType.WAITING_RESPONSE, context)

    # 验证状态没有自动跳过
    assert state_machine.current_state_type == GameStateType.WAITING_RESPONSE
    assert len(context.active_responders) > 0
```

---

## 相关文档

- **实施计划**: `docs/plans/2026-01-23-auto-skip-state-pattern.md`
- **问题修复记录**: `docs/plans/2026-01-23-cli-interaction-fixes.md`
- **测试报告**: `TASK9_SUMMARY.md`
- **测试指南**: `TASK9_TESTING_GUIDE.md`

---

**文档状态**: ✅ 完成
**最后更新**: 2026-01-23
**下次审查**: 当状态机逻辑被修改时
