# Auto-Skip State Pattern Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 深入完整地修复 CLI 交互问题：当所有玩家都只能 PASS 动作时，系统仍然要求手动输入

**Architecture:** 引入"自动跳过状态"模式（Null Action Pattern），让状态机支持状态的自动跳过。通过在 `GameState` 基类中添加 `should_auto_skip()` 方法，允许状态声明可以自动跳过，由状态机在 `transition_to()` 中统一处理自动转换。

**Tech Stack:** Python 3.x, PettingZoo AECEnv, 状态机模式

---

## 问题背景

### 当前问题
当所有玩家都只能选择 PASS 动作时：
1. `WaitResponseState.enter()` 调用 `_select_best_response()` 返回 `DRAWING`
2. 但状态机没有真正转换（因为 `enter()` 内部无法调用 `transition_to()`）
3. `current_state_type` 仍然是 `WAITING_RESPONSE`
4. env.step() 的自动推进循环检测到 `WAITING_RESPONSE`，执行 break
5. agent_selection 没有更新，ManualController 请求旧玩家输入

### 根本原因
- **职责混乱**: `enter()` 混合了初始化和业务逻辑（判断是否应该跳过）
- **状态转换不一致**: 逻辑上应该转换但 `current_state_type` 没有更新
- **缺乏自动跳过机制**: 没有统一的"可跳过状态"处理机制

---

## Task 1: 在 GameState 基类中添加 should_auto_skip() 方法

**Files:**
- Modify: `src/mahjong_rl/state_machine/base.py:60-64`

**Step 1: 添加 should_auto_skip() 方法**

在 `GameState` 基类的 `build_observation()` 方法后添加新方法：

```python
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

**Step 2: 验证语法**

运行: `python -m py_compile src/mahjong_rl/state_machine/base.py`
预期: 无语法错误

**Step 3: 运行现有测试确保没有破坏**

运行: `python -m pytest tests/ -v -k "state" 2>/dev/null || echo "No state tests found"`
预期: 现有测试通过

**Step 4: Commit**

```bash
git add src/mahjong_rl/state_machine/base.py
git commit -m "feat(state): add should_auto_skip() method to GameState base class

- Add virtual method for auto-skip state pattern
- Default implementation returns False (no auto-skip)
- Subclasses can override to support automatic state transitions"
```

---

## Task 2: 修改 WaitResponseState.enter() - 移除业务逻辑

**Files:**
- Modify: `src/mahjong_rl/state_machine/states/wait_response_state.py:48-93`

**Step 1: 简化 enter() 方法**

替换整个 `enter()` 方法为简化版本：

```python
def enter(self, context: GameContext) -> None:
    """
    进入等待响应状态

    职责：仅初始化，不执行任何状态转换逻辑
    - 初始化响应收集器
    - 构建响应者列表（区分需要决策和只能 PASS 的玩家）
    - 为需要决策的玩家生成观测

    设计原则：
    - SRP: enter() 只负责初始化
    - 状态转换逻辑由 should_auto_skip() 和 transition_to() 处理
    """
    context.current_state = GameStateType.WAITING_RESPONSE

    # 初始化响应收集器
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    from mahjong_rl.state_machine.ResponseCollector import ResponseCollector
    context.response_collector = ResponseCollector()

    # 确保响应顺序已设置（自动排除出牌者）
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
                -1  # PASS 无参数
            )

    # 关键修改：如果所有玩家都只能 PASS，不在这里调用 _select_best_response
    # 而是通过 should_auto_skip() 让状态机处理自动跳过

    # 如果有需要决策的玩家，为第一个生成观测
    if context.active_responders:
        first_responder = context.active_responders[0]
        context.current_player_idx = first_responder
        self.build_observation(context)
```

**Step 2: 验证语法**

运行: `python -m py_compile src/mahjong_rl/state_machine/states/wait_response_state.py`
预期: 无语法错误

**Step 3: Commit**

```bash
git add src/mahjong_rl/state_machine/states/wait_response_state.py
git commit -m "refactor(wait_response): simplify enter() method - remove business logic

- Remove _select_best_response() call from enter()
- enter() now only handles initialization (SRP)
- State transition logic will be handled by should_auto_skip()"
```

---

## Task 3: 在 WaitResponseState 中实现 should_auto_skip()

**Files:**
- Modify: `src/mahjong_rl/state_machine/states/wait_response_state.py` (在 `exit()` 方法后添加)

**Step 1: 添加 should_auto_skip() 方法**

在 `_can_only_pass()` 方法后添加：

```python
def should_auto_skip(self, context: GameContext) -> bool:
    """
    检查是否应该自动跳过此状态

    如果所有玩家都只能 PASS，则自动跳过，无需等待任何输入。
    这允许状态机在 transition_to() 中自动推进到下一个状态。

    设计意图：
    - 避免在 enter() 中执行状态转换逻辑
    - 由状态机统一处理自动跳过
    - 保持 enter() 的单一职责（初始化）

    Args:
        context: 游戏上下文

    Returns:
        True 如果所有玩家都只能 PASS（应该自动跳过）
        False 如果有玩家需要决策
    """
    return len(context.active_responders) == 0
```

**Step 2: 验证语法**

运行: `python -m py_compile src/mahjong_rl/state_machine/states/wait_response_state.py`
预期: 无语法错误

**Step 3: Commit**

```bash
git add src/mahjong_rl/state_machine/states/wait_response_state.py
git commit -m "feat(wait_response): implement should_auto_skip() method

- Return True when all players can only PASS
- Allows state machine to auto-skip WAITING_RESPONSE state
- Fixes issue where manual input was required even with only PASS available"
```

---

## Task 4: 修改 WaitResponseState.step() - 处理自动跳过情况

**Files:**
- Modify: `src/mahjong_rl/state_machine/states/wait_response_state.py:95-147`

**Step 1: 修改 step() 方法开头**

在 `step()` 方法开头添加自动跳过检查：

```python
def step(self, context: GameContext, action: Union[MahjongAction, str]) -> GameStateType:
    """
    处理一个真实响应者的响应

    如果 active_responders 为空（所有玩家都只能 PASS），
    直接调用 _select_best_response() 返回下一个状态。

    Args:
        context: 游戏上下文
        action: 玩家响应动作或'auto'

    Returns:
        WAITING_RESPONSE (继续) 或下一个状态
    """
    # 关键修复：如果 active_responders 为空（所有玩家都只能 PASS），
    # 直接选择最佳响应，无需等待任何输入
    if not context.active_responders:
        return self._select_best_response(context)

    # 检查是否还有待处理的响应者
    if context.active_responder_idx >= len(context.active_responders):
        # 所有真实响应者都已处理
        return self._select_best_response(context)

    # ... 后续代码保持不变 ...
```

**Step 2: 验证语法**

运行: `python -m py_compile src/mahjong_rl/state_machine/states/wait_response_state.py`
预期: 无语法错误

**Step 3: Commit**

```bash
git add src/mahjong_rl/state_machine/states/wait_response_state.py
git commit -m "fix(wait_response): handle auto-skip case in step() method

- Check if active_responders is empty at start of step()
- If empty, directly call _select_best_response()
- This allows the state to be auto-skipped by state machine"
```

---

## Task 5: 修改 MahjongStateMachine.transition_to() - 添加自动跳过逻辑

**Files:**
- Modify: `src/mahjong_rl/state_machine/machine.py:213-249`

**Step 1: 修改 transition_to() 方法**

在 `enter()` 调用后添加自动跳过检查：

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

    Args:
        new_state_type: 目标状态类型
        context: 游戏上下文
    """
    # 退出当前状态
    if self.current_state:
        self.current_state.exit(context)

    # 设置新状态（先设置，确保快照中的state_type正确）
    old_state_type = self.current_state_type
    self.current_state_type = new_state_type
    self.current_state = self.states[new_state_type]

    # 记录快照（不包括终端状态）
    if new_state_type not in [GameStateType.WIN, GameStateType.FLOW_DRAW]:
        self._save_snapshot(context)

    # 记录状态转换日志（外部 logger 和内部 logger）
    if self.external_logger:
        self.external_logger.log_state_transition(old_state_type, new_state_type, context)

    if self.internal_logger:
        self.internal_logger.log_transition(old_state_type, new_state_type, context)

    # 进入新状态
    self.current_state.enter(context)

    # 【新增】检查是否需要自动跳过
    if self.current_state.should_auto_skip(context):
        self._auto_skip_state(context)
```

**Step 2: 验证语法**

运行: `python -m py_compile src/mahjong_rl/state_machine/machine.py`
预期: 无语法错误

**Step 3: Commit**

```bash
git add src/mahjong_rl/state_machine/machine.py
git commit -m "feat(state-machine): add auto-skip check in transition_to()

- Check should_auto_skip() after entering new state
- Allows states to declare they can be auto-skipped
- Delegates auto-skip logic to _auto_skip_state() method"
```

---

## Task 6: 添加 MahjongStateMachine._auto_skip_state() 方法

**Files:**
- Modify: `src/mahjong_rl/state_machine/machine.py` (在 `transition_to()` 方法后添加)

**Step 1: 添加 _auto_skip_state() 方法**

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

**Step 2: 验证语法**

运行: `python -m py_compile src/mahjong_rl/state_machine/machine.py`
预期: 无语法错误

**Step 3: Commit**

```bash
git add src/mahjong_rl/state_machine/machine.py
git commit -m "feat(state-machine): add _auto_skip_state() method

- Execute step() with 'auto' action to trigger state transition
- Support recursive auto-skip (skipped state may also need skip)
- Centralizes auto-skip logic in state machine"
```

---

## Task 7: 更新 logger 接口 - 添加 log_info 方法（如果不存在）

**Files:**
- Check: `src/mahjong_rl/logging/base.py`
- Modify: `src/mahjong_rl/logging/base.py` (如果需要)

**Step 1: 检查 ILogger 接口是否有 log_info 方法**

运行: `grep -n "def log_info" src/mahjong_rl/logging/base.py`

**Step 2a: 如果存在，跳过此任务**

**Step 2b: 如果不存在，添加 log_info 方法**

在 ILogger 类中添加：

```python
@abstractmethod
def log_info(self, message: str) -> None:
    """
    记录信息日志

    Args:
        message: 日志消息
    """
    pass
```

**Step 3: 在所有实现类中实现 log_info()**

对于每个 ILogger 实现类，添加：

```python
def log_info(self, message: str) -> None:
    """记录信息日志"""
    self.log(LogLevel.INFO, message)
```

**Step 4: Commit**

```bash
git add src/mahjong_rl/logging/base.py
git commit -m "feat(logging): add log_info() method to ILogger interface

- Add abstract method for info-level logging
- Implement in all logger classes
- Used by state machine auto-skip logging"
```

---

## Task 8: 编写集成测试

**Files:**
- Create: `tests/integration/test_auto_skip_state.py`

**Step 1: 创建测试文件**

```python
"""
测试自动跳过状态模式

验证当所有玩家都只能 PASS 时，WAITING_RESPONSE 状态能够自动跳过
"""

import pytest
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import GameStateType
from src.mahjong_rl.state_machine.machine import MahjongStateMachine
from src.mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder


def test_wait_response_auto_skip_when_all_pass():
    """
    测试场景：所有玩家都只能 PASS

    预期行为：
    1. 进入 WAITING_RESPONSE 状态
    2. should_auto_skip() 返回 True
    3. 状态自动转换到 DRAWING
    4. agent_selection 正确更新
    """
    # 创建测试环境
    context = GameContext()
    context.players = [PlayerData(player_id=i) for i in range(4)]
    context.current_player_idx = 0

    # 设置弃牌（一张特殊牌，使得无人能吃碰杠胡）
    context.last_discarded_tile = 34  # 红中
    context.discard_player = 0
    context.setup_response_order(0)

    # 创建状态机
    rule_engine = Wuhan7P4LRuleEngine(context)
    observation_builder = Wuhan7P4LObservationBuilder(context)
    state_machine = MahjongStateMachine(
        rule_engine=rule_engine,
        observation_builder=observation_builder,
        logger=None,
        enable_logging=False
    )
    state_machine.set_context(context)

    # 转换到 WAITING_RESPONSE 状态
    state_machine.transition_to(GameStateType.WAITING_RESPONSE, context)

    # 验证状态已自动跳过
    # 由于所有玩家都只能 PASS（红中不能吃碰杠），
    # 状态应该自动转换到 DRAWING
    assert state_machine.current_state_type == GameStateType.DRAWING, \
        f"Expected DRAWING state, got {state_machine.current_state_type}"

    # 验证 agent_selection 已更新
    assert context.current_player_idx == 1, \
        f"Expected player_1, got player_{context.current_player_idx}"


def test_wait_response_no_skip_when_has_responders():
    """
    测试场景：有玩家可以响应（非 PASS）

    预期行为：
    1. 进入 WAITING_RESPONSE 状态
    2. should_auto_skip() 返回 False
    3. 状态保持在 WAITING_RESPONSE
    4. active_responders 不为空
    """
    # 创建测试环境
    context = GameContext()
    context.players = [PlayerData(player_id=i) for i in range(4)]

    # 给玩家1一张可以碰的牌
    from src.mahjong_rl.core.constants import Tiles
    context.players[1].hand_tiles = [Tiles.TILE_1WAN, Tiles.TILE_1WAN, Tiles.TILE_2WAN]
    context.current_player_idx = 0

    # 设置弃牌（一张万子牌）
    context.last_discarded_tile = Tiles.TILE_1WAN
    context.discard_player = 0
    context.setup_response_order(0)

    # 创建状态机
    rule_engine = Wuhan7P4LRuleEngine(context)
    observation_builder = Wuhan7P4LObservationBuilder(context)
    state_machine = MahjongStateMachine(
        rule_engine=rule_engine,
        observation_builder=observation_builder,
        logger=None,
        enable_logging=False
    )
    state_machine.set_context(context)

    # 转换到 WAITING_RESPONSE 状态
    state_machine.transition_to(GameStateType.WAITING_RESPONSE, context)

    # 验证状态没有自动跳过
    assert state_machine.current_state_type == GameStateType.WAITING_RESPONSE, \
        f"Expected WAITING_RESPONSE state, got {state_machine.current_state_type}"

    # 验证 active_responders 不为空
    assert len(context.active_responders) > 0, \
        "Expected active_responders to be non-empty"

    # 验证当前玩家是第一个响应者
    assert context.current_player_idx in context.active_responders, \
        f"Expected current_player to be in active_responders"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Step 2: 运行测试**

运行: `python tests/integration/test_auto_skip_state.py`
预期: 测试通过

**Step 3: Commit**

```bash
git add tests/integration/test_auto_skip_state.py
git commit -m "test: add integration tests for auto-skip state pattern

- Test auto-skip when all players can only PASS
- Test no auto-skip when players have valid responses
- Validates agent_selection update after auto-skip"
```

---

## Task 9: 手动测试验证

**Files:**
- Test: 启动游戏并验证修复

**Step 1: 启动游戏**

运行: `python play_mahjong.py --mode four_human --renderer cli`

**Step 2: 测试场景**

1. 开始游戏
2. 观察是否有"只有 PASS 仍要输入"的情况
3. 预期：所有玩家自动 PASS 时，应该直接跳过，不要求输入

**Step 3: 验证 agent_selection 更新**

在游戏中打印 `env.agent_selection`，验证在自动跳过后正确更新

**Step 4: 如果测试通过，继续；如果有问题，调试修复**

---

## Task 10: 更新文档

**Files:**
- Modify: `docs/plans/2026-01-23-cli-interaction-fixes.md`
- Create: `docs/architecture/auto-skip-state-pattern.md`

**Step 1: 更新问题修复记录**

在 `docs/plans/2026-01-23-cli-interaction-fixes.md` 中更新问题1的解决方案：

```markdown
### 2.3 解决方案（修订版）

**修改文件**:
- `src/mahjong_rl/state_machine/base.py` - 添加 `should_auto_skip()` 方法
- `src/mahjong_rl/state_machine/states/wait_response_state.py` - 实现自动跳过逻辑
- `src/mahjong_rl/state_machine/machine.py` - 添加自动跳过处理

**核心设计**: 引入"自动跳过状态"模式（Null Action Pattern）

1. **GameState.should_auto_skip()** - 声明状态是否可以被自动跳过
2. **transition_to() 自动检查** - 进入状态后检查是否需要跳过
3. **_auto_skip_state()** - 统一处理自动跳过逻辑
```

**Step 2: 创建架构文档**

创建 `docs/architecture/auto-skip-state-pattern.md`:

```markdown
# Auto-Skip State Pattern

## 设计意图

允许状态声明"可以被自动跳过"，避免在 `enter()` 中包含状态转换逻辑。

## 实现方式

1. **GameState.should_auto_skip(context)** - 返回 True/False
2. **MahjongStateMachine.transition_to()** - 调用后检查是否跳过
3. **_auto_skip_state()** - 使用 'auto' 动作执行 step()

## 使用示例

```python
class WaitResponseState(GameState):
    def should_auto_skip(self, context: GameContext) -> bool:
        return len(context.active_responders) == 0
```

## 设计原则

- SRP: enter() 只负责初始化
- OCP: 通过扩展 should_auto_skip() 实现自动跳过
- DIP: 状态机依赖 GameState 抽象
```

**Step 3: Commit**

```bash
git add docs/plans/2026-01-23-cli-interaction-fixes.md
git add docs/architecture/auto-skip-state-pattern.md
git commit -m "docs: update CLI interaction fixes documentation

- Revise Problem 1 solution with auto-skip pattern
- Add architecture documentation for auto-skip state pattern"
```

---

## Task 11: 清理旧的修复代码

**Files:**
- Modify: `example_mahjong_env.py:381-403`

**Step 1: 删除旧的修复代码**

删除 `example_mahjong_env.py` 中的 `enter_auto_converted` 检测逻辑：

```python
# 删除这些行
# 记录执行前的状态（用于检测 enter() 中的自动转换）
state_before_step = self.state_machine.current_state_type

# ...

# 检测 enter() 中的自动转换（修复问题1）
state_after_step = self.state_machine.current_state_type
enter_auto_converted = (state_before_step == GameStateType.WAITING_RESPONSE and
                        state_after_step != GameStateType.WAITING_RESPONSE and
                        next_state_type is not None)
```

**Step 2: 简化 step() 方法**

```python
def step(self, action):
    # ... 前面的逻辑保持不变 ...

    # 执行状态机step
    try:
        next_state_type = self.state_machine.step(self.context, mahjong_action)
    except Exception as e:
        # ... 错误处理 ...

    # 状态转换后更新 agent_selection
    if not self.state_machine.is_terminal():
        self.agent_selection = self.state_machine.get_current_agent()

    # 自动推进循环
    while not self.state_machine.is_terminal():
        current_state = self.state_machine.current_state_type

        # 需要agent动作的四个状态 - 停止自动推进
        if current_state in [
            GameStateType.PLAYER_DECISION,
            GameStateType.MELD_DECISION,
            GameStateType.WAITING_RESPONSE,
            GameStateType.WAIT_ROB_KONG
        ]:
            break

        # 其他状态都是自动状态，使用'auto'推进
        try:
            next_state_type = self.state_machine.step(self.context, 'auto')
            if next_state_type is None:
                break
        except Exception as e:
            print(f"自动推进错误: {e}")
            break

    # ... 后续逻辑保持不变 ...
```

**Step 3: Commit**

```bash
git add example_mahjong_env.py
git commit -m "refactor(env): remove old auto-skip detection code

- Remove enter_auto_converted detection logic
- Simplified step() method
- Auto-skip is now handled by state machine pattern"
```

---

## Task 12: 最终验证和测试

**Files:**
- Test: 完整的游戏流程测试

**Step 1: 运行所有测试**

运行: `python -m pytest tests/ -v`
预期: 所有测试通过

**Step 2: 手动端到端测试**

运行: `python play_mahjong.py --mode four_human --renderer cli`

验证：
1. 游戏正常运行
2. 所有玩家自动 PASS 时直接跳过
3. agent_selection 正确更新
4. 无错误日志

**Step 3: 性能测试**

运行多局游戏，确保性能没有下降

**Step 4: 最终 Commit**

```bash
git commit --allow-empty -m "chore: complete auto-skip state pattern implementation

- All tests passing
- Manual verification completed
- Documentation updated
- Ready for merge"
```

---

## 设计原则符合性

| 原则 | 符合性 | 说明 |
|------|--------|------|
| **SRP** | ✅ | enter() 只负责初始化，should_auto_skip() 负责跳过判断 |
| **OCP** | ✅ | 通过扩展 should_auto_skip() 实现自动跳过，不需要修改状态机核心 |
| **LSP** | ✅ | 所有状态都可以重写 should_auto_skip()，不影响其他状态 |
| **ISP** | ✅ | should_auto_skip() 是可选接口，默认实现返回 False |
| **DIP** | ✅ | 状态机依赖 GameState 抽象，不关心具体实现 |
| **LoD** | ✅ | transition_to() 只调用当前状态的方法，不深入内部 |

## 参考文档

- `docs/plans/2026-01-23-cli-interaction-fixes.md` - 原问题分析
- `docs/architecture/auto-skip-state-pattern.md` - 架构文档
- PettingZoo AECEnv 规范
