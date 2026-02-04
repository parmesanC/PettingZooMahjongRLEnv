# WaitRobKongState Auto-Skip Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 `WaitRobKongState` 实现自动跳过功能，当没有玩家可以抢杠和时自动跳过该阶段，与 `WaitResponseState` 保持一致的设计模式。

**Architecture:** 参考 `docs/architecture/auto-skip-state-pattern.md`，在 `WaitRobKongState` 中添加 `should_auto_skip()` 方法，并修改 `step()` 方法正确处理状态机的 `'auto'` 动作调用。

**Tech Stack:** Python 3.x, PettingZoo AECEnv, 现有状态机框架

---

## Context

### 问题背景

当前 `WaitRobKongState` 在没有玩家可以抢杠和时，仍然需要等待输入，无法自动跳过。而 `WaitResponseState` 已经实现了自动跳过功能（通过 `should_auto_skip()` 方法），两者设计不一致。

### 相关文件

**实现文件：**
- `src/mahjong_rl/state_machine/states/wait_rob_kong_state.py` - 需要修改
- `src/mahjong_rl/state_machine/states/wait_response_state.py` - 参考实现

**测试文件：**
- `tests/integration/test_rob_kong.py` - 现有测试
- `tests/integration/test_auto_skip_state.py` - 参考测试模式

**架构文档：**
- `docs/architecture/auto-skip-state-pattern.md` - 自动跳过模式设计文档

---

## Task 1: 添加 `should_auto_skip()` 方法

**Files:**
- Modify: `src/mahjong_rl/state_machine/states/wait_rob_kong_state.py`

**Step 1: 在文件末尾添加 `should_auto_skip()` 方法**

在 `wait_rob_kong_state.py` 文件的 `WaitRobKongState` 类中，`_check_rob_kong_result()` 方法之后添加以下代码：

```python
def should_auto_skip(self, context: GameContext) -> bool:
    """
    检查是否应该自动跳过此状态

    如果没有玩家可以抢杠和（active_responders 为空），则自动跳过。
    这允许状态机在 transition_to() 中自动推进到下一个状态。

    设计意图：
    - 避免在 enter() 中执行状态转换逻辑
    - 由状态机统一处理自动跳过
    - 保持 enter() 的单一职责（初始化）
    - 与 WaitResponseState 保持一致的设计模式

    Args:
        context: 游戏上下文

    Returns:
        True 如果没有玩家能抢杠和（应该自动跳过）
        False 如果有玩家需要决策
    """
    return len(context.active_responders) == 0
```

**位置参考：** 添加在 `_check_rob_kong_result()` 方法之后，`exit()` 方法之前。

---

## Task 2: 修改 `step()` 方法正确处理 `'auto'` 动作

**Files:**
- Modify: `src/mahjong_rl/state_machine/states/wait_rob_kong_state.py`

**Step 1: 修改 `step()` 方法开头部分**

将现有的 `step()` 方法开头：

```python
# 当前代码（需要修改的部分）
def step(self, context: GameContext, action: Union[MahjongAction, str]) -> GameStateType:
    # 如果没有玩家能抢杠，直接执行补杠
    if not context.active_responders:
        return self._check_rob_kong_result(context)

    # 获取当前响应者
    if context.active_responder_idx >= len(context.active_responders):
        # 所有玩家都已响应，检查结果
        return self._check_rob_kong_result(context)

    current_responder = context.active_responders[context.active_responder_idx]

    # 处理当前玩家的响应
    if action == 'auto':
        # 如果是自动模式，默认PASS
        response_action = MahjongAction(ActionType.PASS, -1)
```

**替换为：**

```python
def step(self, context: GameContext, action: Union[MahjongAction, str]) -> GameStateType:
    """
    收集一个玩家的抢杠和响应

    处理当前玩家的响应，移动到下一个响应者。
    当所有可抢杠和的玩家都响应后，决定下一个状态。

    Args:
        context: 游戏上下文
        action: 玩家响应动作（MahjongAction对象）或'auto'

    Returns:
        WIN 如果有玩家抢杠和成功
        DRAWING_AFTER_GONG 如果都PASS（执行补杠）

    Raises:
        ValueError: 如果动作类型不是 MahjongAction 或 'auto'
        ValueError: 如果动作类型不是 WIN 或 PASS
        ValueError: 如果玩家不在 active_responders 列表中却尝试抢杠
    """
    # 【新增】优先处理自动跳过场景
    # 当 should_auto_skip() 返回 True 时，状态机会调用 step(context, 'auto')
    if action == 'auto':
        if not context.active_responders:
            # 没有玩家能抢杠，直接执行补杠逻辑
            return self._check_rob_kong_result(context)
        # 有响应者时，不应该用 'auto' 调用
        # （正常流程由状态机在 enter 后检查 should_auto_skip）
        raise ValueError(
            f"Unexpected 'auto' action with active responders. "
            f"State machine should skip this state via should_auto_skip() "
            f"when active_responders is empty."
        )

    # 获取当前响应者
    if context.active_responder_idx >= len(context.active_responders):
        # 所有玩家都已响应，检查结果
        return self._check_rob_kong_result(context)

    current_responder = context.active_responders[context.active_responder_idx]

    # 处理当前玩家的响应
    if not isinstance(action, MahjongAction):
        raise ValueError(
            f"WaitRobKongState expects MahjongAction or 'auto', got {type(action).__name__}"
        )

    response_action = action
```

**Step 2: 移除后续重复的 `action == 'auto'` 检查**

由于在方法开头已经处理了 `'auto'` 动作，后续代码中不再需要这个检查。确保后续代码直接使用 `response_action = action`（已在 Step 1 中完成）。

---

## Task 3: 编写单元测试

**Files:**
- Create: `tests/integration/test_wait_rob_kong_auto_skip.py`

**Step 1: 创建测试文件**

创建新文件 `tests/integration/test_wait_rob_kong_auto_skip.py`：

```python
"""测试 WaitRobKongState 自动跳过功能

验证当没有玩家可以抢杠和时，状态能够自动跳过。
"""

import pytest
from collections import deque

from mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder
from mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
from mahjong_rl.state_machine import MahjongStateMachine
from mahjong_rl.state_machine.states.wait_rob_kong_state import WaitRobKongState
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData, Meld
from src.mahjong_rl.core.constants import GameStateType, ActionType
from src.mahjong_rl.core.mahjong_action import MahjongAction


class TestWaitRobKongAutoSkip:
    """测试 WaitRobKongState 自动跳过功能"""

    def test_should_auto_skip_when_no_responders(self):
        """测试没有玩家能抢杠时，should_auto_skip 返回 True"""
        context = GameContext()
        context.current_state = GameStateType.WAIT_ROB_KONG
        context.wall = deque([i for i in range(34) for _ in range(4)])
        context.lazy_tile = 34
        context.skin_tile = [-1, -1]
        context.red_dragon = 31

        rule_engine = Wuhan7P4LRuleEngine(context)
        observation_builder = Wuhan7P4LObservationBuilder(context)
        state = WaitRobKongState(rule_engine, observation_builder)

        # 设置场景：玩家0补杠，但没有玩家能抢杠
        context.current_player_idx = 0
        context.kong_player_idx = 0
        context.rob_kong_tile = 0  # 1万

        # 玩家0有碰牌准备补杠
        player0 = context.players[0]
        player0.hand_tiles = [1, 2, 3, 9, 10, 11, 18, 19, 20]
        player0.melds = [Meld(
            action_type=MahjongAction(ActionType.PONG, 0),
            tiles=[0, 0, 0],
            from_player=1
        )]
        player0.special_gangs = [0, 0, 0]

        # 其他玩家手牌（无法胡牌）
        for i in [1, 2, 3]:
            player = context.players[i]
            player.hand_tiles = [4, 5, 6, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27]
            player.melds = []
            player.special_gangs = [0, 0, 0]

        # 调用 enter() 设置 active_responders
        state.enter(context)

        # 验证 active_responders 为空
        assert len(context.active_responders) == 0

        # 验证 should_auto_skip 返回 True
        assert state.should_auto_skip(context) is True

    def test_should_not_auto_skip_when_has_responders(self):
        """测试有玩家能抢杠时，should_auto_skip 返回 False"""
        context = GameContext()
        context.current_state = GameStateType.WAIT_ROB_KONG
        context.wall = deque([i for i in range(34) for _ in range(4)])
        context.lazy_tile = 34
        context.skin_tile = [-1, -1]
        context.red_dragon = 31

        rule_engine = Wuhan7P4LRuleEngine(context)
        observation_builder = Wuhan7P4LObservationBuilder(context)
        state = WaitRobKongState(rule_engine, observation_builder)

        # 设置场景：玩家0补杠，玩家1能抢杠
        context.current_player_idx = 0
        context.kong_player_idx = 0
        context.rob_kong_tile = 0  # 1万

        # 玩家0有碰牌准备补杠
        player0 = context.players[0]
        player0.hand_tiles = [1, 2, 3, 9, 10, 11, 18, 19, 20]
        player0.melds = [Meld(
            action_type=MahjongAction(ActionType.PONG, 0),
            tiles=[0, 0, 0],
            from_player=1
        )]
        player0.special_gangs = [0, 0, 0]

        # 玩家1可以抢杠和（手牌+1万能胡）
        player1 = context.players[1]
        player1.hand_tiles = [1, 1, 2, 2, 3, 3, 9, 10, 11, 18]  # 有将牌，加1万可胡
        player1.melds = [Meld(
            action_type=MahjongAction(ActionType.PONG, 21),
            tiles=[21, 21, 21],
            from_player=2
        )]
        player1.special_gangs = [0, 0, 0]

        # 玩家2、3普通手牌
        for i in [2, 3]:
            player = context.players[i]
            player.hand_tiles = [4, 5, 6, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27]
            player.melds = []
            player.special_gangs = [0, 0, 0]

        # 调用 enter() 设置 active_responders
        state.enter(context)

        # 验证 active_responders 不为空
        assert len(context.active_responders) > 0

        # 验证 should_auto_skip 返回 False
        assert state.should_auto_skip(context) is False

    def test_step_with_auto_action_when_no_responders(self):
        """测试 step() 方法正确处理 'auto' 动作（没有响应者时）"""
        context = GameContext()
        context.current_state = GameStateType.WAIT_ROB_KONG
        context.wall = deque([i for i in range(34) for _ in range(4)])
        context.lazy_tile = 34
        context.skin_tile = [-1, -1]
        context.red_dragon = 31

        rule_engine = Wuhan7P4LRuleEngine(context)
        observation_builder = Wuhan7P4LObservationBuilder(context)
        state = WaitRobKongState(rule_engine, observation_builder)

        # 设置场景：玩家0补杠，但没有玩家能抢杠
        context.current_player_idx = 0
        context.kong_player_idx = 0
        context.rob_kong_tile = 0

        player0 = context.players[0]
        player0.hand_tiles = [1, 2, 3, 9, 10, 11, 18, 19, 20]
        player0.melds = [Meld(
            action_type=MahjongAction(ActionType.PONG, 0),
            tiles=[0, 0, 0],
            from_player=1
        )]
        player0.special_gangs = [0, 0, 0]

        for i in [1, 2, 3]:
            player = context.players[i]
            player.hand_tiles = [4, 5, 6, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27]
            player.melds = []
            player.special_gangs = [0, 0, 0]

        state.enter(context)

        # 调用 step(context, 'auto') 应该返回 DRAWING_AFTER_GONG
        next_state = state.step(context, 'auto')
        assert next_state == GameStateType.DRAWING_AFTER_GONG

        # 验证补杠已执行
        assert len(player0.melds) == 1
        assert player0.melds[0].action_type.action_type == ActionType.KONG_SUPPLEMENT

    def test_step_with_auto_action_raises_when_has_responders(self):
        """测试 step() 方法在有响应者时用 'auto' 调用会抛出异常"""
        context = GameContext()
        context.current_state = GameStateType.WAIT_ROB_KONG
        context.wall = deque([i for i in range(34) for _ in range(4)])
        context.lazy_tile = 34
        context.skin_tile = [-1, -1]
        context.red_dragon = 31

        rule_engine = Wuhan7P4LRuleEngine(context)
        observation_builder = Wuhan7P4LObservationBuilder(context)
        state = WaitRobKongState(rule_engine, observation_builder)

        # 设置场景：有玩家能抢杠
        context.current_player_idx = 0
        context.kong_player_idx = 0
        context.rob_kong_tile = 0

        player0 = context.players[0]
        player0.hand_tiles = [1, 2, 3, 9, 10, 11, 18, 19, 20]
        player0.melds = [Meld(
            action_type=MahjongAction(ActionType.PONG, 0),
            tiles=[0, 0, 0],
            from_player=1
        )]
        player0.special_gangs = [0, 0, 0]

        player1 = context.players[1]
        player1.hand_tiles = [1, 1, 2, 2, 3, 3, 9, 10, 11, 18]
        player1.melds = [Meld(
            action_type=MahjongAction(ActionType.PONG, 21),
            tiles=[21, 21, 21],
            from_player=2
        )]
        player1.special_gangs = [0, 0, 0]

        for i in [2, 3]:
            player = context.players[i]
            player.hand_tiles = [4, 5, 6, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27]
            player.melds = []
            player.special_gangs = [0, 0, 0]

        state.enter(context)

        # 调用 step(context, 'auto') 应该抛出 ValueError
        with pytest.raises(ValueError, match="Unexpected 'auto' action"):
            state.step(context, 'auto')

    def test_state_machine_auto_skips_when_no_responders(self):
        """测试状态机在进入 WAIT_ROB_KONG 后自动跳过（集成测试）"""
        context = GameContext()
        context.wall = deque([i for i in range(34) for _ in range(4)])
        context.lazy_tile = 34
        context.skin_tile = [-1, -1]
        context.red_dragon = 31

        rule_engine = Wuhan7P4LRuleEngine(context)
        observation_builder = Wuhan7P4LObservationBuilder(context)
        state_machine = MahjongStateMachine(
            rule_engine=rule_engine,
            observation_builder=observation_builder,
            enable_logging=False
        )
        state_machine.set_context(context)

        # 设置场景：玩家0补杠，但没有玩家能抢杠
        context.current_player_idx = 0
        context.kong_player_idx = 0
        context.rob_kong_tile = 0

        player0 = context.players[0]
        player0.hand_tiles = [1, 2, 3, 9, 10, 11, 18, 19, 20]
        player0.melds = [Meld(
            action_type=MahjongAction(ActionType.PONG, 0),
            tiles=[0, 0, 0],
            from_player=1
        )]
        player0.special_gangs = [0, 0, 0]

        for i in [1, 2, 3]:
            player = context.players[i]
            player.hand_tiles = [4, 5, 6, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27]
            player.melds = []
            player.special_gangs = [0, 0, 0]

        # 转换到 WAIT_ROB_KONG
        state_machine.transition_to(GameStateType.WAIT_ROB_KONG, context)

        # 验证状态已自动跳过到 DRAWING_AFTER_GONG
        assert state_machine.current_state_type == GameStateType.DRAWING_AFTER_GONG
        assert context.is_kong_draw is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Step 2: 运行测试验证失败（预期）**

```bash
cd D:\DATA\Python_Project\Code\PettingZooRLENVMahjong
D:\DATA\Development\Anaconda\condabin\conda.bat activate PettingZooRLMahjong
pytest tests/integration/test_wait_rob_kong_auto_skip.py -v
```

**预期结果：** 部分测试失败（因为 `should_auto_skip()` 方法还不存在）

---

## Task 4: 运行现有测试确保兼容性

**Step 1: 运行现有的 rob kong 测试**

```bash
pytest tests/integration/test_rob_kong.py -v
pytest tests/integration/test_rob_kong_full_transition.py -v
pytest tests/integration/test_rob_kong_scenario.py -v
```

**预期结果：** 所有测试通过（修改应该向后兼容）

---

## Task 5: 提交实现

**Step 1: 提交代码**

```bash
git add src/mahjong_rl/state_machine/states/wait_rob_kong_state.py
git add tests/integration/test_wait_rob_kong_auto_skip.py
git commit -m "feat(wait_rob_kong): add auto-skip support when no players can rob kong

- Add should_auto_skip() method to WaitRobKongState
- Modify step() to properly handle 'auto' action from state machine
- Maintain consistency with WaitResponseState auto-skip pattern
- Add comprehensive unit tests for auto-skip behavior

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Summary

**改动文件：**
1. `src/mahjong_rl/state_machine/states/wait_rob_kong_state.py` - 添加 `should_auto_skip()` 方法，修改 `step()` 方法
2. `tests/integration/test_wait_rob_kong_auto_skip.py` - 新建测试文件

**关键改动：**
- `should_auto_skip()`: 返回 `len(context.active_responders) == 0`
- `step()`: 优先处理 `'auto'` 动作，当没有响应者时直接调用 `_check_rob_kong_result()`

**设计原则：**
- 与 `WaitResponseState` 保持一致
- 遵循 `auto-skip-state-pattern` 架构模式
- 单一职责：`enter()` 只初始化，`should_auto_skip()` 判断是否跳过
