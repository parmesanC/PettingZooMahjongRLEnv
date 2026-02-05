# 场景测试日志增强实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 增强场景测试框架的日志输出，支持每步打印手牌、动作合法性检测（红色标记非法动作）、状态转换跟踪。

**架构：** 在 `TestExecutor` 中添加格式化、打印和验证方法，复用现有的 `TileTextVisualizer` 实现牌名显示。

**技术栈：**
- Python 3.x
- ANSI 转义码实现颜色输出
- 现有 `TileTextVisualizer` 类（已实现）

---

## Task 1: 添加常量和导入

**文件：**
- Modify: `tests/scenario/executor.py:1-12`

**Step 1: 添加导入和颜色常量**

在文件顶部添加：

```python
from typing import Optional, List, Tuple
import datetime
from src.mahjong_rl.core.constants import Tiles
from src.mahjong_rl.visualization.TileVisualization import TileTextVisualizer

# ANSI 颜色代码
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
```

**Step 2: 验证语法正确**

Run: `python -m py_compile tests/scenario/executor.py`
Expected: 无错误

**Step 3: 提交**

```bash
git add tests/scenario/executor.py
git commit -m "feat(test): add imports and color constants for logging enhancement"
```

---

## Task 2: 修改 TestExecutor.__init__ 添加配置参数

**文件：**
- Modify: `tests/scenario/executor.py:20-28`

**Step 1: 修改 __init__ 方法签名和初始化**

将：
```python
def __init__(self, scenario: ScenarioContext):
    """初始化执行器

    Args:
        scenario: 测试场景配置
    """
    self.scenario = scenario
    self.env = None
    self.result = TestResult(scenario_name=scenario.name, success=False)
```

改为：
```python
def __init__(self, scenario: ScenarioContext, verbose: bool = True, tile_format: str = "name"):
    """初始化执行器

    Args:
        scenario: 测试场景配置
        verbose: 是否打印详细信息
        tile_format: 牌显示格式，"name"（牌名）或 "number"（数字）
    """
    self.scenario = scenario
    self.env = None
    self.result = TestResult(scenario_name=scenario.name, success=False)
    self.verbose = verbose
    self.tile_format = tile_format
    self.visualizer = TileTextVisualizer()
    self.state_history: List[Tuple[GameStateType, str]] = []
```

**Step 2: 验证语法正确**

Run: `python -m py_compile tests/scenario/executor.py`
Expected: 无错误

**Step 3: 提交**

```bash
git add tests/scenario/executor.py
git commit -m "feat(test): add verbose and tile_format params to TestExecutor"
```

---

## Task 3: 添加格式化方法

**文件：**
- Modify: `tests/scenario/executor.py` (在 `_run_validations` 方法之前添加)

**Step 1: 添加格式化方法**

在 `_run_validations` 方法前添加：

```python
    # ==================== 格式化方法 ====================

    def _format_hand(self, hand_tiles: List[int]) -> str:
        """格式化手牌显示

        示例输出: [1万, 5万, 8条, 1筒, 2筒, 东, 南, 西, 北]
        """
        if self.tile_format == "name":
            # 使用 TileTextVisualizer，不分组，得到 "1万 5万 8条..."
            formatted = self.visualizer.format_hand(hand_tiles, group_by_suit=False)
            # 替换空格为 ", " 并加上方括号
            return "[" + ", ".join(formatted.split()) + "]"
        else:
            # 数字格式: [1, 5, 8, 19, 20, 28, 29, 30, 31]
            return "[" + ", ".join(map(str, sorted(hand_tiles))) + "]"

    def _format_action_param(self, action_type: ActionType, param: int) -> str:
        """格式化动作参数，牌ID转为牌名"""
        # 需要格式化牌名的动作类型
        tile_actions = {
            ActionType.DISCARD,
            ActionType.KONG_SUPPLEMENT,
            ActionType.KONG_CONCEALED,
            ActionType.KONG_SKIN,
            ActionType.KONG_LAZY,
            ActionType.KONG_RED,
        }

        if action_type in tile_actions and param >= 0:
            if self.tile_format == "name":
                return self.visualizer.format_tile(Tiles(param))
            else:
                return str(param)
        return str(param)
```

**Step 2: 验证语法正确**

Run: `python -m py_compile tests/scenario/executor.py`
Expected: 无错误

**Step 3: 提交**

```bash
git add tests/scenario/executor.py
git commit -m "feat(test): add tile formatting methods"
```

---

## Task 4: 添加状态打印方法

**文件：**
- Modify: `tests/scenario/executor.py` (在格式化方法之后添加)

**Step 1: 添加 _print_game_state 方法**

```python
    # ==================== 打印方法 ====================

    def _print_game_state(self, step: StepConfig, is_before: bool = True):
        """打印当前游戏状态

        Args:
            step: 步骤配置
            is_before: True表示步骤开始前，False表示步骤执行后
        """
        prefix = "步骤执行前" if is_before else "步骤执行后"
        print(f"\n{'='*60}")
        print(f"步骤 {step.step_number}: {step.description} [{prefix}]")
        print(f"{'='*60}")

        context = self.env.context

        # 当前状态和玩家
        print(f"当前状态: {context.current_state.name}")
        print(f"当前玩家: {context.current_player_idx}")

        # 手牌
        print(f"\n--- 手牌 ---")
        for i, player in enumerate(context.players):
            print(f"  玩家 {i}: {self._format_hand(player.hand_tiles)}")

        # 弃牌堆（显示最后10张）
        if context.discard_pile:
            print(f"\n--- 弃牌堆 (最近10张) ---")
            recent = context.discard_pile[-10:]
            formatted_recent = [self._format_action_param(ActionType.DISCARD, t) for t in recent]
            print(f"  [{', '.join(formatted_recent)}]")

        # 牌墙数量
        print(f"\n牌墙剩余: {len(context.wall)} 张")
```

**Step 2: 添加 _print_action 方法**

```python
    def _print_action(self, player: int, action_type: ActionType, param: int, is_valid: bool):
        """打印动作，非法动作用红色，参数使用牌名"""
        formatted_param = self._format_action_param(action_type, param)
        action_str = f"{action_type.name}({formatted_param})"

        if is_valid:
            print(f"  玩家 {player} 执行: {action_str}")
        else:
            print(f"  {RED}玩家 {player} 尝试: {action_str} - 非法动作!{RESET}")
```

**Step 3: 添加 _record_state 方法**

```python
    # ==================== 状态记录方法 ====================

    def _record_state(self):
        """记录当前状态到历史"""
        state = self.env.state_machine.current_state_type
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.state_history.append((state, timestamp))
```

**Step 4: 验证语法正确**

Run: `python -m py_compile tests/scenario/executor.py`
Expected: 无错误

**Step 5: 提交**

```bash
git add tests/scenario/executor.py
git commit -m "feat(test): add state printing methods"
```

---

## Task 5: 添加动作合法性检测方法

**文件：**
- Modify: `tests/scenario/executor.py` (在状态记录方法之后添加)

**Step 1: 添加 _check_action_valid 方法**

```python
    # ==================== 动作验证方法 ====================

    def _check_action_valid(self, action: tuple) -> bool:
        """检查动作是否合法

        Args:
            action: (action_type_value, parameter) 元组
        """
        action_type, param = action
        mahjong_action = MahjongAction(ActionType(action_type), param)

        # 转换为 action index
        try:
            action_index = self.env._action_to_index(mahjong_action)
        except (ValueError, AttributeError):
            return False

        # 检查 action_mask
        mask = self.env.context.action_mask
        if mask is None or action_index < 0 or action_index >= len(mask):
            return False

        return mask[action_index] == 1
```

**Step 2: 验证语法正确**

Run: `python -m py_compile tests/scenario/executor.py`
Expected: 无错误

**Step 3: 提交**

```bash
git add tests/scenario/executor.py
git commit -m "feat(test): add action validity check method"
```

---

## Task 6: 修改 _execute_step 方法

**文件：**
- Modify: `tests/scenario/executor.py:113-133`

**Step 1: 替换 _execute_step 方法**

将原有的 `_execute_step` 方法替换为：

```python
    def _execute_step(self, step: StepConfig):
        """执行单个步骤

        Args:
            step: 步骤配置

        Raises:
            AssertionError: 验证失败
            Exception: 执行错误
        """
        # 1. 打印步骤执行前的状态
        if self.verbose:
            self._print_game_state(step, is_before=True)

        # 2. 检查动作合法性（如果是动作步骤）
        is_valid_action = True
        if step.is_action:
            action = (step.action_type.value, step.parameter)
            is_valid_action = self._check_action_valid(action)

        # 3. 执行步骤
        if step.is_auto:
            self._auto_advance(step)
        elif step.is_action:
            self._execute_action(step, is_valid_action)

        # 4. 记录状态转换
        self._record_state()

        # 5. 打印步骤执行后的状态
        if self.verbose:
            self._print_game_state(step, is_before=False)

        # 6. 执行验证
        self._run_validations(step)
```

**Step 2: 验证语法正确**

Run: `python -m py_compile tests/scenario/executor.py`
Expected: 无错误

**Step 3: 提交**

```bash
git add tests/scenario/executor.py
git commit -m "feat(test): update _execute_step with logging and validation"
```

---

## Task 7: 修改 _execute_action 方法

**文件：**
- Modify: `tests/scenario/executor.py:135-152`

**Step 1: 替换 _execute_action 方法**

将原有的 `_execute_action` 方法替换为：

```python
    def _execute_action(self, step: StepConfig, is_valid_action: bool):
        """执行动作步骤

        Args:
            step: 步骤配置
            is_valid_action: 动作是否合法

        Raises:
            AssertionError: 验证失败
        """
        # 打印动作（带颜色）
        self._print_action(step.player, step.action_type, step.parameter, is_valid_action)

        # 只有合法动作才执行
        if is_valid_action:
            action = (step.action_type.value, step.parameter)
            obs, reward, terminated, truncated, info = self.env.step(action)
            print(f"  → 转移到状态: {self.env.state_machine.current_state_type.name}")
        else:
            print(f"  → 动作未执行")
```

**Step 2: 验证语法正确**

Run: `python -m py_compile tests/scenario/executor.py`
Expected: 无错误

**Step 3: 提交**

```bash
git add tests/scenario/executor.py
git commit -m "feat(test): update _execute_action with validity check and colored output"
```

---

## Task 8: 运行测试验证功能

**文件：**
- Test: 使用现有场景测试文件验证

**Step 1: 创建简单测试脚本**

创建 `tests/integration/test_scenario_logging.py`:

```python
"""测试场景测试的日志增强功能"""

from tests.scenario.builder import ScenarioBuilder
from src.mahjong_rl.core.constants import GameStateType, ActionType

def test_basic_scenario_with_logging():
    """测试基本场景的日志输出"""

    result = (
        ScenarioBuilder("日志测试")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                1: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                2: [26, 27, 28, 29, 30, 31, 32, 33, 0, 1, 2, 3, 4],
                3: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            },
            'wall': list(range(18, 100)),
            'special_tiles': {'lazy': 8, 'skins': [7, 9]},
            'last_drawn_tile': 12,
        })
        .step(1, "玩家0打牌")
            .action(0, ActionType.DISCARD, 5)
            .expect_state(GameStateType.WAITING_RESPONSE)
        .run()
    )

    assert result.success, f"测试失败: {result.failure_message}"
    print("\n=== 测试通过 ===")

if __name__ == "__main__":
    test_basic_scenario_with_logging()
```

**Step 2: 运行测试查看日志输出**

Run: `python tests/integration/test_scenario_logging.py`
Expected: 看到格式化的手牌、动作输出，带颜色标记

**Step 3: 测试非法动作（红色输出）**

在测试中添加非法动作，验证红色输出：
```python
.step(2, "玩家1尝试非法动作")
    .action(1, ActionType.DISCARD, 999)  # 不在手牌中的牌
```

**Step 4: 提交测试文件**

```bash
git add tests/integration/test_scenario_logging.py
git commit -m "test(test): add scenario logging test"
```

---

## Task 9: 更新 ScenarioBuilder 以支持传递 verbose 参数

**文件：**
- Modify: `tests/scenario/builder.py:324-336`

**Step 1: 修改 run 方法支持参数传递**

将：
```python
    def run(self) -> 'TestResult':
        """执行测试场景

        Returns:
            TestResult 测试结果
        """
        from tests.scenario.executor import TestExecutor

        # 自动添加最后一个未添加的步骤
        self._add_pending_step()

        executor = TestExecutor(self.context)
        return executor.run()
```

改为：
```python
    def run(self, verbose: bool = True, tile_format: str = "name") -> 'TestResult':
        """执行测试场景

        Args:
            verbose: 是否打印详细信息
            tile_format: 牌显示格式，"name"（牌名）或 "number"（数字）

        Returns:
            TestResult 测试结果
        """
        from tests.scenario.executor import TestExecutor

        # 自动添加最后一个未添加的步骤
        self._add_pending_step()

        executor = TestExecutor(self.context, verbose=verbose, tile_format=tile_format)
        return executor.run()
```

**Step 2: 验证语法正确**

Run: `python -m py_compile tests/scenario/builder.py`
Expected: 无错误

**Step 3: 提交**

```bash
git add tests/scenario/builder.py
git commit -m "feat(test): support verbose and tile_format in ScenarioBuilder.run()"
```

---

## Task 10: 最终测试和文档

**文件：**
- Create: `docs/plans/2026-02-05-scenario-test-logging-usage.md`

**Step 1: 创建使用文档**

创建文档说明如何使用新的日志功能：

```markdown
# 场景测试日志增强使用指南

## 功能概述

场景测试框架现在支持详细的日志输出，包括：
- 每步执行前后的游戏状态（手牌、弃牌堆等）
- 动作合法性检测（非法动作用红色标记）
- 牌名显示（1万、5条、东风等）

## 使用方法

### 基本使用（默认启用详细日志）

```python
result = ScenarioBuilder("测试场景")
    .step(1, "第一步")
        .action(0, ActionType.DISCARD, 5)
    .run()  # 默认 verbose=True, tile_format="name"
```

### 关闭详细日志

```python
result = ScenarioBuilder("测试场景")
    .step(1, "第一步")
        .action(0, ActionType.DISCARD, 5)
    .run(verbose=False)
```

### 使用数字格式显示牌

```python
result = ScenarioBuilder("测试场景")
    .step(1, "第一步")
        .action(0, ActionType.DISCARD, 5)
    .run(tile_format="number")  # 输出: DISCARD(5) 而不是 DISCARD(5万)
```

## 输出示例

```
============================================================
步骤 1: 玩家0打牌 [步骤执行前]
============================================================
当前状态: PLAYER_DECISION
当前玩家: 0

--- 手牌 ---
  玩家 0: [一万, 二万, 三万, 四万, 五万, 六万, 七万, 八万, 九万, 一条, 二条, 三条, 四条]
  玩家 1: [...]

牌墙剩余: 82 张

  玩家 0 执行: DISCARD(五万)
  → 转移到状态: WAITING_RESPONSE
```

## 非法动作检测

当测试中包含非法动作时，会以红色显示：

```
  <red>玩家 0 尝试: DISCARD(999) - 非法动作!</red>
  → 动作未执行
```
```

**Step 2: 提交文档**

```bash
git add docs/plans/2026-02-05-scenario-test-logging-usage.md
git commit -m "docs(test): add scenario logging usage guide"
```

**Step 3: 运行完整测试验证所有功能**

Run: `python tests/integration/test_scenario_logging.py`
Expected: 所有功能正常工作，日志输出正确

---

## 总结

本计划完成了以下功能：

1. ✅ 添加颜色常量和必要导入
2. ✅ 修改 `TestExecutor.__init__` 支持配置参数
3. ✅ 实现手牌和动作参数格式化（复用 `TileTextVisualizer`）
4. ✅ 实现状态打印方法（执行前后状态、手牌、弃牌堆）
5. ✅ 实现动作合法性检测（基于 action_mask）
6. ✅ 实现带颜色的动作打印（非法动作红色标记）
7. ✅ 修改执行流程集成所有新功能
8. ✅ 支持切换牌显示格式（牌名/数字）
9. ✅ 添加使用文档

## 提交记录

完成后应该有以下提交：
- feat(test): add imports and color constants
- feat(test): add verbose and tile_format params
- feat(test): add tile formatting methods
- feat(test): add state printing methods
- feat(test): add action validity check method
- feat(test): update _execute_step with logging
- feat(test): update _execute_action with validity check
- test(test): add scenario logging test
- feat(test): support params in ScenarioBuilder.run()
- docs(test): add usage guide
