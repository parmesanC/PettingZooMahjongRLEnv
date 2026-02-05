# 自定义初始状态功能实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 为场景测试框架添加自定义初始状态功能，允许用户绕过 InitialState 的自动初始化，直接设置手牌、牌墙、庄家等，实现精确的游戏状态控制。

**架构：** 在 ScenarioContext 中添加 initial_config 字段，在 ScenarioBuilder 中添加 with_initial_state() 方法，在 TestExecutor 中检测并应用自定义初始化，完全绕过 env.reset() 的自动发牌流程。

**Tech Stack：** Python 3.x, dataclasses, 武汉麻将状态机, PettingZoo AECEnv

---

## Task 1: 扩展 ScenarioContext 数据结构

**Files:**
- Modify: `tests/scenario/context.py:36-53`

**Step 1: 添加 initial_config 字段**

在 `ScenarioContext` 类中添加新字段：

```python
@dataclass
class ScenarioContext:
    """测试场景上下文"""
    name: str
    description: str = ""

    # 游戏初始配置
    wall: List[int] = field(default_factory=list)
    special_tiles: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None

    # 自定义初始状态配置（绕过 InitialState）
    initial_config: Optional[Dict[str, Any]] = None  # 新增

    # 步骤配置
    steps: List[StepConfig] = field(default_factory=list)

    # 终止验证（游戏结束时）
    final_validators: List[Callable] = field(default_factory=list)
    expect_winner: Optional[List[int]] = None
```

**Step 2: 运行语法检查**

Run: `python -m py_compile tests/scenario/context.py`
Expected: 无输出（语法正确）

**Step 3: 提交**

```bash
git add tests/scenario/context.py
git commit -m "feat(scenario): add initial_config field for custom game state"
```

---

## Task 2: 添加 with_initial_state() 方法

**Files:**
- Modify: `tests/scenario/builder.py`
- Test: 后续任务会使用此方法

**Step 1: 在 ScenarioBuilder 类中添加 with_initial_state() 方法**

在 `tests/scenario/builder.py` 的 `ScenarioBuilder` 类中添加方法（在 `expect_winner()` 方法之后）：

```python
def with_initial_state(self, config: Dict[str, Any]) -> 'ScenarioBuilder':
    """设置自定义初始状态，绕过 InitialState 的自动初始化

    这允许用户完全控制游戏初始状态，包括：
    - 庄家位置和当前玩家
    - 每个玩家的手牌
    - 牌墙顺序
    - 特殊牌（赖子、皮子）
    - 庄家刚摸的牌（用于 PLAYER_DECISION 状态）

    Args:
        config: 初始状态配置字典
            - dealer_idx (int): 庄家位置 (0-3)
            - current_player_idx (int): 当前玩家 (0-3)
            - hands (Dict[int, List[int]]): 玩家手牌 {player_id: [tiles]}
            - wall (List[int]): 牌墙
            - special_tiles (Dict): 特殊牌 {lazy: int, skins: [int, int]}
            - last_drawn_tile (int, optional): 庄家刚摸的牌

    Returns:
        self，支持链式调用

    Example:
        ```python
        result = (
            ScenarioBuilder("自定义状态测试")
            .with_initial_state({
                'dealer_idx': 0,
                'current_player_idx': 0,
                'hands': {
                    0: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6],  # 13张
                    1: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                    2: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                    3: [33, 33, 33, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
                'wall': [11, 12, 13, ...],  # 剩余牌墙
                'special_tiles': {'lazy': 8, 'skins': [7, 9]},
                'last_drawn_tile': 6,
            })
            .run()
        )
        ```
    """
    self.context.initial_config = config
    return self
```

**Step 2: 运行语法检查**

Run: `python -m py_compile tests/scenario/builder.py`
Expected: 无输出（语法正确）

**Step 3: 提交**

```bash
git add tests/scenario/builder.py
git commit -m "feat(scenario): add with_initial_state() method for custom game initialization"
```

---

## Task 3: 实现 _apply_custom_initialization() 方法

**Files:**
- Modify: `tests/scenario/executor.py`

**Step 1: 添加 _apply_custom_initialization() 方法**

在 `TestExecutor` 类中添加新方法（在 `_create_snapshot()` 方法之后）：

```python
def _apply_custom_initialization(self) -> None:
    """应用自定义初始状态配置

    根据 scenario.initial_config 配置游戏状态，
    完全绕过 InitialState 的自动初始化流程。

    Raises:
        ValueError: 如果配置缺少必需字段
    """
    config = self.scenario.initial_config
    if not config:
        return

    context = self.env.context

    # 1. 设置庄家
    if 'dealer_idx' in config:
        dealer_idx = config['dealer_idx']
        if not 0 <= dealer_idx <= 3:
            raise ValueError(f"dealer_idx 必须在 0-3 之间，得到 {dealer_idx}")
        context.dealer_idx = dealer_idx

    # 2. 设置当前玩家
    if 'current_player_idx' in config:
        current_player = config['current_player_idx']
        if not 0 <= current_player <= 3:
            raise ValueError(f"current_player_idx 必须在 0-3 之间，得到 {current_player}")
        context.current_player_idx = current_player

    # 3. 设置玩家手牌
    if 'hands' in config:
        hands = config['hands']
        for player_id, tiles in hands.items():
            if not 0 <= player_id <= 3:
                raise ValueError(f"玩家ID必须在 0-3 之间，得到 {player_id}")
            context.players[player_id].hand_tiles = tiles.copy()
            # 设置 is_dealer 标志
            if 'dealer_idx' in config:
                context.players[player_id].is_dealer = (player_id == config['dealer_idx'])

    # 4. 设置牌墙
    if 'wall' in config:
        context.wall.clear()
        context.wall.extend(config['wall'])

    # 5. 设置特殊牌
    if 'special_tiles' in config:
        special = config['special_tiles']
        if 'lazy' in special:
            context.lazy_tile = special['lazy']
        if 'skins' in special:
            skins = special['skins']
            if len(skins) >= 2:
                context.skin_tile = [skins[0], skins[1]]
        # 更新 special_tiles 元组
        context.special_tiles = (
            context.lazy_tile,
            context.skin_tile[0] if context.skin_tile else -1,
            context.skin_tile[1] if context.skin_tile else -1,
            context.red_dragon
        )

    # 6. 设置 last_drawn_tile（庄家刚摸的牌）
    if 'last_drawn_tile' in config:
        context.last_drawn_tile = config['last_drawn_tile']

    # 7. 初始化其他必要字段
    from src.mahjong_rl.core.constants import GameStateType
    context.current_state = GameStateType.PLAYER_DECISION
    context.observation = None
    context.action_mask = None

    # 重置响应相关状态
    context.discard_pile = []
    context.last_discarded_tile = None
    context.pending_responses = {}
    context.response_order = []
    context.current_responder_idx = 0
    context.selected_responder = None
    context.response_priorities = {}

    # 重置杠牌相关状态
    context.last_kong_action = None
    context.last_kong_player_idx = None

    # 重置游戏结果状态
    context.is_win = False
    context.is_flush = False
    context.winner_ids = []
    context.reward = 0.0

    # 初始化 special_gangs
    for i in range(4):
        context.players[i].special_gangs = [0, 0, 0]
```

**Step 2: 运行语法检查**

Run: `python -m py_compile tests/scenario/executor.py`
Expected: 无输出（语法正确）

**Step 3: 提交**

```bash
git add tests/scenario/executor.py
git commit -m "feat(scenario): add _apply_custom_initialization() method"
```

---

## Task 4: 修改 run() 方法使用自定义初始化

**Files:**
- Modify: `tests/scenario/executor.py:30-102`

**Step 1: 修改 run() 方法**

将现有的 `run()` 方法中从 `# 重置环境` 到 `# 配置牌墙` 的部分替换为：

```python
def run(self) -> TestResult:
    """执行测试场景

    Returns:
        测试结果
    """
    try:
        # 延迟导入避免循环依赖
        from example_mahjong_env import WuhanMahjongEnv

        # 创建环境
        self.env = WuhanMahjongEnv(
            render_mode=None,
            training_phase=3,  # 完整信息
            enable_logging=False  # 测试时关闭日志
        )

        # 检查是否有自定义初始状态配置
        if self.scenario.initial_config is not None:
            # 使用自定义初始化，绕过 env.reset()
            self._apply_custom_initialization()
        else:
            # 使用标准初始化流程
            self.env.reset(seed=self.scenario.seed)

            # 配置牌墙（标准流程）
            if self.scenario.wall:
                self.env.context.wall.clear()
                self.env.context.wall.extend(self.scenario.wall)

            # 配置特殊牌（标准流程）
            if self.scenario.special_tiles:
                if 'lazy' in self.scenario.special_tiles:
                    self.env.context.lazy_tile = self.scenario.special_tiles['lazy']
                if 'skins' in self.scenario.special_tiles:
                    skins = self.scenario.special_tiles['skins']
                    if len(skins) >= 2:
                        self.env.context.skin_tile = [skins[0], skins[1]]

        self.result.total_steps = len(self.scenario.steps)

        # 执行每个步骤
        for step_config in self.scenario.steps:
            self._execute_step(step_config)
            self.result.executed_steps += 1

        # 执行最终验证
        if self.scenario.final_validators:
            for validator in self.scenario.final_validators:
                if not validator(self.env.context):
                    raise AssertionError(f"最终验证失败: {validator.__name__}")

        # 验证获胜者
        if self.scenario.expect_winner is not None:
            if set(self.env.context.winner_ids) != set(self.scenario.expect_winner):
                raise AssertionError(
                    f"获胜者验证失败: 预期 {self.scenario.expect_winner}, "
                    f"实际 {self.env.context.winner_ids}"
                )

        self.result.success = True
        self.result.final_state = self.env.state_machine.current_state_type

    except Exception as e:
        self.result.success = False
        self.result.failed_step = self.result.executed_steps + 1
        self.result.failure_message = str(e)

        # 保存快照用于调试
        if self.env and self.env.context:
            self.result.final_context_snapshot = self._create_snapshot()

    finally:
        # 确保环境资源被正确释放
        if self.env is not None:
            self.env.close()

    return self.result
```

**Step 2: 运行语法检查**

Run: `python -m py_compile tests/scenario/executor.py`
Expected: 无输出（语法正确）

**Step 3: 提交**

```bash
git add tests/scenario/executor.py
git commit -m "feat(scenario): modify run() to support custom initialization"
```

---

## Task 5: 添加单元测试

**Files:**
- Create: `tests/integration/test_custom_initialization.py`

**Step 1: 编写测试用例**

创建 `tests/integration/test_custom_initialization.py`：

```python
"""测试自定义初始状态功能"""

import pytest
from tests.scenario.builder import ScenarioBuilder
from tests.scenario.validators import hand_count_equals, wall_count_equals, state_is
from src.mahjong_rl.core.constants import GameStateType, ActionType


def test_custom_initialization_basic():
    """测试基本自定义初始化"""
    result = (
        ScenarioBuilder("自定义初始化测试")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # 13张
                1: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                2: [26, 27, 28, 29, 30, 31, 32, 33, 0, 1, 2, 3, 4],
                3: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            },
            'wall': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
            'special_tiles': {'lazy': 8, 'skins': [7, 9]},
            'last_drawn_tile': 12,
        })
        .step(1, "验证初始状态")
            .auto_advance()
            .verify("玩家0手牌13张", hand_count_equals(0, 13))
            .verify("玩家1手牌13张", hand_count_equals(1, 13))
            .verify("玩家2手牌13张", hand_count_equals(2, 13))
            .verify("玩家3手牌13张", hand_count_equals(3, 13))
            .verify("牌墙剩余16张", wall_count_equals(16))
            .verify("当前是PLAYER_DECISION状态", state_is(GameStateType.PLAYER_DECISION))
        .run()
    )

    assert result.success, f"测试失败: {result.failure_message}"


def test_custom_initialization_with_action():
    """测试自定义初始化后执行动作"""
    result = (
        ScenarioBuilder("自定义初始化后打牌")
        .with_initial_state({
            'dealer_idx': 1,  # 玩家1是庄家
            'current_player_idx': 1,
            'hands': {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                1: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                2: [26, 27, 28, 29, 30, 31, 32, 33, 0, 1, 2, 3, 4],
                3: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            },
            'wall': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
            'special_tiles': {'lazy': 5, 'skins': [4, 6]},
            'last_drawn_tile': 25,
        })
        .step(1, "玩家1打牌")
            .action(1, ActionType.DISCARD, 25)
            .expect_state(GameStateType.WAITING_RESPONSE)
        .run()
    )

    assert result.success, f"测试失败: {result.failure_message}"


def test_custom_initialization_dealer():
    """测试庄家设置"""
    result = (
        ScenarioBuilder("庄家设置测试")
        .with_initial_state({
            'dealer_idx': 2,  # 玩家2是庄家
            'current_player_idx': 2,
            'hands': {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                1: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                2: [26, 27, 28, 29, 30, 31, 32, 33, 0, 1, 2, 3, 4],
                3: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            },
            'wall': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
            'special_tiles': {'lazy': 8, 'skins': [7, 9]},
            'last_drawn_tile': 30,
        })
        .step(1, "验证庄家")
            .auto_advance()
            .verify("当前玩家是玩家2", lambda ctx: ctx.current_player_idx == 2)
            .verify("玩家2是庄家", lambda ctx: ctx.players[2].is_dealer)
        .run()
    )

    assert result.success, f"测试失败: {result.failure_message}"


def test_custom_initialization_special_tiles():
    """测试特殊牌设置"""
    result = (
        ScenarioBuilder("特殊牌设置测试")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                1: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                2: [26, 27, 28, 29, 30, 31, 32, 33, 0, 1, 2, 3, 4],
                3: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            },
            'wall': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
            'special_tiles': {'lazy': 5, 'skins': [4, 6]},
            'last_drawn_tile': 12,
        })
        .step(1, "验证特殊牌")
            .auto_advance()
            .verify("赖子是5", lambda ctx: ctx.lazy_tile == 5)
            .verify("皮子是4和6", lambda ctx: ctx.skin_tile == [4, 6])
        .run()
    )

    assert result.success, f"测试失败: {result.failure_message}"


def test_custom_initialization_error_handling():
    """测试错误处理"""
    # 测试无效的 dealer_idx
    result = (
        ScenarioBuilder("错误处理测试")
        .with_initial_state({
            'dealer_idx': 5,  # 无效值
            'current_player_idx': 0,
            'hands': {0: [0, 1, 2]},
            'wall': [],
        })
        .run()
    )

    assert not result.success
    assert "dealer_idx 必须在 0-3 之间" in result.failure_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Step 2: 运行测试**

Run: `pytest tests/integration/test_custom_initialization.py -v`
Expected: 部分测试通过（错误处理测试应该通过，其他测试可能需要调试）

**Step 3: 提交**

```bash
git add tests/integration/test_custom_initialization.py
git commit -m "test(scenario): add tests for custom initialization feature"
```

---

## Task 6: 更新使用文档

**Files:**
- Modify: `tests/scenario/USAGE.md`

**Step 1: 在 USAGE.md 中添加新章节**

在 `tests/scenario/USAGE.md` 中添加新章节（在"完整使用示例"之后，"调试技巧"之前）：

```markdown
## 自定义初始状态

使用 `with_initial_state()` 方法可以完全绕过 InitialState 的自动初始化，
直接设置手牌、牌墙、庄家等，实现精确的游戏状态控制。

### 基本用法

```python
from tests.scenario.builder import ScenarioBuilder

result = (
    ScenarioBuilder("自定义状态测试")
    .with_initial_state({
        # 庄家相关
        'dealer_idx': 0,           # 庄家是谁 (0-3)
        'current_player_idx': 0,   # 当前该谁出牌 (0-3)

        # 手牌（每人13张）
        'hands': {
            0: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6],
            1: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            2: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
            3: [33, 33, 33, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        },

        # 牌墙
        'wall': [11, 12, 13, 14, 15, 16, 17, 18, ...],  # 剩余的牌

        # 特殊牌
        'special_tiles': {
            'lazy': 8,        # 赖子牌ID
            'skins': [7, 9],  # 皮子牌ID [skin1, skin2]
        },

        # 庄家刚摸的牌（用于 PLAYER_DECISION 状态）
        'last_drawn_tile': 6,
    })
    .run()
)
```

### 配置字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `dealer_idx` | int | 是 | 庄家位置 (0-3) |
| `current_player_idx` | int | 是 | 当前玩家 (0-3) |
| `hands` | Dict[int, List[int]] | 是 | 玩家手牌 {player_id: [tiles]} |
| `wall` | List[int] | 是 | 牌墙 |
| `special_tiles` | Dict | 是 | 特殊牌 {lazy: int, skins: [int, int]} |
| `last_drawn_tile` | int | 推荐 | 庄家刚摸的牌 |

### 完整示例：测试特定胡牌场景

```python
from tests.scenario.builder import ScenarioBuilder
from tests.scenario.validators import hand_contains
from src.mahjong_rl.core.constants import GameStateType, ActionType

result = (
    ScenarioBuilder("测试玩家0自摸胡牌")
    .with_initial_state({
        'dealer_idx': 0,
        'current_player_idx': 0,
        'hands': {
            # 玩家0：一手能胡的牌（对对胡）
            0: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
            # 其他玩家：普通手牌
            1: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            2: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
            3: [33, 33, 33, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
        # 牌墙只剩最后一张，玩家0摸到就能胡
        'wall': [6],  # 这张牌让玩家0摸到
        'special_tiles': {'lazy': 8, 'skins': [7, 9]},
        'last_drawn_tile': 6,
    })
    .step(1, "玩家0自摸胡牌")
        .action(0, ActionType.WIN, -1)
        .expect_state(GameStateType.WIN)
    .run()
)

assert result.success
assert 0 in result.final_context_snapshot['winner_ids']
```

### 与标准初始化的区别

**标准初始化**（使用 `with_wall()`）：
- 牌墙会自动洗牌
- 玩家手牌从牌墙随机发牌
- 庄家随机确定
- 手牌和牌墙不匹配

**自定义初始化**（使用 `with_initial_state()`）：
- 完全控制手牌和牌墙
- 牌和牌完美匹配
- 精确测试特定场景

### 注意事项

1. **手牌数量**：闲家13张，庄家14张（包括 last_drawn_tile）
2. **牌墙完整性**：确保牌墙 + 所有手牌 = 136张（标准牌墙）
3. **状态一致性**：当前玩家应该是庄家（除非测试特殊状态）
```

**Step 2: 提交**

```bash
git add tests/scenario/USAGE.md
git commit -m "docs(scenario): add documentation for custom initial state feature"
```

---

## Task 7: 添加到 __init__.py 导出

**Files:**
- Modify: `tests/scenario/__init__.py`

**Step 1: 更新导出**

确保 `with_initial_state()` 方法可以被用户导入使用（可选，如果用户通过 ScenarioBuilder 导入则不需要）。

**Step 2: 运行测试验证**

Run: `python -c "from tests.scenario.builder import ScenarioBuilder; print('导入成功')"`
Expected: 打印 "导入成功"

---

## 完成检查清单

- [ ] 所有测试通过: `pytest tests/integration/test_custom_initialization.py -v`
- [ ] 标准初始化仍然工作: `pytest tests/integration/test_scenarios.py -v`
- [ ] 代码符合项目风格
- [ ] 所有文件已提交
- [ ] 文档已更新

## 运行所有测试

```bash
# 运行自定义初始化测试
pytest tests/integration/test_custom_initialization.py -v

# 运行所有场景测试（确保向后兼容）
pytest tests/scenario/ tests/integration/test_*scenarios.py -v

# 运行所有相关测试
pytest tests/scenario/ tests/integration/ -v
```
