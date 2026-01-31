# 场景测试框架使用文档

## 简介

场景测试框架是一个流式测试构建器，用于按照**预定牌墙顺序**和**出牌节奏**测试武汉麻将状态机。你可以精确控制每一步的动作，验证状态转换、游戏状态、动作验证和计分结果。

## 快速开始

### 基本结构

```python
from tests.scenario.builder import ScenarioBuilder
from src.mahjong_rl.core.constants import GameStateType, ActionType

result = (
    ScenarioBuilder("测试名称")
    .with_wall([...])           # 配置牌墙
    .with_special_tiles(...)    # 配置特殊牌
    .step(1, "第一步")
        .action(0, ActionType.DISCARD, 5)  # 玩家0打牌5
        .expect_state(GameStateType.WAITING_RESPONSE)
    .run()
)

assert result.success
```

---

## 核心API

### 1. ScenarioBuilder

场景构建器，用于流式构建测试场景。

#### 方法列表

| 方法 | 参数 | 说明 |
|------|------|------|
| `description(desc)` | `desc: str` | 设置场景描述 |
| `with_wall(tiles)` | `tiles: List[int]` | 设置牌墙顺序（牌ID列表） |
| `with_special_tiles(lazy, skins)` | `lazy: int`, `skins: List[int]` | 设置赖子和皮子 |
| `with_initial_state(config)` | `config: Dict` | 自定义初始状态（绕过自动初始化） |
| `step(num, desc)` | `num: int`, `desc: str` | 开始一个新步骤，返回 StepBuilder |
| `expect_winner(winners)` | `winners: List[int]` | 设置预期获胜玩家 |
| `run()` | 无 | 执行测试场景，返回 TestResult |

---

### 2. StepBuilder

步骤构建器，用于配置单个测试步骤。所有方法都返回 `self`，支持链式调用。

#### 方法列表

| 方法 | 参数 | 说明 |
|------|------|------|
| `action(player, type, param)` | `player: int`, `type: ActionType`, `param: int` | 指定玩家动作 |
| `auto_advance()` | 无 | 自动推进（用于自动状态） |
| `expect_state(state)` | `state: GameStateType` | 预期下一个状态 |
| `expect_action_mask(actions)` | `actions: List[ActionType]` | 预期可用的动作类型 |
| `verify(desc, validator)` | `desc: str`, `validator: Callable` | 添加自定义验证条件 |
| `verify_hand(player, tiles)` | `player: int`, `tiles: List[int]` | 验证玩家手牌 |
| `verify_wall_count(count)` | `count: int` | 验证牌墙剩余数量 |

---

## 常用验证器

验证器位于 `tests/scenario.validators`，用于验证游戏状态。

```python
from tests.scenario.validators import (
    hand_count_equals,      # 验证手牌数量
    hand_contains,          # 验证手牌包含指定牌
    wall_count_equals,      # 验证牌墙剩余数量
    discard_pile_contains,  # 验证弃牌堆包含某张牌
    state_is,               # 验证当前状态
    meld_count_equals,      # 验证副露数量
)
```

### 验证器使用示例

```python
# 验证玩家0有10张牌
.verify("手牌数量", hand_count_equals(0, 10))

# 验证玩家0手牌包含 [0, 1, 2]
.verify("手牌包含", hand_contains(0, [0, 1, 2]))

# 验证牌墙剩余130张
.verify("牌墙数量", wall_count_equals(130))

# 验证弃牌堆包含牌5
.verify("弃牌堆", discard_pile_contains(5))

# 验证当前状态
.verify("当前状态", state_is(GameStateType.PLAYER_DECISION))

# 验证副露数量
.verify("副白数量", meld_count_equals(0, 2))
```

---

## 完整使用示例

### 示例1：基本打牌流程

```python
from tests.scenario.builder import ScenarioBuilder
from src.mahjong_rl.core.constants import GameStateType, ActionType

# 创建标准牌墙
def create_wall():
    wall = []
    for tile_id in range(34):
        wall.extend([tile_id] * 4)
    return wall

result = (
    ScenarioBuilder("打牌测试")
    .with_wall(create_wall())

    # 步骤1：玩家0打出一张牌
    .step(1, "玩家0打牌")
        .action(0, ActionType.DISCARD, 5)  # 玩家0打出牌ID=5
        .expect_state(GameStateType.WAITING_RESPONSE)

    .run()
)

assert result.success
```

### 示例2：暗杠流程

```python
from tests.scenario.builder import ScenarioBuilder
from tests.scenario.validators import wall_count_equals, hand_count_equals
from src.mahjong_rl.core.constants import GameStateType, ActionType

result = (
    ScenarioBuilder("暗杠测试")
    .with_wall(create_wall())
    .with_special_tiles(lazy=8, skins=[7, 9])

    # 步骤1：玩家0暗杠1万
    .step(1, "暗杠1万")
        .action(0, ActionType.KONG_CONCEALED, 0)
        .expect_state(GameStateType.GONG)
        .verify("牌墙未变", wall_count_equals(136))  # 暗杠不摸牌
        .verify("手牌减少4张", hand_count_equals(0, 9))  # 13-4=9

    # 步骤2：杠后补牌
    .step(2, "杠后补牌")
        .auto_advance()
        .expect_state(GameStateType.DRAWING_AFTER_GONG)
        .verify("牌墙减少1张", wall_count_equals(135))

    .run()
)

assert result.success
```

### 示例3：多步流程

```python
from tests.scenario.builder import ScenarioBuilder
from tests.scenario.validators import state_is
from src.mahjong_rl.core.constants import GameStateType, ActionType

result = (
    ScenarioBuilder("完整对局片段")
    .with_wall(create_wall())

    # 玩家0打牌
    .step(1, "玩家0打牌")
        .action(0, ActionType.DISCARD, 0)
        .expect_state(GameStateType.WAITING_RESPONSE)

    # 其他玩家都过
    .step(2, "所有人过")
        .auto_advance()
        .expect_state(GameStateType.PLAYER_DECISION)

    # 玩家1打牌
    .step(3, "玩家1打牌")
        .action(1, ActionType.DISCARD, 5)
        .expect_state(GameStateType.WAITING_RESPONSE)

    # 玩家0碰牌
    .step(4, "玩家0碰牌")
        .action(0, ActionType.PONG, 5)
        .expect_state(GameStateType.PROCESSING_MELD)

    # 自动处理
    .step(5, "碰牌后处理")
        .auto_advance()
        .verify("回到决策状态", state_is(GameStateType.PLAYER_DECISION))

    .run()
)

assert result.success
```

### 示例4：自定义验证器

```python
from tests.scenario.builder import ScenarioBuilder

result = (
    ScenarioBuilder("自定义验证")
    .with_wall(create_wall())

    .step(1, "验证自定义条件")
        .auto_advance()
        .verify("玩家0是当前玩家", lambda ctx: ctx.current_player_idx == 0)
        .verify("牌墙不为空", lambda ctx: len(ctx.wall) > 0)
        .verify("所有玩家都有手牌", lambda ctx: all(len(p.hand_tiles) > 0 for p in ctx.players))

    .run()
)

assert result.success
```

---

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

---

## 常用动作类型

```python
from src.mahjong_rl.core.constants import ActionType

# 打牌
ActionType.DISCARD          # 打出一张牌

# 吃碰杠
ActionType.CHOW             # 吃牌
ActionType.PONG             # 碰牌
ActionType.KONG_EXPOSED     # 明杠
ActionType.KONG_CONCEALED   # 暗杠
ActionType.KONG_SUPPLEMENT  # 补杠
ActionType.KONG_RED         # 红中杠
ActionType.KONG_SKIN        # 皮子杠
ActionType.KONG_LAZY        # 赖子杠

# 胡牌
ActionType.WIN              # 胡牌

# 过
ActionType.PASS             # 过
```

---

## 常用状态类型

```python
from src.mahjong_rl.core.constants import GameStateType

# 手动状态（需要agent动作）
GameStateType.PLAYER_DECISION      # 玩家决策
GameStateType.WAITING_RESPONSE     # 等待响应
GameStateType.WAIT_ROB_KONG        # 等待抢杠
GameStateType.MELD_DECISION        # 吃牌决策

# 自动状态（自动推进）
GameStateType.INITIAL              # 初始
GameStateType.DISCARDING           # 出牌中
GameStateType.PROCESSING_MELD      # 处理吃碰
GameStateType.GONG                 # 杠牌中
GameStateType.DRAWING              # 摸牌中
GameStateType.DRAWING_AFTER_GONG   # 杠后补牌

# 终端状态
GameStateType.WIN                  # 胡牌
GameStateType.FLOW_DRAW            # 流局
```

---

## 牌ID对照表

```python
# 万字牌 (0-8)
0 = 一万, 1 = 二万, 2 = 三万, 3 = 四万, 4 = 五万,
5 = 六万, 6 = 七万, 7 = 八万, 8 = 九万

# 条子牌 (9-17)
9 = 一条, 10 = 二条, 11 = 三条, 12 = 四条, 13 = 五条,
14 = 六条, 15 = 七条, 16 = 八条, 17 = 九条

# 筒子牌 (18-26)
18 = 一筒, 19 = 二筒, 20 = 三筒, 21 = 四筒, 22 = 五筒,
23 = 六筒, 24 = 七筒, 25 = 八筒, 26 = 九筒

# 风牌 (27-30)
27 = 东, 28 = 南, 29 = 西, 30 = 北

# 箭牌 (31-33)
31 = 中, 32 = 发, 33 = 白
```

---

## 调试技巧

### 查看测试结果

```python
result = ScenarioBuilder("测试").run()

if not result.success:
    print(f"失败步骤: {result.failed_step}")
    print(f"失败原因: {result.failure_message}")
    print(f"快照: {result.final_context_snapshot}")
```

### 快照内容

测试失败时，`result.final_context_snapshot` 包含：

```python
{
    'current_state': 'PLAYER_DECISION',
    'current_player': 0,
    'wall_count': 130,
    'discard_pile': [5, 10, 15],  # 最后10张
    'player_hand_counts': [10, 13, 13, 13],
    'winner_ids': []
}
```

---

## 运行测试

```bash
# 运行单个测试文件
pytest tests/integration/test_scenarios.py -v

# 运行所有场景测试
pytest tests/scenario/ tests/integration/test_*scenarios.py -v

# 运行单个测试函数
pytest tests/integration/test_scenarios.py::test_scenario_builder_basic -v

# 显示详细输出
pytest tests/integration/test_scenarios.py -v -s
```

---

## 常见问题

### Q: 如何创建自定义牌墙？

```python
def create_custom_wall():
    # 完全自定义
    wall = [0, 0, 0, 0, 1, 1, 1, 1, ...]  # 136张牌

    # 或者循环生成
    wall = []
    for i in range(34):
        wall.extend([i] * 4)  # 每种牌4张
    return wall
```

### Q: 如何测试特定场景？

```python
# 先配置牌墙，让特定玩家拿到特定的牌
wall = [0] * 100 + [1, 1, 1, 1] + [...]  # 控制牌序

result = (
    ScenarioBuilder("测试特定场景")
    .with_wall(wall)
    .step(1, "...")
    .run()
)
```

### Q: 如何验证多个条件？

```python
.step(1, "多条件验证")
    .action(0, ActionType.DISCARD, 0)
    .verify("条件1", lambda ctx: len(ctx.wall) == 135)
    .verify("条件2", lambda ctx: ctx.current_player_idx == 1)
    .verify("条件3", lambda ctx: len(ctx.players[0].hand_tiles) == 10)
```

---

## 完整测试模板

```python
"""测试模板"""
import pytest
from tests.scenario.builder import ScenarioBuilder
from tests.scenario.validators import wall_count_equals, hand_count_equals
from src.mahjong_rl.core.constants import GameStateType, ActionType


def create_standard_wall():
    """创建标准牌墙"""
    wall = []
    for tile_id in range(34):
        wall.extend([tile_id] * 4)
    return wall


def test_xxx_scenario():
    """测试xxx场景"""
    wall = create_standard_wall()

    result = (
        ScenarioBuilder("xxx场景测试")
        .description("测试xxx功能")
        .with_wall(wall)
        .with_special_tiles(lazy=8, skins=[7, 9])

        # 步骤1
        .step(1, "第一步")
        .action(0, ActionType.DISCARD, 5)
        .expect_state(GameStateType.WAITING_RESPONSE)

        # 步骤2
        .step(2, "第二步")
        .auto_advance()
        .expect_state(GameStateType.PLAYER_DECISION)

        .run()
    )

    assert result.success, f"测试失败: {result.failure_message}"
```
