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
    .run(tile_format="number")  # 输出: DISCARD(5) 而不是 DISCARD(五万)
```

### 同时关闭日志和使用数字格式

```python
result = ScenarioBuilder("测试场景")
    .step(1, "第一步")
        .action(0, ActionType.DISCARD, 5)
    .run(verbose=False, tile_format="number")
```

## 输出示例

### 正常动作输出

```
============================================================
步骤 1: 玩家0打牌 [步骤执行前]
============================================================
当前状态: PLAYER_DECISION
当前玩家: 0

--- 手牌 ---
  玩家 0: [一万, 二万, 三万, 四万, 五万, 六万, 七万, 八万, 九万, 一条, 二条, 三条, 四条]
  玩家 1: [五条, 六条, 七条, 八条, 九条, 一筒, 二筒, 三筒, 四筒, 五筒, 六筒, 七筒, 八筒]
  玩家 2: [东风, 南风, 西风, 北风, 红中, 发财, 白板, 一万, 二万, 三万, 四万, 五万, 六万]
  玩家 3: [七万, 八万, 九万, 一条, 二条, 三条, 四条, 五条, 六条, 七条, 八条, 九条]

牌墙剩余: 82 张

  玩家 0 执行: DISCARD(五万)
  → 转移到状态: WAITING_RESPONSE

============================================================
步骤 1: 玩家0打牌 [步骤执行后]
============================================================
当前状态: WAITING_RESPONSE
当前玩家: 1

--- 手牌 ---
  玩家 0: [一万, 二万, 三万, 四万, 六万, 七万, 八万, 九万, 一条, 二条, 三条, 四条]
  玩家 1: [五条, 六条, 七条, 八条, 九条, 一筒, 二筒, 三筒, 四筒, 五筒, 六筒, 七筒, 八筒]
  ...

--- 弃牌堆 (最近10张) ---
  [五万]

牌墙剩余: 82 张
```

### 非法动作检测

当测试中包含非法动作时，会以红色显示：

```
  <red>玩家 0 尝试: DISCARD(999) - 非法动作!</red>
  → 动作未执行
```

### 杠牌动作输出

```
  玩家 0 执行: KONG_CONCEALED(五万)
  → 转移到状态: GONG
```

### 补杠动作输出

```
  玩家 1 执行: KONG_SUPPLEMENT(三条)
  → 转移到状态: GONG
```

## 参数说明

### verbose 参数

- **True** (默认): 打印详细的日志信息，包括：
  - 步骤执行前后的游戏状态
  - 所有玩家的手牌
  - 弃牌堆（最近10张）
  - 牌墙剩余数量
  - 动作执行信息

- **False**: 只显示基本的步骤信息，不打印详细状态

### tile_format 参数

- **"name"** (默认): 使用中文牌名显示
  - 示例: `DISCARD(五万)`, `KONG_CONCEALED(东风)`

- **"number"**: 使用数字显示牌ID
  - 示例: `DISCARD(5)`, `KONG_CONCEALED(28)`

## 支持的动作类型

以下动作类型的参数会被格式化为牌名：
- `DISCARD` - 打牌
- `KONG_SUPPLEMENT` - 补杠
- `KONG_CONCEALED` - 暗杠
- `KONG_SKIN` - 皮子杠
- `KONG_LAZY` - 赖子杠
- `KONG_RED` - 红中杠

其他动作类型（如 `PASS`, `WIN`, `PONG` 等）的参数保持数字显示。

## 完整示例

```python
from tests.scenario.builder import ScenarioBuilder
from src.mahjong_rl.core.constants import GameStateType, ActionType

result = (
    ScenarioBuilder("完整日志示例")
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
    .step(2, "其他玩家过牌")
        .auto_advance()
        .expect_state(GameStateType.DRAWING)
    .run(verbose=True, tile_format="name")
)

if result.success:
    print("测试成功!")
else:
    print(f"测试失败: {result.failure_message}")
```

## 注意事项

1. **ANSI 颜色支持**: 红色输出依赖于终端支持 ANSI 转义码。大部分现代终端（包括 Windows 10+ 的 cmd、PowerShell）都支持。

2. **性能影响**: 启用 `verbose=True` 会打印大量信息，可能影响性能。在批量测试时建议使用 `verbose=False`。

3. **牌名映射**: 牌名显示依赖于 `TileTextVisualizer` 类，该类已经实现了所有牌的中文映射。
