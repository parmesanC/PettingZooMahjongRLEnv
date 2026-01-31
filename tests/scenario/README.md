# 场景测试框架

流式测试构建器，用于按照预定牌墙顺序和出牌节奏测试状态机。

## 基本用法

```python
from tests.scenario.builder import ScenarioBuilder
from src.mahjong_rl.core.constants import GameStateType, ActionType

result = (
    ScenarioBuilder("测试名称")
    .description("测试描述")
    .with_wall([0,0,0,0, 1,1,1,1, ...])
    .with_special_tiles(lazy=5, skins=[4, 6])
    .step(1, "第一步描述")
        .action(0, ActionType.DISCARD, 5)
        .expect_state(GameStateType.WAITING_RESPONSE)
    .run()
)

assert result.success
```

## 验证器

- `hand_count_equals(player, count)` - 验证手牌数量
- `hand_contains(player, tiles)` - 验证手牌包含指定牌
- `wall_count_equals(count)` - 验证牌墙剩余数量
- `state_is(state)` - 验证当前状态

## 调试

测试失败时，`result.final_context_snapshot` 包含调试信息。
