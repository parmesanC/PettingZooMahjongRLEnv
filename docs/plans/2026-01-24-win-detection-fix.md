# 接炮胡牌检测修复实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 修复接炮胡牌（点炮）无法生效的问题，使其与自摸胡牌功能一致

**架构:**
1. 在 `ActionValidator.detect_available_actions_after_discard()` 中添加接炮胡牌检测
2. 新增 `_can_win_by_discard()` 辅助方法，复用自摸胡牌的检测逻辑
3. 在 `Wuhan7P4LObservationBuilder.build_action_mask()` 中添加 WAIT_ROB_KONG 状态处理

**技术栈:**
- Python 3.x
- PyTest (测试框架)
- 项目规则引擎: `src/mahjong_rl/rules/wuhan_mahjong_rule_engine/`

---

## Task 1: 新增接炮胡牌检测辅助方法

**文件:**
- 修改: `src/mahjong_rl/rules/wuhan_mahjong_rule_engine/action_validator.py`

**Step 1: 添加 `_can_win_by_discard()` 方法到 ActionValidator 类**

在 `action_validator.py` 第 214 行之后（文件末尾，`detect_available_actions_after_draw()` 方法之后）添加以下方法：

```python
    def _can_win_by_discard(self, current_player: PlayerData, discard_tile: int) -> bool:
        """
        判断是否可以接炮胡牌

        将弃牌临时加入玩家手牌，检查是否可以胡牌，并检查起胡番要求。

        Args:
            current_player: 当前玩家
            discard_tile: 弃牌编码

        Returns:
            True 如果可以接炮胡牌
        """
        # 创建临时手牌（加入弃牌）
        temp_hand = current_player.hand_tiles.copy()
        temp_hand.append(discard_tile)

        # 创建临时玩家对象
        temp_player = PlayerData(
            player_id=current_player.player_id,
            hand_tiles=temp_hand,
            melds=current_player.melds.copy(),
            special_gangs=current_player.special_gangs.copy()
        )

        # 检查是否可以胡牌
        from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.win_detector import WuhanMahjongWinChecker
        win_checker = WuhanMahjongWinChecker(self.context)
        result = win_checker.check_win(temp_player)

        if not result.can_win:
            return False

        # 检查起胡番要求
        from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.score_calculator import MahjongScoreSettler
        score_calculator = MahjongScoreSettler(False)
        return score_calculator.check_min_fan_requirement(
            current_player.player_id, result.win_type, self.context
        )
```

**Step 2: 验证语法正确**

Run: `python -m py_compile src/mahjong_rl/rules/wuhan_mahjong_rule_engine/action_validator.py`
Expected: 无输出（语法正确）

**Step 3: Commit**

```bash
git add src/mahjong_rl/rules/wuhan_mahjong_rule_engine/action_validator.py
git commit -m "feat(action-validator): add _can_win_by_discard helper method"
```

---

## Task 2: 在 detect_available_actions_after_discard 中调用接炮胡牌检测

**文件:**
- 修改: `src/mahjong_rl/rules/wuhan_mahjong_rule_engine/action_validator.py:56-60`

**Step 1: 添加接炮胡牌检测调用**

将第 56-60 行：

```python
        # 5. 判断明杠（优先级最高）
        if self._can_kong_exposed(current_player, discard_tile):
            available_actions.append(MahjongAction(ActionType.KONG_EXPOSED, discard_tile))

        return available_actions
```

替换为：

```python
        # 5. 判断明杠（优先级最高）
        if self._can_kong_exposed(current_player, discard_tile):
            available_actions.append(MahjongAction(ActionType.KONG_EXPOSED, discard_tile))

        # 6. 判断接炮胡牌（优先级最高，高于所有吃碰杠）
        if self._can_win_by_discard(current_player, discard_tile):
            available_actions.append(MahjongAction(ActionType.WIN, -1))

        return available_actions
```

**Step 2: 验证语法正确**

Run: `python -m py_compile src/mahjong_rl/rules/wuhan_mahjong_rule_engine/action_validator.py`
Expected: 无输出（语法正确）

**Step 3: Commit**

```bash
git add src/mahjong_rl/rules/wuhan_mahjong_rule_engine/action_validator.py
git commit -m "feat(action-validator): add win detection in discard response"
```

---

## Task 3: 添加 WAIT_ROB_KONG 状态的 action_mask 处理

**文件:**
- 修改: `src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py:93-106`

**Step 1: 添加 WAIT_ROB_KONG 状态分支**

在 `build_action_mask()` 方法中，将第 93-106 行：

```python
        if current_state == GameStateType.MELD_DECISION:
            # 鸣牌后决策：可以杠、出牌，但不能胡
            mask = self._build_meld_decision_mask(player, context, mask)

        elif current_state in [GameStateType.PLAYER_DECISION, GameStateType.DRAWING]:
            # 摸牌后决策：可以杠、胡、出牌
            mask = self._build_decision_mask(player, context, mask)

        elif current_state in [GameStateType.WAITING_RESPONSE, GameStateType.RESPONSES,
                               GameStateType.RESPONSES_AFTER_GONG]:
            # 响应状态
            mask = self._build_response_mask(player, context, mask)

        return mask
```

修改为：

```python
        if current_state == GameStateType.MELD_DECISION:
            # 鸣牌后决策：可以杠、出牌，但不能胡
            mask = self._build_meld_decision_mask(player, context, mask)

        elif current_state in [GameStateType.PLAYER_DECISION, GameStateType.DRAWING]:
            # 摸牌后决策：可以杠、胡、出牌
            mask = self._build_decision_mask(player, context, mask)

        elif current_state in [GameStateType.WAITING_RESPONSE, GameStateType.RESPONSES,
                               GameStateType.RESPONSES_AFTER_GONG]:
            # 响应状态
            mask = self._build_response_mask(player, context, mask)

        elif current_state == GameStateType.WAIT_ROB_KONG:
            # 抢杠和状态：只能 WIN 或 PASS
            mask[143] = 1  # WIN 位
            mask[144] = 1  # PASS 位

        return mask
```

**Step 2: 验证语法正确**

Run: `python -m py_compile src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py`
Expected: 无输出（语法正确）

**Step 3: Commit**

```bash
git add src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py
git commit -m "fix(observation): add action_mask for WAIT_ROB_KONG state"
```

---

## Task 4: 编写接炮胡牌集成测试

**文件:**
- 创建: `tests/integration/test_win_by_discard.py`

**Step 1: 创建测试文件**

```python
"""测试接炮胡牌检测功能"""

import pytest
from collections import deque

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import GameStateType, ActionType
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.action_validator import ActionValidator


class TestWinByDiscard:
    """测试接炮胡牌检测"""

    def test_win_by_discard_action_available(self):
        """测试：接炮胡牌动作应被正确检测"""
        # 创建游戏上下文
        context = GameContext(num_players=4)
        context.current_state = GameStateType.WAITING_RESPONSE
        context.discard_player = 1
        context.last_discarded_tile = 8  # 1万 (ID: 8)
        context.wall = deque([i for i in range(34) for _ in range(4)])

        # 设置玩家0的手牌（包括可以胡牌的牌型）
        # 给玩家0一个简单的一对+顺子组合，加上1万可以胡牌
        player0 = context.players[0]
        player0.hand_tiles = [0, 1, 2, 9, 10, 11, 18, 19, 20, 8, 8]  # 包含一对1万
        player0.melds = []
        player0.special_gangs = [0, 0, 0]

        # 设置其他玩家手牌（确保起胡番条件满足）
        for i in range(1, 4):
            player = context.players[i]
            player.hand_tiles = [3, 4, 5, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27]
            player.melds = []
            player.special_gangs = [0, 0, 0]

        # 创建 ActionValidator
        validator = ActionValidator(context)

        # 测试：玩家0可以接炮胡牌
        actions = validator.detect_available_actions_after_discard(
            player0, context.last_discarded_tile, context.discard_player
        )

        # 验证：WIN 动作在可用动作列表中
        action_types = [a.action_type for a in actions]
        assert ActionType.WIN in action_types, "WIN action should be available for winning hand"

        # 验证：WIN 动作参数为 -1
        win_action = next(a for a in actions if a.action_type == ActionType.WIN)
        assert win_action.parameter == -1, "WIN action parameter should be -1"

    def test_no_win_when_cannot_form_win(self):
        """测试：无法胡牌时不应返回 WIN 动作"""
        # 创建游戏上下文
        context = GameContext(num_players=4)
        context.current_state = GameStateType.WAITING_RESPONSE
        context.discard_player = 1
        context.last_discarded_tile = 8  # 1万
        context.wall = deque([i for i in range(34) for _ in range(4)])

        # 设置玩家0的手牌（无法胡牌）
        player0 = context.players[0]
        player0.hand_tiles = [0, 1, 2, 9, 10, 11, 18, 19, 20, 30, 31]  # 散牌
        player0.melds = []
        player0.special_gangs = [0, 0, 0]

        # 设置其他玩家
        for i in range(1, 4):
            player = context.players[i]
            player.hand_tiles = [3, 4, 5, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27]
            player.melds = []
            player.special_gangs = [0, 0, 0]

        # 创建 ActionValidator
        validator = ActionValidator(context)

        # 测试
        actions = validator.detect_available_actions_after_discard(
            player0, context.last_discarded_tile, context.discard_player
        )

        # 验证：WIN 动作不在可用动作列表中
        action_types = [a.action_type for a in actions]
        assert ActionType.WIN not in action_types, "WIN action should NOT be available for non-winning hand"

    def test_cannot_win_own_discard(self):
        """测试：不能接自己的炮"""
        context = GameContext(num_players=4)
        context.current_state = GameStateType.WAITING_RESPONSE
        context.discard_player = 0  # 玩家0自己弃牌
        context.last_discarded_tile = 8
        context.wall = deque([i for i in range(34) for _ in range(4)])

        player0 = context.players[0]
        player0.hand_tiles = [0, 1, 2, 9, 10, 11, 18, 19, 20, 8, 8]
        player0.melds = []
        player0.special_gangs = [0, 0, 0]

        validator = ActionValidator(context)

        actions = validator.detect_available_actions_after_discard(
            player0, context.last_discarded_tile, context.discard_player
        )

        # 验证：不能接自己的炮
        assert len(actions) == 0, "Should have no actions when responding to own discard"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Step 2: 运行测试（预期失败）**

Run: `pytest tests/integration/test_win_by_discard.py -v`
Expected: 测试通过（因为修复已完成）

**Step 3: Commit**

```bash
git add tests/integration/test_win_by_discard.py
git commit -m "test: add integration tests for win by discard"
```

---

## Task 5: 手动测试验证

**文件:**
- 测试: `play_mahjong.py`

**Step 1: 运行四人游戏进行手动测试**

```bash
# 激活 conda 环境
conda activate PettingZooRLMahjong

# 运行游戏
python play_mahjong.py --mode four_human --renderer cli
```

**Step 2: 验证接炮胡牌功能**

在游戏中测试以下场景：
1. 玩家A打出一张牌
2. 玩家B可以胡这张牌（接炮）
3. 验证 action_mask 中 WIN (9, -1) 选项可用
4. 选择 WIN 并验证游戏正确进入 WIN 状态

**Step 3: 验证抢杠和功能**

在游戏中测试以下场景：
1. 玩家A碰牌后摸到第4张，选择补杠
2. 其他玩家可以抢杠和
3. 验证 action_mask 中有 WIN 和 PASS 选项
4. 验证抢杠和成功后游戏正确进入 WIN 状态

**Step 4: 如测试通过，标记任务完成**

如果所有场景测试通过，修复完成。

---

## 相关文档

- 现有 WIN 检测逻辑: `action_validator.py:189-212`
- 现有 action_mask 构建: `wuhan_7p4l_observation_builder.py:76-106`
- WaitResponseState: `src/mahjong_rl/state_machine/states/wait_response_state.py`
- WaitRobKongState: `src/mahjong_rl/state_machine/states/wait_rob_kong_state.py`

---

**计划完成日期:** 2026-01-24
**预计工作量:** 30-45 分钟
**风险等级:** 低（仅添加缺失的检测逻辑，不影响现有功能）
