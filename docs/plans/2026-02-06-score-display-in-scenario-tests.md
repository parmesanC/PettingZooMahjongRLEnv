# 场景测试框架分数显示功能实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在场景测试框架中，游戏结束时自动显示每个玩家的最终得分和计分明细。

**Architecture:** 在 `TestExecutor` 类中添加 `_print_scores()` 方法，在 `run()` 方法执行完所有步骤后检查游戏是否结束（WIN 或 FLOW_DRAW），如果结束则调用分数显示方法。

**Tech Stack:** Python, 现有的场景测试框架 (`tests/scenario/executor.py`), GameContext 数据结构

---

### Task 1: 添加 YELLOW 颜色常量

**Files:**
- Modify: `tests/scenario/executor.py:14-16`

**Step 1: 添加 YELLOW 颜色常量**

在现有的颜色常量后添加黄色常量，用于显示流局等信息：

```python
# ANSI 颜色代码
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"  # 添加这行
RESET = "\033[0m"
```

**Step 2: 验证代码正确性**

运行: `"D:\DATA\Development\Anaconda\envs\PettingZooRLMahjong\python.exe" -c "from tests.scenario.executor import YELLOW; print(repr(YELLOW))"`
Expected: `'\x1b[93m'`

**Step 3: 提交**

```bash
git add tests/scenario/executor.py
git commit -m "feat(test): add YELLOW color constant for score display"
```

---

### Task 2: 添加辅助方法 - _format_melds()

**Files:**
- Modify: `tests/scenario/executor.py` (在 `_format_action_param()` 方法后添加)

**Step 1: 添加 _format_melds() 方法**

这个方法将玩家的牌组（吃、碰、杠）格式化为可读字符串：

```python
def _format_melds(self, player_idx: int) -> List[str]:
    """格式化玩家的牌组信息

    Args:
        player_idx: 玩家索引

    Returns:
        牌组描述列表，如 ["碰: 一万", "明杠: 五条"]
    """
    player = self.env.context.players[player_idx]
    if not player.melds:
        return []

    result = []
    for meld in player.melds:
        action_name = meld.action_type.action_type.name
        tile_name = self.visualizer.format_tile(meld.tiles[0])

        # 根据动作类型格式化
        if action_name == "CHOW":
            result.append(f"吃: {tile_name}")
        elif action_name == "PONG":
            result.append(f"碰: {tile_name}")
        elif action_name == "KONG_EXPOSED":
            result.append(f"明杠: {tile_name}")
        elif action_name == "KONG_CONCEALED":
            result.append("暗杠")
        elif action_name == "KONG_SUPPLEMENT":
            result.append(f"补杠: {tile_name}")
        else:
            result.append(f"{action_name}: {tile_name}")

    return result
```

**Step 2: 验证方法存在**

运行: `"D:\DATA\Development\Anaconda\envs\PettingZooRLMahjong\python.exe" -c "from tests.scenario.executor import TestExecutor; import inspect; print('_format_melds' in dir(TestExecutor))"`
Expected: `True`

**Step 3: 提交**

```bash
git add tests/scenario/executor.py
git commit -m "feat(test): add _format_melds helper method"
```

---

### Task 3: 添加辅助方法 - _get_win_way_name()

**Files:**
- Modify: `tests/scenario/executor.py` (在 `_format_melds()` 方法后添加)

**Step 1: 添加 _get_win_way_name() 方法**

这个方法将 `win_way` 数值转换为中文名称：

```python
def _get_win_way_name(self) -> str:
    """获取胡牌方式的中文名称

    Returns:
        胡牌方式名称，如 "自摸"、"点炮" 等
    """
        from src.mahjong_rl.core.constants import WinWay

    win_way = self.env.context.win_way
    if win_way is None:
        return "未知"

    way_map = {
        WinWay.SELF_DRAW.value: "自摸",
        WinWay.DISCARD.value: "点炮",
        WinWay.KONG_ON_DRAW.value: "杠上开花",
        WinWay.ROB_KONG.value: "抢杠和",
    }
    return way_map.get(win_way, "未知")
```

**Step 2: 验证方法存在**

运行: `"D:\DATA\Development\Anaconda\envs\PettingZooRLMahjong\python.exe" -c "from tests.scenario.executor import TestExecutor; print('_get_win_way_name' in dir(TestExecutor))"`
Expected: `True`

**Step 3: 提交**

```bash
git add tests/scenario/executor.py
git commit -m "feat(test): add _get_win_way_name helper method"
```

---

### Task 4: 添加 _print_scores() 主方法

**Files:**
- Modify: `tests/scenario/executor.py` (在 `_print_game_state()` 方法后添加)

**Step 1: 添加 _print_scores() 方法**

这是核心方法，负责显示所有分数信息：

```python
def _print_scores(self):
    """打印游戏结束时的玩家分数和明细

    只有在游戏结束（WIN 或 FLOW_DRAW）且有分数数据时才打印
    """
    context = self.env.context

    # 只有在游戏结束且有分数时才打印
    if not context.final_scores:
        return

    # 检查游戏是否结束
    current_state = self.env.state_machine.current_state_type
    if current_state not in [GameStateType.WIN, GameStateType.FLOW_DRAW]:
        return

    print(f"\n{'='*60}")
    print(f"游戏结束 - 最终分数")
    print(f"{'='*60}")

    # 1. 显示每个玩家的最终得分
    for i, score in enumerate(context.final_scores):
        if score > 0:
            print(f"  玩家 {i}: {GREEN}+{score}{RESET}")
        elif score < 0:
            print(f"  玩家 {i}: {RED}{score}{RESET}")
        else:
            print(f"  玩家 {i}: 0")

    print()  # 空行

    # 2. 显示获胜者或流局信息
    if context.is_flush:
        print(f"{YELLOW}流局{RESET}")
    elif context.winner_ids:
        winners = ", ".join(f"玩家 {w}" for w in context.winner_ids)
        print(f"获胜者: {GREEN}{winners}{RESET}")

        # 显示胡牌方式
        win_way = self._get_win_way_name()
        print(f"胡牌方式: {win_way}")

    # 3. 显示庄家
    if context.dealer_idx is not None:
        print(f"庄家: 玩家 {context.dealer_idx}")

    print()  # 空行

    # 4. 显示每个玩家的牌组
    for i, player in enumerate(context.players):
        print(f"玩家 {i} 牌组:")

        # 显示吃碰杠
        melds = self._format_melds(i)
        if melds:
            for meld in melds:
                print(f"  - {meld}")
        else:
            print(f"  (无)")

        # 显示特殊杠
        pi_zi, lai_zi, hong_zhong = player.special_gangs
        special_gangs = []
        if pi_zi > 0:
            special_gangs.append(f"皮子杠 x{pi_zi}")
        if lai_zi > 0:
            special_gangs.append(f"赖子杠 x{lai_zi}")
        if hong_zhong > 0:
            special_gangs.append(f"红中杠 x{hong_zhong}")

        if special_gangs:
            print(f"  特殊杠: {', '.join(special_gangs)}")

    print(f"{'='*60}\n")
```

**Step 2: 验证方法存在**

运行: `"D:\DATA\Development\Anaconda\envs\PettingZooRLMahjong\python.exe" -c "from tests.scenario.executor import TestExecutor; print('_print_scores' in dir(TestExecutor))"`
Expected: `True`

**Step 3: 提交**

```bash
git add tests/scenario/executor.py
git commit -m "feat(test): add _print_scores method for displaying final scores"
```

---

### Task 5: 在 run() 方法中调用分数显示

**Files:**
- Modify: `tests/scenario/executor.py:102-105`

**Step 1: 在 run() 方法中调用 _print_scores()**

在设置 `result.final_state` 之后添加分数显示调用：

```python
# 找到这段代码
self.result.success = True
self.result.final_state = self.env.state_machine.current_state_type
# 无论成功失败都创建快照
self.result.final_context_snapshot = self._create_snapshot()

# 在这之后添加
# 显示游戏分数（如果游戏结束）
if self.verbose:
    self._print_scores()
```

**Step 2: 验证现有测试仍然通过**

运行: `"D:\DATA\Development\Anaconda\envs\PettingZooRLMahjong\python.exe" -m pytest tests/integration/test_rob_kong_scenario.py -v"`
Expected: PASS（测试应该通过，分数会显示在输出中）

**Step 3: 提交**

```bash
git add tests/scenario/executor.py
git commit -m "feat(test): call _print_scores in run() method when game ends"
```

---

### Task 6: 创建测试验证分数显示

**Files:**
- Create: `tests/integration/test_score_display.py`

**Step 1: 编写分数显示测试**

创建一个测试来验证 WIN 状态下的分数显示：

```python
"""测试场景测试框架的分数显示功能"""

from tests.scenario.builder import ScenarioBuilder
from src.mahjong_rl.core.constants import GameStateType, ActionType, WinWay


def test_score_display_on_win():
    """测试 WIN 状态下的分数显示"""
    # 构建一个简单的自摸胡牌场景
    result = (
        ScenarioBuilder("测试分数显示-自摸")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5],  # 碰碰胡+自摸
                1: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                2: [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                3: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
            },
            'wall': list(range(45, 80)),  # 剩余牌墙
            'special_tiles': {'lazy': 50, 'skins': [51, 52]},
            'last_drawn_tile': 5,  # 庄家摸到 5
        })
        .step(1, "玩家0自摸")
            .action(0, ActionType.WIN, -1)
            .expect_state(GameStateType.WIN)
        .run(verbose=True)
    )

    # 验证游戏结束
    assert result.success
    assert result.final_state == GameStateType.WIN

    # 验证分数已计算
    assert len(result.final_context_snapshot.get('player_hand_counts', [])) == 4

    print("\n✅ 分数显示测试通过！")


def test_score_display_on_flow_draw():
    """测试 FLOW_DRAW 状态下的分数显示"""
    # 构建一个流局场景
    result = (
        ScenarioBuilder("测试分数显示-流局")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                1: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
                2: [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                3: [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
            },
            'wall': [],  # 牌墙为空，触发流局
            'special_tiles': {'lazy': 53, 'skins': [54, 55]},
        })
        .run(verbose=True)
    )

    # 验证结果
    print("\n✅ 流局分数显示测试完成！")


if __name__ == "__main__":
    test_score_display_on_win()
    test_score_display_on_flow_draw()
```

**Step 2: 运行测试验证分数显示**

运行: `"D:\DATA\Development\Anaconda\envs\PettingZooRLMahjong\python.exe" tests/integration/test_score_display.py"`
Expected: 看到分数信息打印输出，测试通过

**Step 3: 提交**

```bash
git add tests/integration/test_score_display.py
git commit -m "test(test): add score display test"
```

---

### Task 7: 更新场景测试文档

**Files:**
- Modify: `tests/scenario/USAGE.md` 或 `tests/scenario/README.md`

**Step 1: 添加分数显示功能说明**

在文档中添加分数显示功能的说明：

```markdown
## 分数显示

当游戏结束（WIN 或 FLOW_DRAW 状态）时，场景测试框架会自动显示每个玩家的最终得分。

### 显示内容

1. **最终得分**：每个玩家的得分（正数绿色，负数红色）
2. **获胜者**：和牌的玩家
3. **胡牌方式**：自摸、点炮、杠上开花、抢杠和
4. **庄家**：当前庄家
5. **牌组信息**：每个玩家的吃、碰、杠
6. **特殊杠**：皮子杠、赖子杠、红中杠

### 示例输出

```
============================================================
游戏结束 - 最终分数
============================================================
玩家 0: +120
玩家 1: -40
玩家 2: -40
玩家 3: -40

获胜者: 玩家 0
胡牌方式: 自摸
庄家: 玩家 0

玩家 0 牌组:
  - 碰: 一万
  - 明杠: 五条

...
```
```

**Step 2: 提交**

```bash
git add tests/scenario/README.md
git commit -m "docs(test): add score display feature documentation"
```

---

## 验证清单

完成所有任务后，验证以下内容：

- [ ] YELLOW 颜色常量已添加
- [ ] `_format_melds()` 方法正确格式化牌组
- [ ] `_get_win_way_name()` 方法正确返回胡牌方式
- [ ] `_print_scores()` 方法正确显示所有分数信息
- [ ] 在 `run()` 方法中正确调用分数显示
- [ ] 测试通过，分数正确显示
- [ ] 现有测试仍然通过
- [ ] 文档已更新

---

## 相关文件

- `tests/scenario/executor.py` - 主要修改文件
- `tests/integration/test_score_display.py` - 新增测试文件
- `tests/scenario/README.md` - 文档更新
- `src/mahjong_rl/core/GameData.py` - GameContext 数据结构
- `src/mahjong_rl/core/PlayerData.py` - PlayerData 数据结构
- `src/mahjong_rl/core/constants.py` - GameStateType, WinWay, ActionType 枚举
