# CLI 交互问题修复记录

> **文档类型**: 问题修复记录（简洁参考式）
> **创建日期**: 2026-01-23
> **修改范围**: 3个文件，约 175 行代码
> **目的**: 记录 CLI 交互问题的根因分析和解决方案，便于将来参考和代码审查

---

## 1. 问题概览

| 问题 | 严重程度 | 描述 |
|------|----------|------|
| **问题1** | 高 | 频繁出现只有 PASS 动作但仍要输入 |
| **问题2** | 中 | CHOW 动作提示不明确，不显示具体牌名 |
| **问题3** | 高 | WIN 动作检测不正确，满足胡牌条件时 WIN 不可用 |

---

## 2. 问题1：只有 PASS 仍要输入

### 2.1 问题描述

**现象**: 在命令行游戏时，即使某个玩家只有 PASS 动作可用，系统仍然要求手动输入 `(10, -1)`。

**频率**: 频繁出现，几乎每局都会遇到。

### 2.2 根本原因

**位置**: `example_mahjong_env.py` 的 `step()` 方法

**问题**: `WaitResponseState.enter()` 中所有玩家自动 PASS 后，状态自动转换（如 WAITING_RESPONSE → DRAWING → PLAYER_DECISION），但 `env.agent_selection` 没有正确同步更新。

**代码路径**:
1. `WaitResponseState.enter()` 检测到所有玩家只能 PASS
2. 设置 `active_responders = []`（空列表）
3. 调用 `_select_best_response(context)` 直接转换状态
4. 状态转换到 `DRAWING`，`context.current_player_idx` 已更新
5. **但 `env.agent_selection` 仍指向旧的响应玩家**
6. ManualController 迭代到旧 agent，要求输入
7. action_mask 显示只有 PASS 可用

### 2.3 解决方案（修订版）

**修改文件**:
- `src/mahjong_rl/state_machine/base.py` - 添加 `should_auto_skip()` 方法
- `src/mahjong_rl/state_machine/states/wait_response_state.py` - 实现自动跳过逻辑
- `src/mahjong_rl/state_machine/machine.py` - 添加自动跳过处理

**核心设计**: 引入"自动跳过状态"模式（Null Action Pattern）

1. **GameState.should_auto_skip()** - 声明状态是否可以被自动跳过
2. **transition_to() 自动检查** - 进入状态后检查是否跳过
3. **_auto_skip_state()** - 统一处理自动跳过逻辑

**设计原则**（符合六大设计原则）:
- **SRP**: `enter()` 只负责初始化，`should_auto_skip()` 负责跳过判断
- **OCP**: 通过扩展 `should_auto_skip()` 实现自动跳过，不需要修改状态机核心
- **LSP**: 所有状态都可以重写 `should_auto_skip()`，不影响其他状态
- **ISP**: `should_auto_skip()` 是可选接口，默认实现返回 False
- **DIP**: 状态机依赖 GameState 抽象，不关心具体实现
- **LoD**: `transition_to()` 只调用当前状态的方法，不深入内部

**实施状态**: ✅ 已完成（Tasks 1-8）

### 2.4 影响范围

- **文件**:
  - `src/mahjong_rl/state_machine/base.py` (新增方法)
  - `src/mahjong_rl/state_machine/states/wait_response_state.py` (重构)
  - `src/mahjong_rl/state_machine/machine.py` (新增逻辑)
- **行数**: 约 100 行（新增/修改）
- **测试**: `tests/integration/test_auto_skip_state.py`

---

## 3. 问题2：CHOW 动作提示不明确

### 3.1 问题描述

**现象**:
- 当前显示: `(1, 0/1/2) 吃牌 - 0=左吃, 1=中吃, 2=右吃`
- 期望显示: "弃牌3万，只能：右吃(1,2)"
- 非法输入应该报错，而不是自动 PASS

### 3.2 根本原因

**位置**: `cli_controller.py` 的 `_parse_tuple_input()` 方法

**问题**:
1. 没有显示弃牌名称
2. 没有显示具体需要的牌名
3. 错误信息不够具体

### 3.3 解决方案

**修改文件**: `cli_controller.py`

**新增方法**:

1. **`_get_chow_help_info()`**: 获取吃牌的详细帮助信息
   - 获取弃牌名称
   - 检查可用的吃牌类型
   - 显示具体需要的牌名

2. **`_get_chow_tiles_description()`**: 获取吃牌需要的具体牌描述
   - 根据弃牌和吃牌类型计算需要的牌
   - 转换为具体的牌名

**效果示例**:
```
# 之前
⚠️ 吃牌 当前不可用，可用: 左吃(1,0), 中吃(1,1)

# 之后
⚠️ 吃牌 当前不可用，弃牌3万，只能：右吃(1,2) [需1万2万]
```

### 3.4 改进前后对比

| 项目 | 之前 | 之后 |
|------|------|------|
| 显示格式 | `(1, 0/1/2) 吃牌` | `弃牌X万，只能：右吃(1,2) [需X万Y万]` |
| 错误提示 | 简单列出可用类型 | 显示具体需要的牌名 |
| 用户体验 | 需要玩家自己推算 | 直接显示具体信息 |

---

## 4. 问题3：WIN 动作检测不正确

### 4.1 问题描述

**现象**: 当手牌满足一切胡牌条件时，WIN 动作仍然不在 action_mask 中。

**场景**: 摸牌后手牌满足胡牌条件，但 WIN 动作不可用。

### 4.2 根本原因

**位置**: `action_validator.py` 的 `detect_available_actions_after_draw()` 方法

**问题链条**:
1. 原代码调用 `score_calculator.settle(result, self.context)`
2. `settle()` 内部调用 `_find_winner(ctx)`
3. `_find_winner()` 检查 `player.is_win` 属性
4. 但在检测可用动作时，`player.is_win` 还是 `False`（因为玩家还没胡）
5. 返回 `[0.0, 0.0, 0.0, 0.0]`
6. 条件判断失败，WIN 动作不被添加

**原始错误代码**:
```python
score_list = score_calculator.settle(result, self.context)
winer_score = max(score_list)
if all(abs(score) + winer_score >= 16 for score in score_list):
    available_actions.append(MahjongAction(ActionType.WIN, -1))
```

### 4.3 解决方案

**修改文件**:
- `action_validator.py`（简化调用）
- `score_calculator.py`（新增公共方法）

**设计原则**（符合六大设计原则）:
- **单一职责原则**: score_calculator 负责所有分数/番数计算
- **代码复用**: 复用 `_get_base_fan_score()` 方法
- **依赖倒置**: action_validator 依赖 score_calculator 的公共接口

**新增公共方法** (`score_calculator.py`):
```python
def check_min_fan_requirement(self, winner_id: int, win_types: list, ctx: GameContext) -> bool:
    """
    检查是否满足起胡番要求

    规则：胡牌者自身番数 × 场上番数最小玩家番数 >= 16
    注意：番数不包括自摸、放冲因子（这些用于计算分数，不计入起胡番）
    """
    # 计算胡牌者的基础番数（包括硬胡因子）
    base_fan = self._get_base_fan_score(winner)
    if WinType.HARD_WIN in win_types:
        winner_fan = base_fan * 2.0
    else:
        winner_fan = base_fan

    # 计算所有玩家的番数，找到最小值
    min_other_fan = min([
        self._get_base_fan_score(p)
        for p in ctx.players
        if p.player_id != winner_id
    ])

    # 检查：赢家番数 × 最小番数 >= 16
    return winner_fan * min_other_fan >= 16
```

**调用代码** (`action_validator.py`):
```python
if result.can_win:
    # 修复问题3：调用 score_calculator 的方法检查起胡番
    from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.score_calculator import MahjongScoreSettler
    score_calculator = MahjongScoreSettler(False)
    can_win = score_calculator.check_min_fan_requirement(
        current_player.player_id, result.win_type, self.context
    )
    if can_win:
        available_actions.append(MahjongAction(ActionType.WIN, -1))
    return available_actions
```

### 4.4 关键改进

| 方面 | 改进 |
|------|------|
| **逻辑正确性** | 正确实现"赢家番数 × 最小番数 >= 16"的规则 |
| **职责分离** | score_calculator 专门计算番数，action_validator 负责动作验证 |
| **代码质量** | 消除约 100 行重复代码 |
| **可维护性** | 集中管理番数计算逻辑 |

---

## 5. 快速参考

### 5.1 修改文件清单

| 文件 | 修改类型 | 新增/修改行数 |
|------|----------|---------------|
| `example_mahjong_env.py` | 逻辑修复 | ~25 行 |
| `cli_controller.py` | 功能增强 | ~110 行 |
| `score_calculator.py` | 新增公共方法 | ~40 行 |
| `action_validator.py` | 逻辑修复 | ~-40 行（简化） |
| **总计** | - | **~135 行** |

### 5.2 关键代码位置索引

**问题1**: `example_mahjong_env.py:381-403`
**问题2**: `cli_controller.py:118-124`, `214-321`
**问题3**: `score_calculator.py:360-398`, `action_validator.py:203-212`

### 5.3 相关文档

- `docs/plans/2026-01-23-auto-skip-state-pattern.md`: 自动跳过状态模式实施计划
- `docs/architecture/auto-skip-state-pattern.md`: 自动跳过状态模式架构文档
- `TASK9_SUMMARY.md`: Task 9 手动测试验证报告
- `TASK9_TESTING_GUIDE.md`: Task 9 测试指南

---

## 6. 测试验证

### 6.1 验证步骤

1. **启动游戏**:
   ```bash
   python play_mahjong.py --mode four_human --renderer cli
   ```

2. **测试问题1**:
   - 观察是否还会出现只有 PASS 仍要输入的情况
   - 预期：所有玩家自动 PASS 时，应该直接跳过，不要求输入

3. **测试问题2**:
   - 尝试输入非法的 CHOW 动作
   - 观察是否显示具体的牌名信息
   - 预期：显示类似 "弃牌3万，只能：右吃(1,2) [需1万2万]"

4. **测试问题3**:
   - 摸牌后手牌满足胡牌条件
   - 检查 WIN 动作是否可用
   - 预期：WIN 动作在 action_mask 中正确显示

### 6.1 验证步骤

1. **启动游戏**:
   ```bash
   python play_mahjong.py --mode four_human --renderer cli
   ```

2. **测试问题1**:
   - 观察是否还会出现只有 PASS 仍要输入的情况
   - 预期：所有玩家自动 PASS 时，应该直接跳过，不要求输入

3. **测试问题2**:
   - 尝试输入非法的 CHOW 动作
   - 观察是否显示具体的牌名信息
   - 预期：显示类似 "弃牌3万，只能：右吃(1,2) [需1万2万]"

4. **测试问题3**:
   - 摸牌后手牌满足胡牌条件
   - 检查 WIN 动作是否可用
   - 预期：WIN 动作在 action_mask 中正确显示

5. **运行集成测试**:
   ```bash
   python tests/integration/test_auto_skip_state.py
   ```

### 6.2 预期行为

- **问题1**: 不再出现只有 PASS 仍要输入的情况
- **问题2**: CHOW 动作提示清晰，显示具体牌名
- **问题3**: 满足胡牌条件时 WIN 动作可用

### 6.3 回滚计划

**注意**: 问题1已通过状态机模式修复，回滚需要回退多个提交。

如需回滚这些修改：
```bash
# 回滚到问题1修复之前（约8个提交）
git log --oneline -n 10  # 查找提交历史
git revert <commit-range>  # 或使用 git reset

# 问题2和问题3的回滚
git checkout HEAD~1 src/mahjong_rl/manual_control/cli_controller.py
git checkout HEAD~1 src/mahjong_rl/rules/wuhan_mahjong_rule_engine/action_validator.py
git checkout HEAD~1 src/mahjong_rl/rules/wuhan_mahjong_rule_engine/score_calculator.py
```

---

**文档状态**: ✅ 完成
**最后更新**: 2026-01-23
**下次审查**: 当相关代码被修改时
