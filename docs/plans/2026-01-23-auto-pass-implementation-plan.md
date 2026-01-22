# WaitResponseState 自动 PASS 优化实施计划

> **For Claude:** REQUIRED SUBKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 在响应收集阶段自动跳过只能 PASS 的玩家，减少约 25% 的训练时间步

**架构:** 在 WaitResponseState.enter() 中预先检测所有响应者，构建 `active_responders` 列表（只包含需要决策的玩家），自动 PASS 的玩家直接跳过。

**Tech Stack:** Python 3, PyTest, NumPy

---

## 任务结构

### 任务 1: 在 GameData.py 中添加新属性

**文件:**
- 修改: `src/mahjong_rl/core/GameData.py`

**步骤 1: 在 GameContext 类的 __init__ 方法中添加新属性**

在 GameData.py 的 GameContext 类的 `__init__` 方法中添加：

```python
# 响应状态优化：真实响应者列表（排除只能 PASS 的玩家）
self.active_responders = []  # 真实需要响应的玩家列表
self.active_responder_idx = 0  # 当前在 active_responders 中的索引
```

**步骤 2: 运行测试验证**

运行: `python -m pytest tests/ -k "test_game" -v` （或相关测试）
预期: 所有测试通过

**步骤 3: 提交**

```bash
git add src/mahjong_rl/core/GameData.py
git commit -m "feat: add active_responders properties to GameContext

为 WaitResponseState 优化添加数据结构支持：
- active_responders: 只包含需要决策的玩家列表
- active_responder_idx: 当前处理索引

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### 任务 2: 添加 _can_only_pass() 辅助方法

**文件:**
- 修改: `src/mahjong_rl/state_machine/states/wait_response_state.py`

**步骤 1: 在 WaitResponseState 类中添加辅助方法**

在 `wait_response_state.py` 的 WaitResponseState 类中添加 `_can_only_pass()` 方法：

```python
def _can_only_pass(self, context: GameContext, player_id: int) -> bool:
    """
    检查玩家是否只能 PASS

    Args:
        context: 游戏上下文
        player_id: 玩家ID

    Returns:
        True 如果只能 PASS，False 如果有其他可用动作
    """
    player = context.players[player_id]
    discard_tile = context.last_discarded_tile
    discard_player = context.discard_player

    # 获取可用动作
    available_actions = self.rule_engine.detect_available_actions_after_discard(
        player, discard_tile, discard_player
    )

    # 检查是否有非 PASS 的动作
    for action in available_actions:
        if action.action_type != ActionType.PASS:
            return False

    return True
```

**步骤 2: 运行测试验证**

运行: `python -m pytest tests/ -k "response" -v` （如果存在响应相关测试）
预期: 所有测试通过

**步骤 3: 提交**

```bash
git add src/mahjong_rl/state_machine/states/wait_response_state.py
git commit -m "feat: add _can_only_pass helper to WaitResponseState

添加辅助方法用于判断玩家是否只能 PASS，
为自动 PASS 优化提供判断逻辑

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### 任务 3: 重写 enter() 方法实现自动 PASS 逻辑

**文件:**
- 修改: `src/mahjong_rl/state_machine/states/wait_response_state.py`

**步骤 1: 替换 enter() 方法**

完全替换 `enter()` 方法为以下实现：

```python
def enter(self, context: GameContext) -> None:
    """
    进入等待响应状态

    优化：自动处理只能 PASS 的玩家
    """
    context.current_state = GameStateType.WAITING_RESPONSE

    # 初始化响应收集器
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from mahjong_rl.state_machine.ResponseCollector import ResponseCollector
    context.response_collector = ResponseCollector()

    # 确保响应顺序已设置（自动排除出牌者）
    if not context.response_order:
        context.setup_response_order(context.discard_player)

    # 构建真实响应者列表（排除只能 PASS 的玩家）
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

    # 检查是否需要响应
    if not context.active_responders:
        # 所有人都只能 PASS，直接选择最佳响应
        self._select_best_response(context)
    else:
        # 为第一个真实响应者生成观测
        first_responder = context.active_responders[0]
        context.current_player_idx = first_responder
        # 立即生成观测和动作掩码
        self.build_observation(context)
```

**步骤 2: 运行测试验证**

运行: `python test_four_ai.py` （运行几局游戏测试）
预期: 游戏正常运行，响应阶段自动跳过只能 PASS 的玩家

**步骤 3: 提交**

```bash
git add src/mahjong_rl/state_machine/states/wait_response_state.py
git commit -m "feat: implement auto-pass logic in WaitResponseState.enter()

重构 enter() 方法：
- 预先检测所有响应者，过滤出只能 PASS 的玩家
- 构建 active_responders 列表，只包含需要决策的玩家
- 自动 PASS 的玩家直接添加到 response_collector
- 如果所有人都能 PASS，直接调用 _select_best_response()

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### 任务 4: 修改 step() 方法只处理 active_responders

**文件:**
- 修改: `src/mahjong_rl/state_machine/states/wait_response_state.py`

**步骤 1: 替换 step() 方法**

完全替换 `step()` 方法为以下实现：

```python
def step(self, context: GameContext, action: Union[MahjongAction, str]) -> GameStateType:
    """
    处理一个真实响应者的响应

    Args:
        context: 游戏上下文
        action: 玩家响应动作或'auto'

    Returns:
        WAITING_RESPONSE (继续) 或下一个状态
    """
    # 检查是否还有待处理的响应者
    if context.active_responder_idx >= len(context.active_responders):
        # 所有真实响应者都已处理
        return self._select_best_response(context)

    # 获取当前真实响应者
    current_responder = context.active_responders[context.active_responder_idx]

    # 处理响应
    if action == 'auto':
        response_action = MahjongAction(ActionType.PASS, -1)
    else:
        response_action = action

    # 验证动作有效性
    if not self._is_action_valid(context, current_responder, response_action):
        response_action = MahjongAction(ActionType.PASS, -1)

    # 添加到响应收集器
    priority = self._get_action_priority(response_action.action_type)
    context.response_collector.add_response(
        current_responder,
        response_action.action_type,
        priority,
        response_action.parameter
    )

    # 移动到下一个真实响应者
    context.active_responder_idx += 1

    # 检查是否还有待处理的响应者
    if context.active_responder_idx >= len(context.active_responders):
        # 所有真实响应者处理完毕
        return self._select_best_response(context)

    # 为下一个真实响应者生成观测
    next_responder = context.active_responders[context.active_responder_idx]
    context.current_player_idx = next_responder
    # 立即生成观测和动作掩码
    self.build_observation(context)

    return GameStateType.WAITING_RESPONSE
```

**步骤 2: 运行测试验证**

运行: `python test_four_ai.py` （运行10局游戏测试）
预期: 游戏正常运行，响应阶段正确处理

**步骤 3: 提交**

```bash
git add src/mahjong_rl/state_machine/states/wait_response_state.py
git commit -m "feat: update step() to only process active_responders

修改 step() 方法：
- 只处理 active_responders 中的玩家
- 自动 PASS 的玩家已在 enter() 时处理
- 处理完所有响应者后调用 _select_best_response()
- 为下一个真实响应者生成观测

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### 任务 5: 集成测试和验证

**文件:**
- 创建: `tests/test_auto_pass_optimization.py`（可选）

**步骤 1: 创建集成测试**

创建测试文件 `tests/test_auto_pass_optimization.py`：

```python
"""
测试 WaitResponseState 自动 PASS 优化功能
"""
import pytest
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.state_machine.states.wait_response_state import WaitResponseState
from src.mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder


def test_all_pass_auto():
    """测试所有人只能 PASS 的场景"""
    # TODO: 设置一个所有人只能 PASS 的场景
    # 验证：active_responders 为空
    # 验证：状态直接转换到 DRAWING
    pass


def test_partial_responders():
    """测试部分玩家可以响应的场景"""
    # TODO: 设置场景：3个玩家中只有1个能碰牌
    # 验证：active_responders 只有1个玩家
    # 验证：其他2个玩家自动 PASS
    pass


def test_response_order_preserved():
    """测试响应顺序不被改变"""
    # TODO: 验证 active_responders 保持原始 response_order 的相对顺序
    pass
```

**步骤 2: 运行测试**

运行: `python -m pytest tests/test_auto_pass_optimization.py -v`

**步骤 3: 提交**

```bash
git add tests/test_auto_pass_optimization.py
git commit -m "test: add auto-pass optimization integration tests

添加集成测试：
- 所有人只能 PASS 的场景
- 部分玩家可以响应的场景
- 响应顺序保持正确

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### 任务 6: 更新文档

**文件:**
- 修改: 相关 README 或文档（可选）

**步骤 1: 更新相关文档**

如果项目中维护了状态机或流程的文档，更新相关说明以反映自动 PASS 优化。

**步骤 2: 提交**

```bash
git add <相关文档文件>
git commit -m "docs: update documentation for auto-pass optimization"
```

---

## 测试验证清单

在完成所有任务后，运行以下测试进行验证：

- [ ] **单元测试**: `python -m pytest tests/ -v`
- [ ] **集成测试**: `python test_four_ai.py`（运行10局，观察是否有异常）
- [ ] **手动测试**: 使用 `example_mahjong_env.py` 进行手动对局测试
- [ ] **观察点检查**:
  - 所有人只能 PASS 时，是否直接进入 DRAWING 状态
  - 部分玩家可响应时，active_responders 是否正确
  - 自动 PASS 的玩家是否不需要输入
  - action_mask 是否正确

---

## 预期结果

**性能提升**:
- 训练时间步减少约 25%
- 无意义时间步比例从 ~70% 降至 ~40%

**用户体验提升**:
- 减少 30% 的等待输入操作
- 游戏流程更流畅

---

## 注意事项

1. **向后兼容**: 此修改不改变游戏逻辑，只优化流程
2. **状态一致性**: 确保 active_responders 在状态转换时正确清理
3. **响应优先级**: 保持原有的优先级逻辑（胡 > 杠 > 碰 > 吃）
4. **边界情况**: 处理所有人都自动 PASS 的情况

---

## 回滚计划

如果发现问题需要回滚：
```bash
git revert <commit-hash>  # 回滚特定提交
# 或
git reset --hard HEAD~1  # 回滚到上一个提交
```

---

**总代码变更量**: 约 100 行（包括测试）
**预计实施时间**: 30-45 分钟
