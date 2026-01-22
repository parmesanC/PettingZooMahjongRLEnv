# 设计文档：重载 agent_iter() 实现麻将玩家轮转规则

## 文档信息

| 项目 | 内容 |
|------|------|
| **目标** | 重载 PettingZoo AECEnv 的 `agent_iter()` 方法，实现符合麻将规则的玩家轮转 |
| **日期** | 2026-01-22 |
| **范围** | 单局内轮转（不涉及跨局轮庄） |
| **兼容性** | 严格兼容 PettingZoo AECEnv 规范 |

---

## 1. 问题分析

### 1.1 当前问题

PettingZoo 的 `AECEnv` 基类提供了默认的 `agent_iter()` 实现：

```python
# PettingZoo 默认实现
def agent_iter(self, num_steps=0):
    if num_steps == 0:
        yield from self.agents  # 按 agents 列表顺序迭代
```

这种实现适用于大多数回合制游戏（如国际象棋、围棋），但**不适用于麻游戏**，因为：

| 游戏类型 | 玩家轮转规则 | 是否适合默认 agent_iter() |
|----------|--------------|---------------------------|
| 国际象棋 | 1→2→1→2 固定交替 | ✅ 适合 |
| 围棋 | 1→2→1→2 固定交替 | ✅ 适合 |
| 麻将 | 根据碰/吃/杠动态变化 | ❌ 不适合 |

### 1.2 麻将的特殊轮转规则

麻将的玩家轮转由游戏状态决定，而非固定顺序：

```
正常出牌:  玩家0 → 玩家1 → 玩家2 → 玩家3 → 玩家0 ...

碰牌后:    玩家0打牌 → 玩家2碰牌 → 玩家2继续出牌（跳过玩家1）
              ↓
          轮转顺序改变

等待响应:  玩家0打牌 → 询问玩家1 → 询问玩家2 → 询问玩家3
              ↓              ↓              ↓              ↓
          同一状态内多个玩家依次响应
```

### 1.3 WAITING_RESPONSE 状态的特殊性

`WaitResponseState` 采用**单步收集模式**：

```python
# WaitResponseState.step() 的行为
def step(self, context, action):
    # 1. 处理当前响应者的动作
    # 2. 移动到下一个响应者
    context.move_to_next_responder()
    context.current_player_idx = next_responder

    # 3. 返回 WAITING_RESPONSE（保持同一状态）
    return GameStateType.WAITING_RESPONSE
```

这意味着在**同一个 `WAITING_RESPONSE` 状态**内：
- `current_player_idx` 会逐个指向不同的响应者
- `agent_iter()` 需要依次产生每个响应者
- 所有玩家响应完成后，才转换到下一个状态

---

## 2. 解决方案设计

### 2.1 设计概述

**核心设计理念**：将玩家轮转逻辑的职责分离

| 组件 | 职责 |
|------|------|
| `agent_iter()` | 简单产生当前的 `agent_selection`，不决定轮转顺序 |
| `WuhanMahjongEnv.step()` | 执行动作后更新 `agent_selection` |
| `MahjongStateMachine` | 根据 `current_player_idx` 确定当前玩家 |

**关键决策**：
- `agent_iter()` **不实现**任何轮转逻辑，只作为 `agent_selection` 的暴露接口
- 玩家轮转完全由**状态机 + step() 方法**配合完成
- 这种设计符合 PettingZoo AECEnv 的"当前 agent 驱动"模式

### 2.2 架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                         游戏循环流程                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   for agent in env.agent_iter():                  │
│       obs, reward, terminated, truncated, info = env.last()         │
│       action = agent.get_action(obs)                                │
│       env.step(action)                                              │
│       #                                                               │
│       # ┌─────────────────────────────────────────────────────────┐ │
│       # │  step() 内部流程:                                        │ │
│       # │  1. 执行状态机 step()                                    │ │
│       # │  2. 状态机更新 current_player_idx                        │ │
│       # │  3. agent_selection = state_machine.get_current_agent()  │ │
│       # └─────────────────────────────────────────────────────────┘ │
│       #                                                               │
│       if terminated: break                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

                    ↓ 状态机内部流程 ↓

┌─────────────────────────────────────────────────────────────────────┐
│                     WAITING_RESPONSE 状态流程                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  玩家0打牌 → 进入 WAITING_RESPONSE                                   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 迭代1: agent_iter() → "player_1"                             │    │
│  │        step() → 状态机处理响应 → current_player_idx = 2       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 迭代2: agent_iter() → "player_2"                             │    │
│  │        step() → 状态机处理响应 → current_player_idx = 3       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 迭代3: agent_iter() → "player_3"                             │    │
│  │        step() → 状态机处理响应 → 所有玩家响应完成              │    │
│  │        → 转换到 PROCESSING_MELD 或 DRAWING 状态                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 详细实现

### 3.1 agent_iter() 实现

```python
# 在 WuhanMahjongEnv 中添加/修改

def agent_iter(self, num_steps: int = 0):
    """
    重载 PettingZoo 的 agent_iter 方法

    麻将的玩家轮转由状态机控制，而不是简单的列表迭代。
    该方法只产生当前 agent_selection 指向的玩家，玩家轮转
    完全由 step() 方法和状态机配合完成。

    Args:
        num_steps: 最大迭代步数（0 表示无限迭代直到游戏结束）

    Yields:
        当前需要动作的 agent 名称

    设计说明：
        - 游戏进行中：产生 agent_selection（由状态机根据
          current_player_idx 确定）
        - WAITING_RESPONSE 状态：状态机逐个更新响应者，
          agent_iter 对应产生每个响应者
        - 游戏结束：agents 列表为空，迭代自动终止

    PettingZoo 兼容性：
        完全兼容 AECEnv 规范，可以用于标准游戏循环：

        for agent in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            if terminated or truncated:
                action = None
            else:
                action = policy(obs)
            env.step(action)
    """
    if num_steps == 0:
        # 无限迭代模式：迭代直到 agents 列表为空
        while self.agents:
            yield self.agent_selection
    else:
        # 限制步数模式：最多产生 num_steps 个 agent
        for _ in range(num_steps):
            if not self.agents:
                break
            yield self.agent_selection
```

### 3.2 step() 方法的更新点

现有 `step()` 方法已经正确处理了 `agent_selection` 的更新，无需修改。确认关键代码：

```python
# example_mahjong_env.py:239-241 (现有代码已正确)
# 更新agent_selection
if not self.state_machine.is_terminal():
    self.agent_selection = self.state_machine.get_current_agent()
```

### 3.3 状态机的角色

状态机中的现有方法已满足需求，**无需修改**：

```python
# src/mahjong_rl/state_machine/machine.py

def get_current_agent(self) -> str:
    """获取当前agent名称（用于AECEnv的agent_selection）"""
    return f"player_{self.get_current_player_id()}"

def get_current_player_id(self) -> int:
    """获取当前玩家ID"""
    if self.context is None:
        return -1
    return self.context.current_player_idx
```

**WaitResponseState.step()** 已经正确实现了响应者轮转：

```python
# src/mahjong_rl/state_machine/states/wait_response_state.py:126-140

# 移动到下一个响应者
context.move_to_next_responder()

# 为下一个响应者生成观测
next_responder = context.get_current_responder()
if next_responder is not None:
    # 更新当前玩家索引到下一个响应者
    context.current_player_idx = next_responder
    # 立即生成观测和动作掩码
    self.build_observation(context)

return GameStateType.WAITING_RESPONSE  # 继续保持同一状态
```

---

## 4. 测试计划

### 4.1 单元测试

创建 `test_agent_iter.py` 文件：

```python
"""测试 agent_iter() 的实现"""

import pytest
from example_mahjong_env import WuhanMahjongEnv


def test_agent_iter_produces_current_selection():
    """测试 agent_iter() 产生当前的 agent_selection"""
    env = WuhanMahjongEnv()
    obs, info = env.reset()

    # 第一次迭代应该产生初始 agent_selection
    agents = list(env.agent_iter(num_steps=1))
    assert len(agents) == 1
    assert agents[0] == env.agent_selection


def test_agent_iter_updates_after_step():
    """测试 step() 后 agent_iter() 产生更新的 agent"""
    env = WuhanMahjongEnv()
    obs, info = env.reset()
    initial_agent = env.agent_selection

    # 执行一步
    action = (0, 0)  # 简化动作
    obs, reward, terminated, truncated, info = env.step(action)

    # agent_iter 应该产生新的 agent
    next_agent = next(env.agent_iter(num_steps=1))
    assert next_agent == env.agent_selection


def test_agent_iter_terminates_when_game_ends():
    """测试游戏结束时 agent_iter() 终止"""
    env = WuhanMahjongEnv()
    obs, info = env.reset()

    # 模拟游戏直到结束
    agent_count = 0
    for agent in env.agent_iter():
        agent_count += 1
        obs, reward, terminated, truncated, info = env.last()
        if terminated:
            action = None
        else:
            action = (0, 0)
        env.step(action)

        if terminated or truncated or agent_count > 1000:
            break

    # 游戏结束后 agents 列表应该为空
    assert len(env.agents) == 0


def test_agent_iter_in_waiting_response():
    """测试 WAITING_RESPONSE 状态下的 agent_iter()"""
    env = WuhanMahjongEnv()
    obs, info = env.reset()

    # 手动设置到 WAITING_RESPONSE 状态
    # （需要实际游戏流程触发，这里简化说明）
    # 验证在该状态下，agent_iter() 依次产生每个响应者

    # 实际测试需要模拟真实的麻将游戏流程
    # 这里只是说明测试思路
    pass
```

### 4.2 集成测试

测试完整的游戏循环：

```python
def test_full_game_loop_with_agent_iter():
    """测试完整的游戏循环"""
    env = WuhanMahjongEnv(render_mode='human')
    obs, info = env.reset(seed=42)

    step_count = 0
    max_steps = 500  # 防止无限循环

    for agent in env.agent_iter():
        step_count += 1

        # 获取观测和奖励
        obs, reward, terminated, truncated, info = env.last()

        # 验证 agent 与 agent_selection 一致
        assert agent == env.agent_selection

        if terminated or truncated:
            action = None
        else:
            # 简化：随机选择动作
            import random
            action = (random.randint(0, 10), random.randint(0, 34))

        # 执行动作
        env.step(action)

        if step_count > max_steps:
            break

    print(f"游戏完成，共 {step_count} 步")
    assert step_count > 0
```

### 4.3 验证步骤

| 步骤 | 操作 | 预期结果 |
|------|------|----------|
| 1 | 实现 `agent_iter()` 方法 | 编译通过 |
| 2 | 运行单元测试 | 所有测试通过 |
| 3 | 运行集成测试 | 游戏正常完成 |
| 4 | 运行现有测试 | 无回归问题 |
| 5 | 手动测试（human_vs_ai） | 玩家轮转符合预期 |

---

## 5. 实施计划

### 5.1 文件修改清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `example_mahjong_env.py` | 添加 `agent_iter()` 方法 | 重载 PettingZoo 方法 |
| `test_agent_iter.py` | 新建 | 单元测试 |
| `docs/plans/2026-01-22-agent-iter-redesign.md` | 新建 | 本设计文档 |

### 5.2 实施步骤

**步骤 1**：添加 `agent_iter()` 方法

在 `WuhanMahjongEnv` 类中添加 `agent_iter()` 方法（参考 3.1 节）。

**步骤 2**：创建单元测试

创建 `test_agent_iter.py` 文件，实现基础测试用例。

**步骤 3**：运行测试

```bash
# 运行单元测试
python -m pytest test_agent_iter.py -v

# 运行集成测试
python -c "
from example_mahjong_env import WuhanMahjongEnv
env = WuhanMahjongEnv()
obs, info = env.reset()

step_count = 0
for agent in env.agent_iter():
    step_count += 1
    obs, reward, terminated, truncated, info = env.last()
    if terminated or truncated:
        action = None
    else:
        import random
        action = (random.randint(0, 10), random.randint(0, 34))
    env.step(action)
    if terminated or step_count > 100:
        break

print(f'测试完成，共 {step_count} 步')
"
```

**步骤 4**：验证现有功能

确保现有测试和游戏模式仍然正常工作：

```bash
# 测试四人 AI 游戏
python test_four_ai.py

# 测试人机对战
python play_mahjong.py --mode human_vs_ai --renderer cli
```

**步骤 5**：提交代码

```bash
git add example_mahjong_env.py test_agent_iter.py docs/plans/2026-01-22-agent-iter-redesign.md
git commit -m "feat(env): override agent_iter() for mahjong player rotation

- Override agent_iter() to yield agent_selection instead of iterating agents list
- Player rotation is controlled by state machine + step() method
- Properly handles WAITING_RESPONSE state with multiple responders
- Fully compatible with PettingZoo AECEnv specification"
```

---

## 6. 总结

### 6.1 设计要点

| 要点 | 说明 |
|------|------|
| **职责分离** | `agent_iter()` 只产生 `agent_selection`，轮转逻辑由状态机控制 |
| **PettingZoo 兼容** | 完全兼容 AECEnv 规范，支持标准游戏循环 |
| **状态驱动** | 玩家轮转由 `current_player_idx` 驱动，而非固定顺序 |
| **响应处理** | `WAITING_RESPONSE` 状态下，逐个产生每个响应者 |

### 6.2 与现有代码的关系

- **无需修改**：状态机、`WaitResponseState`、`GameContext`
- **只需添加**：`WuhanMahjongEnv.agent_iter()` 方法
- **风险较低**：改动范围小，不影响现有逻辑

### 6.3 后续优化（可选）

- 添加调试日志：记录 `agent_selection` 的变化
- 性能优化：如果需要，可以缓存 `get_current_agent()` 结果
- 扩展功能：支持跨局轮庄逻辑（如果需要）
