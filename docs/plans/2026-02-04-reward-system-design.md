# Reward System with Actual Scoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现基于武汉麻将实际计分规则的PettingZoo AEC环境reward系统，确保游戏结束时所有4个玩家都能获得正确的reward。

**Architecture:**
1. 在 `GameContext` 中添加 `final_scores` 字段存储游戏结束时所有玩家的得分
2. 在 `WinState` 和 `FlushState` 中计算并保存分数到 `context.final_scores`
3. 在环境 `step()` 方法中使用 `context.final_scores` 为所有agents分配reward

**Tech Stack:**
- PettingZoo AECEnv
- MahjongScoreSettler（武汉麻将计分器）
- 武汉麻将七皮四赖子规则

---

## Task 1: 添加 final_scores 字段到 GameContext

**Files:**
- Modify: `src/mahjong_rl/core/GameData.py`

**Step 1: 修改 GameContext 类添加 final_scores 字段**

在 `GameContext` 类中添加 `final_scores` 字段：

```python
@dataclass
class GameContext:
    """游戏上下文 - 存储游戏的所有状态信息"""
    # ... 现有字段 ...
    final_scores: List[float] = field(default_factory=list)  # 存储所有玩家的最终得分（长度为4）
```

**Step 2: 验证修改**

运行: `python -c "from src.mahjong_rl.core.GameData import GameContext; ctx = GameContext(); print(hasattr(ctx, 'final_scores'))"`
Expected: `True`

**Step 3: 提交**

```bash
git add src/mahjong_rl/core/GameData.py
git commit -m "feat(game_data): add final_scores field to GameContext"
```

---

## Task 2: 修改 WinState 保存完整分数列表

**Files:**
- Modify: `src/mahjong_rl/state_machine/states/win_state.py:90-112`

**Step 1: 修改 _calculate_scores 方法**

将完整分数列表保存到 `context.final_scores`：

```python
def _calculate_scores(self, context: GameContext):
    """
    计算分数

    使用WuhanMahjongWinChecker和MahjongScoreSettler
    计算每个玩家的胡牌分数。

    Args:
        context: 游戏上下文
    """
    win_checker = WuhanMahjongWinChecker(context)
    score_calculator = MahjongScoreSettler(False)

    # 计算胡牌玩家的分数
    for winner_id in context.winner_ids:
        player = context.players[winner_id]
        win_result = win_checker.check_win(player)

        if win_result.can_win:
            # 计算分数
            score_list = score_calculator.settle(win_result, context)
            # 保存完整分数列表到 context（新增）
            context.final_scores = score_list
            # 同时保留赢家得分（兼容旧逻辑）
            player.fan_count = max(score_list)
```

**Step 2: 运行状态机测试验证**

运行: `python tests/unit/test_state_machine.py`
Expected: 所有测试通过

**Step 3: 提交**

```bash
git add src/mahjong_rl/state_machine/states/win_state.py
git commit -m "feat(win_state): save complete score list to context.final_scores"
```

---

## Task 3: 修改 FlushState 添加流局查大叫结算

**Files:**
- Modify: `src/mahjong_rl/state_machine/states/flush_state.py:45-64`

**Step 1: 在 FlushState 中添加 _calculate_flow_draw_scores 方法**

在类中添加新方法并修改 `enter` 方法：

```python
from typing import Optional

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType
from src.mahjong_rl.observation.builder import IObservationBuilder
from src.mahjong_rl.rules.base import IRuleEngine
from src.mahjong_rl.state_machine.base import GameState


class FlushState(GameState):
    """
    荒牌流局状态

    处理牌墙耗尽时的流局逻辑。
    根据武汉麻将规则，流局时庄家保留。
    这是终端状态，执行step()后不再转换状态。

    Attributes:
        rule_engine: 规则引擎实例
        observation_builder: 观测构建器实例
    """

    def __init__(self, rule_engine: IRuleEngine, observation_builder: IObservationBuilder):
        """
        初始化荒牌状态

        Args:
            rule_engine: 规则引擎实例
            observation_builder: 观测构建器实例
        """
        super().__init__(rule_engine, observation_builder)

    def enter(self, context: GameContext) -> None:
        """
        进入荒牌状态

        标记游戏为荒牌，并设置庄家连庄规则。
        终端状态不需要生成观测和动作掩码。

        Args:
            context: 游戏上下文
        """
        context.current_state = GameStateType.FLOW_DRAW
        context.is_flush = True
        # 终端状态不需要生成观测和动作掩码
        context.observation = None
        context.action_mask = None

        # 计算流局查大叫分数（新增）
        self._calculate_flow_draw_scores(context)

    def step(self, context: GameContext, action: Optional[str] = None) -> Optional[GameStateType]:
        """
        荒牌状态不执行任何动作

        这是终端状态，step()方法返回None表示游戏结束。

        Args:
            context: 游戏上下文
            action: 忽略

        Returns:
            None (终端状态）
        """
        return None  # 终端状态

    def exit(self, context: GameContext) -> None:
        """
        离开荒牌状态

        Args:
            context: 游戏上下文
        """
        pass

    def _calculate_flow_draw_scores(self, context: GameContext):
        """
        计算流局时的查大叫分数

        根据武汉麻将规则，流局时听牌者可以向未听牌者收取分数。

        Args:
            context: 游戏上下文
        """
        from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.score_calculator import MahjongScoreSettler

        score_settler = MahjongScoreSettler(is_kou_kou_fan=True)
        context.final_scores = score_settler.settle_flow_draw(context)
```

**Step 2: 运行状态机测试验证**

运行: `python tests/unit/test_state_machine.py`
Expected: 所有测试通过

**Step 3: 提交**

```bash
git add src/mahjong_rl/state_machine/states/flush_state.py
git commit -m "feat(flush_state): add flow draw score settlement (cha da jiao)"
```

---

## Task 4: 修改环境层使用 final_scores 分配 reward

**Files:**
- Modify: `example_mahjong_env.py:469-500`
- Test: `tests/integration/test_reward_system.py` (新建)

**Step 4.1: 编写测试用例**

创建测试文件验证reward分配：

```python
"""
测试reward系统 - 验证游戏结束时所有玩家都能获得正确的reward
"""
import numpy as np
from example_mahjong_env import WuhanMahjongEnv


def test_all_players_get_reward_on_win():
    """测试胡牌时所有4个玩家都能获得reward"""
    env = WuhanMahjongEnv(render_mode=None, training_phase=3, enable_logging=False)
    env.reset(seed=42)

    # 运行游戏直到结束（使用简单的随机策略）
    terminated = False
    step_count = 0
    max_steps = 1000

    while not terminated and step_count < max_steps:
        current_agent = env.agent_selection
        obs = env.observe(current_agent)
        action_mask = obs['action_mask']

        # 找到第一个可用动作
        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            break

        action_idx = valid_actions[0]

        # 将action_idx转换为(action_type, parameter)
        if action_idx < 34:  # DISCARD
            action = (0, action_idx)
        elif action_idx == 143:  # WIN
            action = (10, -1)
        elif action_idx == 144:  # PASS
            action = (11, -1)
        else:
            action = (0, 0)  # 默认DISCARD第一张

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

    # 验证：游戏应该结束
    assert terminated or truncated, "游戏应该在最大步数内结束"

    # 验证：所有4个玩家都应该有reward
    for agent in env.possible_agents:
        assert agent in env.rewards, f"{agent} 应该有reward"
        # reward不应该全是初始值0（除非流局且没有人听牌）
        # 这里我们只验证key存在

    # 验证：如果是胡牌，rewards总和应该为0（零和）
    if env.context.is_win and env.context.final_scores:
        reward_sum = sum(env.rewards.values())
        assert abs(reward_sum) < 0.01, f"胡牌时reward总和应为0，实际为{reward_sum}"

    print(f"游戏在 {step_count} 步后结束")
    print(f"最终rewards: {env.rewards}")
    if env.context.final_scores:
        print(f"原始分数: {env.context.final_scores}")

    env.close()


def test_reward_matches_final_scores():
    """测试reward应该等于final_scores除以100"""
    env = WuhanMahjongEnv(render_mode=None, training_phase=3, enable_logging=False)
    env.reset(seed=123)

    # 运行游戏直到结束
    terminated = False
    step_count = 0
    max_steps = 1000

    while not terminated and step_count < max_steps:
        current_agent = env.agent_selection
        obs = env.observe(current_agent)
        action_mask = obs['action_mask']

        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            break

        action_idx = valid_actions[0]

        if action_idx < 34:
            action = (0, action_idx)
        elif action_idx == 143:
            action = (10, -1)
        elif action_idx == 144:
            action = (11, -1)
        else:
            action = (0, 0)

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

    # 验证：如果有final_scores，reward应该匹配
    if env.context.final_scores:
        for i, score in enumerate(env.context.final_scores):
            agent_name = f"player_{i}"
            expected_reward = score / 100.0
            actual_reward = env.rewards.get(agent_name, 0.0)
            assert abs(actual_reward - expected_reward) < 0.01, \
                f"{agent_name} reward不匹配: 期望{expected_reward}, 实际{actual_reward}"

    env.close()


if __name__ == "__main__":
    print("测试1: 验证所有玩家都能获得reward")
    test_all_players_get_reward_on_win()
    print("✓ 测试1通过\n")

    print("测试2: 验证reward与final_scores匹配")
    test_reward_matches_final_scores()
    print("✓ 测试2通过\n")

    print("所有测试通过！")
```

**Step 4.2: 运行测试验证当前实现**

运行: `python tests/integration/test_reward_system.py`
Expected: 测试**失败**（因为还没有修改环境）

**Step 4.3: 修改环境 step() 方法**

在 `example_mahjong_env.py` 中修改 `step()` 方法的游戏结束处理部分：

```python
# 检查是否结束
terminated = self.state_machine.is_terminal()
truncated = False
info = self._get_info(agent_idx)

# 游戏结束时更新 round_info 并记录结果
if terminated:
    self._update_round_info()

    # 计算所有玩家的最终得分作为 reward（新增）
    if self.context.final_scores:
        # 将实际分数归一化为 reward（除以100使范围在 [-5, +5]）
        for i, score in enumerate(self.context.final_scores):
            agent_name = f"player_{i}"
            self.rewards[agent_name] = score / 100.0
    else:
        # 兼容：如果没有final_scores，使用旧的简化逻辑
        if self.context.is_win:
            for agent in self.possible_agents:
                agent_idx = self.agents_name_mapping[agent]
                if agent_idx in self.context.winner_ids:
                    self.rewards[agent] = 1.0
                else:
                    self.rewards[agent] = -1.0
        else:
            for agent in self.possible_agents:
                self.rewards[agent] = 0.0

    # 记录游戏结束
    if self.logger and not self._game_logged:
        result = {
            "winners": list(self.context.winner_ids) if self.context.is_win else [],
            "is_flush": self.context.is_flush,
            "win_way": self.context.win_way if self.context.is_win else None,
            "total_steps": self._get_total_steps()
        }
        self.logger.end_game(result)
        self._game_logged = True

# PettingZoo AEC规范：终端状态下移除所有agents
if terminated or truncated:
    # 将所有agents标记为终止，并从agents列表中移除
    for agent in self.agents[:]:
        self.terminations[agent] = True
        self.agents.remove(agent)
    self.agent_selection = None  # 清空agent选择

# Accumulate rewards for PettingZoo AECEnv compatibility
self._accumulate_rewards()

return observation, reward, terminated, truncated, info
```

**Step 4.4: 运行测试验证修改**

运行: `python tests/integration/test_reward_system.py`
Expected: 测试**通过**

**Step 4.5: 提交**

```bash
git add example_mahjong_env.py tests/integration/test_reward_system.py
git commit -m "feat(env): use final_scores to distribute rewards to all players"
```

---

## Task 5: 运行完整集成测试

**Files:**
- Test: 所有相关测试

**Step 1: 运行状态机单元测试**

运行: `python tests/unit/test_state_machine.py`
Expected: 所有测试通过

**Step 2: 运行reward系统测试**

运行: `python tests/integration/test_reward_system.py`
Expected: 所有测试通过

**Step 3: 运行其他集成测试确保没有破坏**

运行: `python tests/integration/test_rob_kong.py`
运行: `python tests/integration/test_auto_skip_state.py`

**Step 4: 最终提交**

如果所有测试通过，创建总结提交：

```bash
git add .
git commit -m "feat(reward): implement actual scoring-based reward system for all players

- Add final_scores field to GameContext
- WinState saves complete score list
- FlushState implements flow draw settlement (cha da jiao)
- Environment distributes rewards to all 4 players on game end
- Add integration tests for reward system"
```

---

## 完成检查清单

- [ ] `GameContext.final_scores` 字段已添加
- [ ] `WinState` 保存完整分数列表
- [ ] `FlushState` 实现流局查大叫结算
- [ ] 环境使用 `final_scores` 分配reward给所有玩家
- [ ] 所有单元测试通过
- [ ] 所有集成测试通过
- [ ] 文档已更新（如需要）

---

## 相关文档

- 武汉麻将规则: `src/mahjong_rl/rules/wuhan_mahjong_rule_engine/wuhan_mahjong_rules.md`
- PettingZoo AEC文档: https://pettingzoo.farama.org/api/aec/
- 计分器实现: `src/mahjong_rl/rules/wuhan_mahjong_rule_engine/score_calculator.py`
