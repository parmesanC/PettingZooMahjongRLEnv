# Mahjong State Machine 快速入门指南

## 5分钟快速开始

### 1. 创建状态机

```python
from src.mahjong_rl.state_machine.machine import MahjongStateMachine
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder

# 创建游戏上下文
context = GameContext.create_new_round(seed=42)

# 初始化规则引擎和观测构建器
rule_engine = Wuhan7P4LRuleEngine(context)
observation_builder = Wuhan7P4LObservationBuilder(context)

# 创建状态机
state_machine = MahjongStateMachine(
    rule_engine=rule_engine,
    observation_builder=observation_builder,
    enable_logging=True  # 启用日志
)

# 设置上下文
state_machine.set_context(context)
```

### 2. 启动游戏

```python
from src.mahjong_rl.core.constants import GameStateType

# 转换到初始状态
state_machine.transition_to(GameStateType.INITIAL, context)

# 执行初始状态（自动）
state_machine.step(context, 'auto')

# 检查当前状态
print(f"当前状态: {state_machine.current_state_type.name}")
print(f"当前玩家: {state_machine.get_current_player_id()}")
```

### 3. 执行游戏步骤

```python
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.core.constants import ActionType

# 游戏主循环
while not state_machine.is_terminal():
    current_state = state_machine.current_state_type
    
    # 检查是否需要agent动作
    if current_state == GameStateType.PLAYER_DECISION:
        # 获取观测
        observation = observation_builder.build(
            context, 
            state_machine.get_current_player_id()
        )
        
        # agent选择动作（这里简化为随机）
        available_actions = rule_engine.detect_available_actions_after_draw(
            context.players[context.current_player_idx],
            context.players[context.current_player_idx].hand_tiles[-1]
        )
        if available_actions:
            action = available_actions[0]  # 简化：选择第一个可用动作
        else:
            # 默认打第一张牌
            action = MahjongAction(
                ActionType.DISCARD,
                context.players[context.current_player_idx].hand_tiles[0]
            )
        
        # 执行动作
        next_state = state_machine.step(context, action)
        print(f"动作: {action.action_type.name} -> {next_state.name}")
    
    elif current_state == GameStateType.DISCARDING:
        # 玩家需要打牌（简化为自动打第一张）
        current_player = context.players[context.current_player_idx]
        if len(current_player.hand_tiles) > 0:
            action = MahjongAction(
                ActionType.DISCARD,
                current_player.hand_tiles[0]
            )
            next_state = state_machine.step(context, action)
            print(f"打牌: {action.parameter} -> {next_state.name}")
        else:
            break
    
    else:
        # 自动状态
        next_state = state_machine.step(context, 'auto')
        if next_state:
            print(f"自动推进: {current_state.name} -> {next_state.name}")
    
    # 检查游戏结束
    if state_machine.is_terminal():
        print("游戏结束！")
        if context.is_win:
            print(f"获胜者: {context.winner_ids}")
            print(f"胡牌方式: {context.win_way}")
        elif context.is_flush:
            print("流局")
        break

    # 防止无限循环（安全退出）
    if len(context.action_history) > 100:
        print("达到最大步数，退出")
        break
```

### 4. 查看日志

```python
logger = state_machine.get_logger()
if logger:
    history = logger.get_history()
    print(f"\n日志记录: {len(history)}条")
    
    # 显示最后10条日志
    print("\n最近日志:")
    for log_entry in history[-10:]:
        log_type = log_entry['type']
        if log_type == 'transition':
            print(f"  [转换] {log_entry['from_state']} -> {log_entry['to_state']}")
        elif log_type == 'action':
            print(f"  [动作] 玩家{log_entry['player_id']}: {log_entry['action_type']}")
        elif log_type == 'log':
            print(f"  [日志] {log_entry['message']}")
        elif log_type == 'error':
            print(f"  [错误] {log_entry['message']}")
```

### 5. 状态回滚

```python
# 保存当前状态（自动保存）
state_machine.step(context, action)

# 回滚1步
context_rolled = state_machine.rollback(1)
print(f"回滚后状态: {state_machine.current_state_type.name}")

# 回滚多步
# context_rolled = state_machine.rollback(5)

# 获取历史
history = state_machine.get_history()
print(f"状态历史: {len(history)}个快照")
```

## PettingZoo集成示例

```python
from example_mahjong_env import WuhanMahjongEnv

# 创建环境
env = WuhanMahjongEnv(training_phase=3)

# 重置环境
observation, info = env.reset(seed=42)
print(f"初始agent: {env.agent_selection}")

# 游戏主循环
for agent in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    
    print(f"\n当前agent: {agent}")
    print(f"奖励: {reward}")
    print(f"是否结束: {terminated}")
    
    if terminated or truncated:
        action = None
    else:
        # 简化：随机选择动作
        import random
        action = random.randint(0, 10), random.randint(0, 34)
    
    # 执行动作
    env.step(action)
    
    if terminated:
        print("\n游戏结束！")
        break

# 关闭环境
env.close()
```

## 常见问题

### Q: 如何知道当前需要什么动作？
```python
current_state = state_machine.current_state_type

if current_state in [GameStateType.PLAYER_DECISION, GameStateType.DISCARDING]:
    # 需要agent动作
    action = agent.get_action(observation)
else:
    # 自动状态，使用'auto'
    action = 'auto'
```

### Q: 如何获取可用动作？
```python
# 对于PLAYER_DECISION状态
if current_state == GameStateType.PLAYER_DECISION:
    player_id = state_machine.get_current_player_id()
    player = context.players[player_id]
    draw_tile = player.hand_tiles[-1]
    actions = rule_engine.detect_available_actions_after_draw(player, draw_tile)

# 对于WAITING_RESPONSE状态
elif current_state == GameStateType.WAITING_RESPONSE:
    player_id = context.get_current_responder()
    player = context.players[player_id]
    discard_tile = context.last_discarded_tile
    discard_player = context.discard_player
    actions = rule_engine.detect_available_actions_after_discard(
        player, discard_tile, discard_player
    )
```

### Q: 如何处理武汉麻将特殊杠？
```python
# 特殊杠在GongState中自动处理
# 只需要在PLAYER_DECISION状态选择相应的动作类型

# 红中杠
action = MahjongAction(ActionType.KONG_RED, 31)

# 皮子杠
action = MahjongAction(ActionType.KONG_SKIN, skin_tile)

# 赖子杠
action = MahjongAction(ActionType.KONG_LAZY, lazy_tile)

# 补杠
action = MahjongAction(ActionType.KONG_SUPPLEMENT, tile)

# 暗杠
action = MahjongAction(ActionType.KONG_CONCEALED, tile)
```

### Q: 如何检查游戏结束原因？
```python
if state_machine.is_terminal():
    if context.is_win:
        if context.win_way == 0:  # 自摸
            print("自摸胡牌")
        elif context.win_way == 1:  # 抢杠
            print("抢杠胡牌")
        elif context.win_way == 2:  # 杠开
            print("杠上开花")
        elif context.win_way == 3:  # 点炮
            print("点炮胡牌")
        print(f"获胜者: {context.winner_ids}")
    elif context.is_flush:
        print("流局（牌墙耗尽）")
```

## 下一步

1. 阅读完整文档：`src/mahjong_rl/state_machine/README.md`
2. 运行测试：`python test_state_machine.py`
3. 查看集成示例：`python example_mahjong_env.py`
4. 查看实施总结：`STATE_MACHINE_IMPLEMENTATION_SUMMARY.md`

## 支持的功能

✅ 完整的状态转换
✅ 自动/手动状态分类
✅ 状态回滚功能
✅ 详细日志记录
✅ PettingZoo AECEnv集成
✅ 武汉麻将所有特殊规则
✅ 单步响应收集
✅ 懒加载观测
✅ 动作验证
✅ 自动PASS优化（减少25%时间步）

祝你使用愉快！🎉
