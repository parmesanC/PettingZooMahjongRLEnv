# Mahjong State Machine

完整的麻将强化学习环境状态机实现，用于管理武汉麻将七皮四赖子游戏的所有游戏阶段。

## 文件结构

```
src/mahjong_rl/state_machine/
├── __init__.py                      # 状态机导出
├── machine.py                      # MahjongStateMachine核心实现
└── states/
    ├── __init__.py                  # 状态类导出
    ├── base.py                      # GameState基类
    ├── initial_state.py             # 初始状态
    ├── drawing_state.py            # 摸牌状态
    ├── discarding_state.py         # 出牌状态
    ├── player_decision_state.py    # 玩家决策状态
    ├── wait_response_state.py     # 等待响应状态
    ├── process_meld_state.py      # 处理鸣牌状态
    ├── gong_state.py              # 杠牌状态
    ├── drawing_after_gong_state.py # 杠后补牌状态
    ├── win_state.py              # 和牌状态（终端）
    └── flush_state.py            # 荒牌流局状态（终端）
```

## 状态分类

### 手动状态（需要agent动作）
- **PLAYER_DECISION**: 玩家决策状态，处理玩家选择出牌、杠牌或和牌
- **DISCARDING**: 出牌状态，处理玩家打出一张牌

### 自动状态（自动推进）
- **DRAWING**: 摸牌状态，自动从牌墙抽取一张牌
- **WAITING_RESPONSE**: 等待响应状态，逐个收集其他玩家的响应（单步模式）
- **PROCESSING_MELD**: 处理鸣牌状态，处理吃或碰牌操作
- **GONG**: 杠牌状态，处理所有类型的杠牌（明杠、暗杠、补杠、红中杠、皮子杠、赖子杠）
- **DRAWING_AFTER_GONG**: 杠后补牌状态，检查抢杠和和杠开

### 终端状态（游戏结束）
- **WIN**: 和牌状态，游戏胜利结束
- **FLOW_DRAW**: 荒牌流局状态，牌墙耗尽

## 使用方法

### 基本初始化

```python
from src.mahjong_rl.state_machine.machine import MahjongStateMachine
from src.mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder
from src.mahjong_rl.core.GameData import GameContext

# 创建游戏上下文
context = GameContext.create_new_round(seed=42)

# 初始化规则引擎和观测构建器
rule_engine = Wuhan7P4LRuleEngine(context)
observation_builder = Wuhan7P4LObservationBuilder(context)

# 创建状态机
state_machine = MahjongStateMachine(
    rule_engine=rule_engine,
    observation_builder=observation_builder,
    enable_logging=True  # 启用日志记录
)

# 设置游戏上下文
state_machine.set_context(context)

# 转换到初始状态
state_machine.transition_to(GameStateType.INITIAL, context)

# 执行初始状态（自动）
state_machine.step(context, 'auto')
```

### 手动状态（需要agent动作）

```python
# 示例：玩家决策状态
if state_machine.current_state_type == GameStateType.PLAYER_DECISION:
    # 获取当前玩家
    player_id = state_machine.get_current_player_id()
    
    # agent选择动作（这里使用MahjongAction对象）
    action = MahjongAction(
        action_type=ActionType.DISCARD,
        parameter=5  # 打出的牌ID
    )
    
    # 执行动作
    next_state = state_machine.step(context, action)
    print(f"下一个状态: {next_state.name}")
```

### 自动状态（自动推进）

```python
# 示例：摸牌状态
if state_machine.current_state_type == GameStateType.DRAWING:
    # 自动状态使用'auto'标记
    next_state = state_machine.step(context, 'auto')
    print(f"下一个状态: {next_state.name}")
```

### 自动推进循环（PettingZoo集成）

```python
# 自动推进直到手动状态或终端
while not state_machine.is_terminal():
    current_state = state_machine.current_state_type
    
    # 手动状态需要agent动作
    if current_state in [GameStateType.PLAYER_DECISION, GameStateType.DISCARDING]:
        agent_id = state_machine.get_current_agent()  # 'player_0', 'player_1', etc.
        observation = observation_builder.build(context, current_state=state_machine.get_current_player_id())
        
        # 获取agent动作
        action = agent.get_action(observation)
        
        # 执行动作
        next_state = state_machine.step(context, action)
    else:
        # 自动状态
        next_state = state_machine.step(context, 'auto')
    
    if next_state is None:
        # 终端状态
        break
    
    if state_machine.is_terminal():
        # 游戏结束
        break
```

## 状态转换图

```
[START] → INITIAL
INITIAL → PLAYER_DECISION

PLAYER_DECISION → WIN (自摸）
PLAYER_DECISION → DISCARDING (出牌）
PLAYER_DECISION → GONG (杠牌）

DISCARDING → WAITING_RESPONSE (等待其他玩家响应）

WAITING_RESPONSE → WIN (和牌）
WAITING_RESPONSE → GONG (明杠）
WAITING_RESPONSE → PROCESSING_MELD (吃或碰）
WAITING_RESPONSE → DRAWING (所有玩家过）

PROCESSING_MELD → PLAYER_DECISION

GONG → DRAWING_AFTER_GONG

DRAWING_AFTER_GONG → WIN (抢杠和或杠开）
DRAWING_AFTER_GONG → PLAYER_DECISION (正常摸牌）

DRAWING → PLAYER_DECISION (正常摸牌）
DRAWING → FLOW_DRAW (牌墙为空）

WIN → [END]
FLOW_DRAW → [END]
```

## 武汉麻将特殊规则

### 杠牌类型

1. **明杠（KONG_EXPOSED）**: 玩家有3张牌，其他玩家打出第4张
2. **暗杠（KONG_CONCEALED）**: 玩家有4张牌在手中
3. **补杠（KONG_SUPPLEMENT）**: 玩家已有碰牌，摸到第4张
4. **红中杠（KONG_RED）**: 杠红中牌
5. **皮子杠（KONG_SKIN）**: 杠皮子牌
6. **赖子杠（KONG_LAZY）**: 杠赖子牌

### 特殊和牌类型

1. **自摸（SELF_DRAW）**: 摸牌后自己胡牌
2. **点炮（DISCARD）**: 其他玩家打出的牌被胡
3. **杠上开花（KONG_SELF_DRAW）**: 杠后摸牌并胡牌
4. **抢杠和（ROB_KONG）**: 其他玩家补杠时抢杠

## 状态回滚

状态机支持完整的状态快照回滚功能：

```python
# 保存历史（自动保存，最多100个快照）
state_machine.step(context, action)

# 回滚1步
context = state_machine.rollback(1)

# 回滚多步
context = state_machine.rollback(5)

# 获取历史
history = state_machine.get_history()
for snapshot in history:
    print(f"状态: {snapshot['state_type']}, 时间: {snapshot['timestamp']}")

# 清空历史
state_machine.clear_history()
```

## 日志记录

状态机提供详细的日志记录功能：

```python
logger = state_machine.get_logger()
if logger:
    # 获取日志历史
    history = logger.get_history()
    
    # 显示日志
    for log_entry in history:
        log_type = log_entry['type']
        if log_type == 'transition':
            print(f"状态转换: {log_entry['from_state']} -> {log_entry['to_state']}")
        elif log_type == 'action':
            print(f"玩家{log_entry['player_id']}动作: {log_entry['action_type']}")
        elif log_type == 'error':
            print(f"错误: {log_entry['message']}")
    
    # 清空日志
    logger.clear()
```

## 动作掩码和观测

每个状态在enter()时会生成观测和动作掩码：

```python
# 在手动状态中获取观测
if state_machine.current_state_type == GameStateType.PLAYER_DECISION:
    observation = context.observation
    action_mask = context.action_mask
    
    # observation包含:
    # - global_hand: 所有玩家手牌（部分可见）
    # - private_hand: 私有手牌
    # - discard_pool_total: 弃牌堆总量
    # - melds: 副露信息
    # - action_history: 动作历史
    # - special_gangs: 特殊杠数量
    # - current_player: 当前玩家
    # - fan_counts: 番数计数
    # - special_indicators: 特殊牌指示器（赖子、皮子）
    # - remaining_tiles: 剩余牌数
    # - dealer: 庄家
    # - action_mask: 动作掩码
```

## PettingZoo AECEnv集成

```python
from pettingzoo import AECEnv

class WuhanMahjongEnv(AECEnv):
    def __init__(self, training_phase=3):
        super().__init__()
        self.state_machine = None
        self.context = None
        # ... 其他初始化
    
    def reset(self, seed=None):
        # 创建游戏上下文
        self.context = GameContext.create_new_round(seed=seed)
        
        # 初始化状态机
        rule_engine = Wuhan7P4LRuleEngine(self.context)
        observation_builder = Wuhan7P4LObservationBuilder(self.context)
        self.state_machine = MahjongStateMachine(rule_engine, observation_builder)
        self.state_machine.set_context(self.context)
        
        # 开始游戏
        self.state_machine.transition_to(GameStateType.INITIAL, self.context)
        self.state_machine.step(self.context, 'auto')
        
        # 返回第一个玩家的观测
        return self.observe(self.agent_selection), {}
    
    def step(self, action):
        # 转换action为MahjongAction
        mahjong_action = self._convert_action(action)
        
        # 执行状态机step
        self.state_machine.step(self.context, mahjong_action)
        
        # 自动推进
        while not self.state_machine.is_terminal():
            current_state = self.state_machine.current_state_type
            if current_state in [GameStateType.PLAYER_DECISION, GameStateType.DISCARDING]:
                break  # 手动状态，停止自动推进
            self.state_machine.step(self.context, 'auto')
        
        # 更新agent_selection
        if not self.state_machine.is_terminal():
            self.agent_selection = self.state_machine.get_current_agent()
        
        # 返回观测
        return self.observe(self.agent_selection), reward, terminated, truncated, info
```

## 测试

运行测试脚本：

```bash
python test_state_machine.py
```

测试包括：
1. 状态机初始化
2. 初始状态执行
3. 状态历史和回滚
4. 自动推进
5. 日志功能

## 注意事项

1. **懒加载观测**: 手动状态的观测在enter()时设置为None，只在需要时构建
2. **动作格式**: 统一使用MahjongAction对象，包含action_type和parameter
3. **自动状态**: 自动状态的step()方法必须传入'auto'字符串
4. **单步响应**: WaitResponseState采用单步模式，每调用一次step()收集一个玩家响应
5. **一炮单响**: 多玩家同时胡牌时，按优先级和距离选择唯一的响应者
6. **武汉麻将规则**: 抢杠和、杠上开花等特殊规则在DrawingAfterGongState中处理

## 性能优化

1. **懒加载观测**: 只在需要时构建观测，减少不必要的计算
2. **状态历史限制**: 最多保存100个快照，控制内存使用
3. **日志分级**: 可通过enable_logging参数控制是否启用日志

## 依赖

- Python 3.8+
- numpy
- 标准麻将规则引擎和观测构建器

## 作者

Mahjong RL Team

## 许可

MIT License
