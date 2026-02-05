# 日志系统设计文档

## 1. 设计目标

1. **文件日志** - 将日志保存到 JSON 格式文件
2. **日志级别** - 支持 DEBUG、INFO、WARNING、ERROR 四个级别
3. **对局记录** - 记录完整游戏过程，用于训练回放或分析
4. **性能监控** - 记录执行时间、内存使用等性能指标

## 2. 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│                    日志系统架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐      ┌──────────────┐      ┌───────────┐  │
│  │  ILogger   │───→  │FileLogger    │───→  │ JSON文件  │  │
│  │  (接口)    │      │(文件日志)     │      │           │  │
│  └─────────────┘      └──────────────┘      └───────────┘  │
│         │                                                │  │
│         │                                                │  │
│         ↓                                                │  │
│  ┌─────────────┐      ┌──────────────┐                  │  │
│  │GameRecorder│───→  │对局记录JSON   │                  │  │
│  │(对局记录)   │      │              │                  │  │
│  └─────────────┘      └──────────────┘                  │  │
│                                                         │  │
│  ┌─────────────┐                                        │  │
│  │PerfMonitor  │                                        │  │
│  │(性能监控)   │                                        │  │
│  └─────────────┘                                        │  │
└─────────────────────────────────────────────────────────────┘
```

## 3. 文件结构

```
src/mahjong_rl/logging/
├── __init__.py              # 导出接口
├── base.py                  # ILogger 接口定义（已存在，需扩展）
├── file_logger.py           # 文件日志实现
├── game_recorder.py         # 对局记录器
├── perf_monitor.py          # 性能监控器
└── formatters.py            # 日志格式化工具
```

## 4. 核心组件

### 4.1 文件日志器 (FileLogger)

**职责：** 将日志写入 JSON 格式文件

**功能特性：**
- 按日期自动分割日志文件（如 `game_2026-01-22.json`）
- 支持日志级别过滤
- 异步写入，不影响游戏性能
- 文件大小限制和自动清理

**数据结构：**
```json
{
  "timestamp": "2026-01-22T10:30:45.123Z",
  "level": "INFO",
  "type": "state_transition",
  "game_id": "game_20260122_103045_abc123",
  "data": {
    "from_state": "PLAYER_DECISION",
    "to_state": "DISCARDING",
    "current_player": 0
  }
}
```

### 4.2 对局记录器 (GameRecorder)

**职责：** 记录完整游戏过程，用于训练回放

**记录内容：**
- 游戏配置（种子、玩家配置等）
- 每一步的状态、动作、观测
- 游戏结果和奖励

**数据结构：**
```json
{
  "game_id": "game_20260122_103045_abc123",
  "start_time": "2026-01-22T10:30:45.123Z",
  "config": {
    "seed": 42,
    "num_players": 4
  },
  "steps": [
    {
      "step": 0,
      "agent": "player_0",
      "observation": {...},
      "action": {"type": "DISCARD", "param": 5},
      "reward": 0.0,
      "next_observation": {...}
    }
  ],
  "result": {
    "winners": [0],
    "end_time": "2026-01-22T10:32:15.456Z",
    "total_steps": 85
  }
}
```

### 4.3 性能监控器 (PerfMonitor)

**职责：** 监控游戏运行性能指标

**监控指标：**
- **时间指标**：每步执行时间、状态转换时间、观测构建时间
- **内存指标**：内存使用峰值、垃圾回收频率
- **吞吐指标**：每秒步数、对局时长

**数据结构：**
```json
{
  "timestamp": "2026-01-22T10:30:45.123Z",
  "game_id": "game_20260122_103045_abc123",
  "metrics": {
    "step_time_ms": 15.3,
    "state_transition_ms": 2.1,
    "observation_build_ms": 8.7,
    "memory_mb": 245.8,
    "gc_count": 3
  }
}
```

### 4.4 日志格式化工具

**职责：** 统一的日志格式化和输出

**日志级别定义：**
```python
class LogLevel(Enum):
    DEBUG = 0    # 详细调试信息
    INFO = 1     # 一般信息
    WARNING = 2  # 警告信息
    ERROR = 3    # 错误信息
```

**日志类型定义：**
```python
class LogType(Enum):
    STATE_TRANSITION = "state_transition"  # 状态转换
    ACTION = "action"                      # 玩家动作
    GAME_START = "game_start"              # 游戏开始
    GAME_END = "game_end"                  # 游戏结束
    ERROR = "error"                        # 错误
    PERFORMANCE = "performance"            # 性能指标
```

## 5. 扩展的 ILogger 接口

```python
class ILogger(ABC):
    """日志系统统一接口"""

    @abstractmethod
    def log(self, level: LogLevel, log_type: LogType, data: Dict):
        """记录日志"""
        pass

    @abstractmethod
    def log_state_transition(self, from_state: GameStateType,
                           to_state: GameStateType, context: GameContext):
        """记录状态转换"""
        pass

    @abstractmethod
    def log_action(self, player_id: int, action: MahjongAction,
                  context: GameContext):
        """记录玩家动作"""
        pass

    @abstractmethod
    def log_performance(self, metrics: Dict):
        """记录性能指标"""
        pass

    @abstractmethod
    def start_game(self, game_id: str, config: Dict):
        """开始新游戏记录"""
        pass

    @abstractmethod
    def end_game(self, result: Dict):
        """结束游戏记录"""
        pass
```

## 6. 使用示例

```python
# 初始化日志系统
logger = CompositeLogger([
    FileLogger(log_dir="logs", level=LogLevel.INFO),
    GameRecorder(replay_dir="replays"),
    PerfMonitor(perf_dir="performance")
])

# 在环境中使用
env = WuhanMahjongEnv(logger=logger)

# 自动记录
obs, info = env.reset()          # 记录游戏开始
obs, reward, terminated, truncated, info = env.step(action)  # 记录每步
```

## 7. 文件输出结构

```
project_root/
├── logs/
│   ├── game_2026-01-22.json        # 当日游戏日志
│   └── game_2026-01-23.json
├── replays/
│   ├── game_20260122_103045_abc123.json  # 完整对局记录
│   └── game_20260122_113020_def456.json
└── performance/
    ├── perf_2026-01-22.json        # 性能监控数据
    └── perf_2026-01-23.json
```

## 8. 实现清单

| 文件 | 功能 | 状态 |
|------|------|------|
| `base.py` | 扩展 ILogger 接口 | 待修改 |
| `formatters.py` | 日志格式化工具 | 新建 |
| `file_logger.py` | 文件日志实现 | 新建 |
| `game_recorder.py` | 对局记录器 | 新建 |
| `perf_monitor.py` | 性能监控器 | 新建 |
| `__init__.py` | 导出接口 | 待修改 |
| `composite_logger.py` | 组合日志器 | 新建 |
