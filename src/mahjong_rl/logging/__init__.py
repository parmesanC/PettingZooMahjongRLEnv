"""
日志系统

提供完整的日志记录功能，包括：
- 文件日志
- 对局记录
- 性能监控
"""

# 基础接口
from src.mahjong_rl.logging.base import ILogger

# 格式化工具
from src.mahjong_rl.logging.formatters import (
    LogLevel,
    LogType,
    LogFormatter,
    to_json
)

# 日志器实现
from src.mahjong_rl.logging.file_logger import FileLogger
from src.mahjong_rl.logging.game_recorder import GameRecorder
from src.mahjong_rl.logging.perf_monitor import PerfMonitor
from src.mahjong_rl.logging.composite_logger import CompositeLogger

__all__ = [
    # 基础接口
    'ILogger',

    # 格式化工具
    'LogLevel',
    'LogType',
    'LogFormatter',
    'to_json',

    # 日志器实现
    'FileLogger',
    'GameRecorder',
    'PerfMonitor',
    'CompositeLogger',
]
