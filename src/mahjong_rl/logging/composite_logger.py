"""
组合日志器

将多个日志器组合在一起，统一管理。
"""

from typing import List, Optional, Dict

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType, ActionType
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.logging.base import ILogger
from src.mahjong_rl.logging.formatters import LogLevel, LogType


class CompositeLogger(ILogger):
    """
    组合日志器

    将多个日志器组合在一起，统一管理和调用。
    所有日志操作会转发给所有子日志器。
    """

    def __init__(self, loggers: Optional[List[ILogger]] = None):
        """
        初始化组合日志器

        Args:
            loggers: 子日志器列表
        """
        self.loggers = loggers if loggers is not None else []

    def add_logger(self, logger: ILogger):
        """添加日志器"""
        self.loggers.append(logger)

    def remove_logger(self, logger: ILogger):
        """移除日志器"""
        if logger in self.loggers:
            self.loggers.remove(logger)

    def clear(self):
        """清空所有日志器"""
        self.loggers.clear()

    def log(self, level: LogLevel, log_type: LogType, data: Dict):
        """记录日志 - 转发给所有子日志器"""
        for logger in self.loggers:
            logger.log(level, log_type, data)

    def log_state_transition(self, from_state: GameStateType, to_state: GameStateType, context: GameContext):
        """记录状态转换 - 转发给所有子日志器"""
        for logger in self.loggers:
            logger.log_state_transition(from_state, to_state, context)

    def log_action(self, player_id: int, action: MahjongAction, context: GameContext):
        """记录玩家动作 - 转发给所有子日志器"""
        for logger in self.loggers:
            logger.log_action(player_id, action, context)

    def log_performance(self, metrics: Dict):
        """记录性能指标 - 转发给所有子日志器"""
        for logger in self.loggers:
            logger.log_performance(metrics)

    def start_game(self, game_id: str, config: Dict):
        """开始新游戏记录 - 转发给所有子日志器"""
        for logger in self.loggers:
            logger.start_game(game_id, config)

    def end_game(self, result: Dict):
        """结束游戏记录 - 转发给所有子日志器"""
        for logger in self.loggers:
            logger.end_game(result)

    def log_info(self, message: str) -> None:
        """记录信息日志 - 转发给所有子日志器"""
        for logger in self.loggers:
            logger.log_info(message)

    def get_logger(self, logger_type: type) -> Optional[ILogger]:
        """
        获取指定类型的日志器

        Args:
            logger_type: 日志器类型

        Returns:
            找到的日志器，如果没有则返回 None
        """
        for logger in self.loggers:
            if isinstance(logger, logger_type):
                return logger
        return None

    def close_all(self):
        """关闭所有日志器"""
        for logger in self.loggers:
            if hasattr(logger, 'close'):
                logger.close()
