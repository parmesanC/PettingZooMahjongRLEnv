"""
日志系统基础接口

定义日志系统的统一接口，所有日志器都需要实现此接口。
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType, ActionType
from src.mahjong_rl.core.mahjong_action import MahjongAction

from src.mahjong_rl.logging.formatters import LogLevel, LogType


class ILogger(ABC):
    """
    日志系统统一接口

    所有日志器都需要实现此接口，提供统一的日志记录功能。
    """

    @abstractmethod
    def log(self, level: LogLevel, log_type: LogType, data: Dict):
        """
        记录通用日志

        Args:
            level: 日志级别
            log_type: 日志类型
            data: 日志数据
        """
        pass

    @abstractmethod
    def log_state_transition(self, from_state: GameStateType, to_state: GameStateType, context: GameContext):
        """
        记录状态转换

        Args:
            from_state: 源状态
            to_state: 目标状态
            context: 游戏上下文
        """
        pass

    @abstractmethod
    def log_action(self, player_id: int, action: MahjongAction, context: GameContext):
        """
        记录玩家动作

        Args:
            player_id: 玩家 ID
            action: 动作对象
            context: 游戏上下文
        """
        pass

    @abstractmethod
    def log_performance(self, metrics: Dict):
        """
        记录性能指标

        Args:
            metrics: 性能指标字典
        """
        pass

    @abstractmethod
    def start_game(self, game_id: str, config: Dict):
        """
        开始新游戏记录

        Args:
            game_id: 游戏 ID
            config: 游戏配置
        """
        pass

    @abstractmethod
    def log_info(self, message: str) -> None:
        """
        记录信息日志

        Args:
            message: 日志消息
        """
        pass

    @abstractmethod
    def end_game(self, result: Dict):
        """
        结束游戏记录

        Args:
            result: 游戏结果
        """
        pass