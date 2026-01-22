"""
日志格式化工具

提供统一的日志格式化和输出功能。
"""

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict


class LogLevel(Enum):
    """日志级别"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class LogType(Enum):
    """日志类型"""
    STATE_TRANSITION = "state_transition"
    ACTION = "action"
    GAME_START = "game_start"
    GAME_END = "game_end"
    ERROR = "error"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


class LogFormatter:
    """日志格式化器"""

    @staticmethod
    def format_timestamp() -> str:
        """格式化当前时间戳为 ISO 8601 格式"""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    @staticmethod
    def generate_game_id() -> str:
        """生成唯一的游戏 ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"game_{timestamp}_{unique_id}"

    @staticmethod
    def format_log_entry(
        level: LogLevel,
        log_type: LogType,
        game_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        格式化日志条目

        Args:
            level: 日志级别
            log_type: 日志类型
            game_id: 游戏 ID
            data: 日志数据

        Returns:
            格式化后的日志条目
        """
        return {
            "timestamp": LogFormatter.format_timestamp(),
            "level": level.name,
            "type": log_type.value,
            "game_id": game_id,
            "data": data
        }

    @staticmethod
    def format_state_transition(
        from_state: str,
        to_state: str,
        current_player: int
    ) -> Dict[str, Any]:
        """格式化状态转换日志"""
        return {
            "from_state": from_state,
            "to_state": to_state,
            "current_player": current_player
        }

    @staticmethod
    def format_action(
        player_id: int,
        action_type: str,
        action_parameter: int,
        state: str
    ) -> Dict[str, Any]:
        """格式化玩家动作日志"""
        return {
            "player_id": player_id,
            "action_type": action_type,
            "action_parameter": action_parameter,
            "state": state
        }

    @staticmethod
    def format_performance(metrics: Dict[str, float]) -> Dict[str, Any]:
        """格式化性能监控日志"""
        return metrics

    @staticmethod
    def format_game_config(
        seed: int,
        num_players: int,
        **kwargs
    ) -> Dict[str, Any]:
        """格式化游戏配置"""
        config = {
            "seed": seed,
            "num_players": num_players
        }
        config.update(kwargs)
        return config

    @staticmethod
    def format_game_result(
        winners: list,
        total_steps: int,
        **kwargs
    ) -> Dict[str, Any]:
        """格式化游戏结果"""
        result = {
            "winners": winners,
            "total_steps": total_steps
        }
        result.update(kwargs)
        return result


def to_json(obj: Any) -> str:
    """
    将对象转换为 JSON 字符串

    Args:
        obj: 要转换的对象

    Returns:
        JSON 字符串
    """
    def default_converter(o):
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, '__dict__'):
            return o.__dict__
        return str(o)

    return json.dumps(obj, default=default_converter, ensure_ascii=False)
