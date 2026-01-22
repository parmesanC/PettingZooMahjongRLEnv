"""
文件日志器

将日志写入 JSON 格式文件。
"""

import json
import os
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Dict, Optional

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType, ActionType
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.logging.base import ILogger
from src.mahjong_rl.logging.formatters import LogLevel, LogType, LogFormatter, to_json


class FileLogger(ILogger):
    """
    文件日志器

    将日志异步写入 JSON 格式文件，按日期自动分割。
    """

    def __init__(
        self,
        log_dir: str = "logs",
        level: LogLevel = LogLevel.INFO,
        max_file_size_mb: int = 100,
        enable_async: bool = True
    ):
        """
        初始化文件日志器

        Args:
            log_dir: 日志目录
            level: 日志级别（低于此级别的日志不会记录）
            max_file_size_mb: 单个日志文件最大大小（MB）
            enable_async: 是否启用异步写入
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.level = level
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.enable_async = enable_async

        self.current_game_id: Optional[str] = None
        self._current_date = datetime.utcnow().date()
        self._current_file: Optional[Path] = None
        self._file_size = 0

        # 异步写入队列
        self._queue: Optional[Queue] = Queue() if enable_async else None
        self._writer_thread: Optional[Thread] = None

        if enable_async:
            self._start_writer_thread()

    def _get_log_file_path(self) -> Path:
        """获取当前日期的日志文件路径"""
        current_date = datetime.utcnow().date()

        # 如果日期变化，切换到新文件
        if current_date != self._current_date or self._current_file is None:
            self._current_date = current_date
            date_str = current_date.strftime("%Y-%m-%d")
            self._current_file = self.log_dir / f"game_{date_str}.json"
            self._file_size = self._current_file.stat().st_size if self._current_file.exists() else 0

        return self._current_file

    def _check_file_size(self):
        """检查文件大小，如果超过限制则创建新文件"""
        if self._file_size > self.max_file_size:
            # 添加时间戳创建新文件
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            date_str = self._current_date.strftime("%Y-%m-%d")
            self._current_file = self.log_dir / f"game_{date_str}_{timestamp}.json"
            self._file_size = 0

    def _write_log(self, log_entry: Dict[str, Any]):
        """写入日志条目到文件"""
        log_file = self._get_log_file_path()
        self._check_file_size()

        # 追加写入 JSON
        json_str = to_json(log_entry) + "\n"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json_str)

        self._file_size += len(json_str.encode('utf-8'))

    def _start_writer_thread(self):
        """启动异步写入线程"""
        def writer():
            while True:
                log_entry = self._queue.get()
                if log_entry is None:  # 结束信号
                    break
                self._write_log(log_entry)
                self._queue.task_done()

        self._writer_thread = Thread(target=writer, daemon=True)
        self._writer_thread.start()

    def _enqueue(self, log_entry: Dict[str, Any]):
        """将日志条目加入队列（或直接写入）"""
        if self.enable_async and self._queue:
            self._queue.put(log_entry)
        else:
            self._write_log(log_entry)

    def log(self, level: LogLevel, log_type: LogType, data: Dict):
        """记录日志"""
        if level.value < self.level.value:
            return

        game_id = self.current_game_id or "unknown"
        log_entry = LogFormatter.format_log_entry(level, log_type, game_id, data)
        self._enqueue(log_entry)

    def log_state_transition(self, from_state: GameStateType, to_state: GameStateType, context: GameContext):
        """记录状态转换"""
        data = LogFormatter.format_state_transition(
            from_state.name if from_state else None,
            to_state.name,
            context.current_player_idx
        )
        self.log(LogLevel.INFO, LogType.STATE_TRANSITION, data)

    def log_action(self, player_id: int, action: MahjongAction, context: GameContext):
        """记录玩家动作"""
        data = LogFormatter.format_action(
            player_id,
            action.action_type.name,
            action.parameter,
            context.current_state.name if context.current_state else None
        )
        self.log(LogLevel.INFO, LogType.ACTION, data)

    def log_performance(self, metrics: Dict):
        """记录性能指标"""
        data = LogFormatter.format_performance(metrics)
        self.log(LogLevel.DEBUG, LogType.PERFORMANCE, data)

    def start_game(self, game_id: str, config: Dict):
        """开始新游戏记录"""
        self.current_game_id = game_id
        data = {
            "game_id": game_id,
            "config": config
        }
        self.log(LogLevel.INFO, LogType.GAME_START, data)

    def end_game(self, result: Dict):
        """结束游戏记录"""
        data = {
            "game_id": self.current_game_id,
            "result": result
        }
        self.log(LogLevel.INFO, LogType.GAME_END, data)
        self.current_game_id = None

    def close(self):
        """关闭日志器"""
        if self._queue:
            self._queue.put(None)  # 发送结束信号
            if self._writer_thread:
                self._writer_thread.join(timeout=5)

    def __del__(self):
        """析构函数"""
        self.close()
