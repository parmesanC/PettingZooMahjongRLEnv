"""
性能监控器

监控游戏运行性能指标。
"""

import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType, ActionType
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.logging.base import ILogger
from src.mahjong_rl.logging.formatters import (
    LogLevel, LogType, LogFormatter, to_json
)


class PerfMonitor(ILogger):
    """
    性能监控器

    监控游戏运行时的性能指标，包括：
    - 时间指标：每步执行时间、状态转换时间、观测构建时间
    - 内存指标：内存使用峰值、垃圾回收频率
    - 吞吐指标：每秒步数、对局时长
    """

    def __init__(self, perf_dir: str = "performance", enabled: bool = True):
        """
        初始化性能监控器

        Args:
            perf_dir: 性能日志目录
            enabled: 是否启用性能监控
        """
        self.perf_dir = Path(perf_dir)
        self.perf_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

        # 当前游戏信息
        self.current_game_id: Optional[str] = None
        self.game_start_time: Optional[float] = None
        self.step_count: int = 0

        # 性能数据
        self.metrics_buffer: list = []

        # 定时器
        self.timers: Dict[str, float] = {}

    def start_timer(self, name: str):
        """启动计时器"""
        if self.enabled:
            self.timers[name] = time.perf_counter()

    def end_timer(self, name: str) -> Optional[float]:
        """结束计时器并返回耗时（毫秒）"""
        if self.enabled and name in self.timers:
            elapsed = (time.perf_counter() - self.timers[name]) * 1000
            del self.timers[name]
            return elapsed
        return None

    def record_step_metrics(self, step_time: float, state_transition_time: float, observation_build_time: float):
        """
        记录步骤性能指标

        Args:
            step_time: 完整步骤执行时间（毫秒）
            state_transition_time: 状态转换时间（毫秒）
            observation_build_time: 观测构建时间（毫秒）
        """
        if not self.enabled:
            return

        metrics = {
            "step_time_ms": round(step_time, 3),
            "state_transition_ms": round(state_transition_time, 3),
            "observation_build_ms": round(observation_build_time, 3)
        }

        self._record_metrics(metrics)

    def _record_metrics(self, metrics: Dict):
        """记录性能指标"""
        if not self.enabled:
            return

        # 添加 GC 指标
        metrics["gc_count"] = gc.get_count()[0]  # generation 0 gc count
        metrics["step"] = self.step_count

        # 添加内存指标（如果 psutil 可用）
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            mem_info = process.memory_info()
            metrics["memory_mb"] = round(mem_info.rss / 1024 / 1024, 2)

        self.metrics_buffer.append(metrics)
        self.step_count += 1

    # ILogger 接口实现

    def log(self, level: LogLevel, log_type: LogType, data: Dict):
        """记录日志（性能监控器通常不需要调用此方法）"""
        pass

    def log_state_transition(self, from_state: GameStateType, to_state: GameStateType, context: GameContext):
        """记录状态转换（性能监控器通过独立方法记录）"""
        pass

    def log_action(self, player_id: int, action: MahjongAction, context: GameContext):
        """记录玩家动作（性能监控器通过独立方法记录）"""
        pass

    def log_performance(self, metrics: Dict):
        """记录性能指标"""
        if not self.enabled:
            return

        log_entry = {
            "timestamp": LogFormatter.format_timestamp(),
            "game_id": self.current_game_id or "unknown",
            "metrics": metrics
        }

        self.metrics_buffer.append(log_entry)

    def log_info(self, message: str) -> None:
        """记录信息日志（性能监控器不记录通用信息）"""
        pass

    def start_game(self, game_id: str, config: Dict):
        """开始新游戏监控"""
        if not self.enabled:
            return

        self.current_game_id = game_id
        self.game_start_time = time.time()
        self.step_count = 0
        self.metrics_buffer = []

    def end_game(self, result: Dict):
        """结束游戏监控并保存数据"""
        if not self.enabled:
            return

        # 计算整体性能指标
        if self.game_start_time:
            total_time = time.time() - self.game_start_time
            steps_per_second = self.step_count / total_time if total_time > 0 else 0

            summary = {
                "timestamp": LogFormatter.format_timestamp(),
                "game_id": self.current_game_id,
                "summary": {
                    "total_time_s": round(total_time, 3),
                    "total_steps": self.step_count,
                    "steps_per_second": round(steps_per_second, 3),
                    "avg_step_time_ms": round(total_time * 1000 / self.step_count, 3) if self.step_count > 0 else 0
                },
                "metrics": self.metrics_buffer
            }

            # 保存到文件
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            file_path = self.perf_dir / f"perf_{date_str}.jsonl"

            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(to_json(summary) + "\n")

        # 重置状态
        self.current_game_id = None
        self.game_start_time = None
        self.metrics_buffer = []
        self.step_count = 0

    def get_current_metrics(self) -> Dict:
        """获取当前性能指标摘要"""
        if not self.enabled or not self.metrics_buffer:
            return {}

        total_time = sum(m.get("step_time_ms", 0) for m in self.metrics_buffer)
        avg_time = total_time / len(self.metrics_buffer) if self.metrics_buffer else 0

        return {
            "step_count": self.step_count,
            "avg_step_time_ms": round(avg_time, 3),
            "total_time_ms": round(total_time, 3)
        }
