"""
对局记录器

记录完整的游戏过程，用于训练回放或分析。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType, ActionType
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.logging.base import ILogger
from src.mahjong_rl.logging.formatters import (
    LogLevel, LogType, LogFormatter, to_json
)


class GameRecorder(ILogger):
    """
    对局记录器

    记录完整的游戏过程，包括配置、每一步的观测、动作、奖励和结果。
    用于训练回放、游戏分析和调试。
    """

    def __init__(self, replay_dir: str = "replays", compress: bool = False):
        """
        初始化对局记录器

        Args:
            replay_dir: 对局记录目录
            compress: 是否压缩保存（暂未实现，预留接口）
        """
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.compress = compress

        # 当前游戏数据
        self.current_game_id: Optional[str] = None
        self.game_data: Optional[Dict[str, Any]] = None
        self.step_count: int = 0

    def start_game(self, game_id: str, config: Dict):
        """开始新游戏记录"""
        self.current_game_id = game_id
        self.step_count = 0
        self.game_data = {
            "game_id": game_id,
            "start_time": LogFormatter.format_timestamp(),
            "config": config,
            "steps": []
        }

    def record_step(
        self,
        agent: str,
        observation: Dict,
        action: Dict,
        reward: float,
        next_observation: Dict,
        info: Dict
    ):
        """
        记录游戏步骤

        Args:
            agent: 当前 agent 名称
            observation: 当前观测
            action: 执行的动作
            reward: 获得的奖励
            next_observation: 下一个观测
            info: 额外信息
        """
        if self.game_data is None:
            return

        step_data = {
            "step": self.step_count,
            "agent": agent,
            "observation": self._serialize_observation(observation),
            "action": action,
            "reward": reward,
            "next_observation": self._serialize_observation(next_observation),
            "info": info
        }

        self.game_data["steps"].append(step_data)
        self.step_count += 1

    def _serialize_observation(self, obs: Dict) -> Dict:
        """
        序列化观测数据

        将 numpy 数组等特殊类型转换为可 JSON 序列化的格式
        """
        def serialize_value(value):
            """递归序列化值"""
            if hasattr(value, 'tolist'):
                # numpy array
                return value.tolist()
            elif isinstance(value, dict):
                # 嵌套字典
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                # 列表或元组
                return [serialize_value(item) for item in value]
            else:
                # 其他类型（int, float, str, bool, None）
                return value

        return {key: serialize_value(value) for key, value in obs.items()}

    def end_game(self, result: Dict):
        """结束游戏记录并保存到文件"""
        if self.game_data is None:
            return

        self.game_data["result"] = result
        self.game_data["end_time"] = LogFormatter.format_timestamp()
        self.game_data["total_steps"] = self.step_count

        # 保存到文件
        self._save_game_data()

        # 重置状态
        game_id = self.current_game_id
        self.current_game_id = None
        self.game_data = None
        self.step_count = 0

        return game_id

    def _save_game_data(self):
        """保存游戏数据到文件"""
        if self.game_data is None:
            return

        file_path = self.replay_dir / f"{self.game_data['game_id']}.json"

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.game_data, f, ensure_ascii=False, indent=2)

    # ILogger 接口实现

    def log(self, level: LogLevel, log_type: LogType, data: Dict):
        """记录日志（对局记录器通常不需要调用此方法）"""
        pass

    def log_state_transition(self, from_state: GameStateType, to_state: GameStateType, context: GameContext):
        """记录状态转换（对局记录器通过 step 记录，不需要单独记录状态转换）"""
        pass

    def log_action(self, player_id: int, action: MahjongAction, context: GameContext):
        """记录玩家动作（对局记录器通过 step 记录，不需要单独记录动作）"""
        pass

    def log_performance(self, metrics: Dict):
        """记录性能指标（对局记录器不记录性能）"""
        pass

    def log_info(self, message: str) -> None:
        """记录信息日志（对局记录器不记录通用信息）"""
        pass

    def get_current_game_id(self) -> Optional[str]:
        """获取当前游戏 ID"""
        return self.current_game_id

    def get_step_count(self) -> int:
        """获取当前步数"""
        return self.step_count
