from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType, ActionType
from src.mahjong_rl.observation.builder import IObservationBuilder
from src.mahjong_rl.rules.base import IRuleEngine


# class GameState(ABC):
#     """状态基类"""
#
#     def __init__(self, context: GameContext):
#         self.context = context
#         self.transitions: Dict[str, 'GameState'] = {}
#
#     def on_enter(self):
#         """进入状态"""
#         pass
#
#     def on_exit(self):
#         """退出状态"""
#         pass
#
#     def update(self, player_id: int, action: int) -> Optional['GameState']:
#         """状态更新，返回新状态或None"""
#         raise NotImplementedError
#
#     def get_valid_actions(self, player_id: int) -> np.ndarray:
#         """获取有效动作掩码"""
#         raise NotImplementedError


class GameState(ABC):
    """游戏状态基类 - 定义状态机的核心接口"""
    def __init__(self, rule_engine, observation_builder):
        self.rule_engine = rule_engine  # 规则引擎（包含武汉麻将规则判断）
        self.observation_builder = observation_builder  # 观测构建器（生成RL所需观测）
        self.name = self.__class__.__name__

    @abstractmethod
    def enter(self, context: GameContext) -> None:
        """进入状态时的初始化逻辑"""
        pass

    @abstractmethod
    def step(self, context: GameContext, action: Optional[tuple] = None) -> GameStateType:
        """
        处理状态内的核心逻辑
        :param context: 游戏上下文，包含所有游戏状态数据
        :param action: Agent的动作（None表示环境自动推进）
        :return: 下一个状态类型
        """
        pass

    @abstractmethod
    def exit(self, context: GameContext) -> None:
        """离开状态时的清理/收尾逻辑"""
        pass

    def build_observation(self, context: GameContext) -> None:
        """生成当前状态的观测和动作掩码（通用实现，子类可重写）"""
        context.observation = self.observation_builder.build(context.current_player_idx, context)
        context.action_mask = self.observation_builder.build_action_mask(context.current_player_idx, context)
