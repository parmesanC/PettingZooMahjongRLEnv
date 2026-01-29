from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType, ActionType
from src.mahjong_rl.core.mahjong_action import MahjongAction
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

    def should_auto_skip(self, context: GameContext) -> bool:
        """
        检查是否应该自动跳过此状态

        默认实现：不跳过
        子类可以重写此方法以支持自动跳过逻辑

        设计意图：
        - 允许状态声明"可以被自动跳过"
        - 由状态机在 transition_to() 中统一处理自动转换
        - 避免在 enter() 中包含状态转换逻辑

        Args:
            context: 游戏上下文

        Returns:
            True 表示应该自动跳过（使用空动作执行 step）
            False 表示需要等待 agent 输入
        """
        return False

    def validate_action(self, context: GameContext, action: MahjongAction, available_actions: List[MahjongAction]) -> bool:
        """
        验证动作是否在可用动作列表中

        这是通用的验证方法，子类可以覆盖以实现自定义验证逻辑。

        Args:
            context: 游戏上下文
            action: 要验证的动作
            available_actions: 可用动作列表（来自ActionValidator）

        Returns:
            True 如果动作合法，False 如果动作非法
        """
        # PASS 总是合法的
        if action.action_type == ActionType.PASS:
            return True

        # 检查动作是否在可用动作列表中
        for valid_action in available_actions:
            # 对于 PONG, KONG_EXPOSED，parameter 可以不同（由规则引擎决定）
            if action.action_type in [ActionType.PONG, ActionType.KONG_EXPOSED]:
                if valid_action.action_type == action.action_type:
                    return True

            # 对于其他动作，action_type 和 parameter 都必须匹配
            if (valid_action.action_type == action.action_type and
                valid_action.parameter == action.parameter):
                return True

        return False

    def get_available_actions(self, context: GameContext) -> List[MahjongAction]:
        """
        获取当前状态下可用的动作列表

        子类应该覆盖此方法以提供特定状态的可用动作检测逻辑。

        Args:
            context: 游戏上下文

        Returns:
            可用动作列表
        """
        # 默认实现：子类应该覆盖
        return []
