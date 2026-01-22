"""
鸣牌后决策状态 - 手动状态

鸣牌（吃/碰）后进入此状态，玩家可以选择：
- 暗杠、补杠、红中杠、赖子杠、皮子杠
- 打出一张牌

不能：胡牌（鸣牌后不能立即胡）、过（必须出牌或杠）

设计原则：
- SRP: 单一职责 - 处理鸣牌后的决策
- OCP: 可扩展 - 通过 action_handlers 映射支持新的动作类型
- LSP: 符合 GameState 接口契约
"""

from typing import Union, Dict, Callable

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import GameStateType, ActionType
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.observation.builder import IObservationBuilder
from src.mahjong_rl.rules.base import IRuleEngine
from src.mahjong_rl.state_machine.base import GameState


class MeldDecisionState(GameState):
    """
    鸣牌后决策状态

    处理鸣牌（吃/碰）后的决策逻辑。
    这是手动状态，需要 agent 传入动作。

    与 PLAYER_DECISION 的区别：
    - PLAYER_DECISION：摸牌后，可以杠、胡、出牌
    - MELD_DECISION：鸣牌后，可以杠、出牌，但不能胡
    """

    def __init__(self, rule_engine: IRuleEngine, observation_builder: IObservationBuilder):
        """
        初始化鸣牌后决策状态

        Args:
            rule_engine: 规则引擎实例
            observation_builder: 观测构建器实例
        """
        super().__init__(rule_engine, observation_builder)

        # 定义动作处理器映射（不包含 WIN）
        self.action_handlers: Dict[ActionType, Callable] = {
            ActionType.DISCARD: self._handle_discard,
            ActionType.KONG_SUPPLEMENT: self._handle_supplement_kong,
            ActionType.KONG_CONCEALED: self._handle_concealed_kong,
            ActionType.KONG_RED: self._handle_red_kong,
            ActionType.KONG_SKIN: self._handle_skin_kong,
            ActionType.KONG_LAZY: self._handle_lazy_kong,
        }

    def enter(self, context: GameContext) -> None:
        """
        进入鸣牌后决策状态

        Args:
            context: 游戏上下文
        """
        context.current_state = GameStateType.MELD_DECISION

        # 清空 last_drawn_tile，因为鸣牌后没有新的摸牌
        context.last_drawn_tile = None

        # 生成观测和动作掩码
        self.build_observation(context)

    def step(self, context: GameContext, action: Union[MahjongAction, str]) -> GameStateType:
        """
        处理鸣牌后决策动作

        Args:
            context: 游戏上下文
            action: MahjongAction对象

        Returns:
            GONG (如果选择杠)
            DISCARDING (如果选择出牌)

        Raises:
            ValueError: 如果动作不是 MahjongAction 或动作类型不允许
        """
        if not isinstance(action, MahjongAction):
            raise ValueError(f"MeldDecisionState expects MahjongAction, got {type(action)}")

        current_player_idx = context.current_player_idx
        current_player_data = context.players[current_player_idx]

        # 保存杠牌动作到context（供GongState使用）
        if action.action_type in [ActionType.KONG_SUPPLEMENT, ActionType.KONG_CONCEALED,
                                ActionType.KONG_RED, ActionType.KONG_SKIN, ActionType.KONG_LAZY]:
            context.last_kong_action = action

        # 解析动作类型
        action_type = action.action_type

        # 使用策略模式处理动作
        handler = self.action_handlers.get(action_type, self._handle_default)
        return handler(context, action, current_player_data)

    def _handle_discard(self, context: GameContext, action: MahjongAction, current_player_data: PlayerData) -> GameStateType:
        """
        处理出牌动作

        记录玩家要打出的牌，实际打牌操作在DISCARDING状态中执行。

        Args:
            context: 游戏上下文
            action: 动作
            current_player_data: 玩家数据

        Returns:
            DISCARDING状态
        """
        discard_tile = action.parameter

        # 验证要打出的牌在手牌中
        if discard_tile not in current_player_data.hand_tiles:
            raise ValueError(
                f"Tile {discard_tile} not in player {current_player_data.player_id}'s hand"
            )

        # 将待打出的牌存储到context中，供DISCARDING状态使用
        context.pending_discard_tile = discard_tile
        return GameStateType.DISCARDING

    def _handle_supplement_kong(self, context: GameContext, action: MahjongAction, current_player_data: PlayerData) -> GameStateType:
        """
        处理补杠动作

        记录玩家的补杠意图，实际的杠牌操作在GongState中处理。

        Args:
            context: 游戏上下文
            action: 动作
            current_player_data: 玩家数据

        Returns:
            GONG状态
        """
        context.pending_kong_action = action
        return GameStateType.GONG

    def _handle_concealed_kong(self, context: GameContext, action: MahjongAction, current_player_data: PlayerData) -> GameStateType:
        """
        处理暗杠动作

        记录玩家的暗杠意图，实际的杠牌操作在GongState中处理。

        Args:
            context: 游戏上下文
            action: 动作
            current_player_data: 玩家数据

        Returns:
            GONG状态
        """
        context.pending_kong_action = action
        return GameStateType.GONG

    def _handle_red_kong(self, context: GameContext, action: MahjongAction, current_player_data: PlayerData) -> GameStateType:
        """
        处理红中杠动作

        记录玩家的红中杠意图，实际的杠牌操作在GongState中处理。

        Args:
            context: 游戏上下文
            action: 动作
            current_player_data: 玩家数据

        Returns:
            GONG状态
        """
        context.pending_kong_action = action
        return GameStateType.GONG

    def _handle_skin_kong(self, context: GameContext, action: MahjongAction, current_player_data: PlayerData) -> GameStateType:
        """
        处理皮子杠动作

        记录玩家的皮子杠意图，实际的杠牌操作在GongState中处理。

        Args:
            context: 游戏上下文
            action: 动作
            current_player_data: 玩家数据

        Returns:
            GONG状态
        """
        context.pending_kong_action = action
        return GameStateType.GONG

    def _handle_lazy_kong(self, context: GameContext, action: MahjongAction, current_player_data: PlayerData) -> GameStateType:
        """
        处理赖子杠动作

        记录玩家的赖子杠意图，实际的杠牌操作在GongState中处理。

        Args:
            context: 游戏上下文
            action: 动作
            current_player_data: 玩家数据

        Returns:
            GONG状态
        """
        context.pending_kong_action = action
        return GameStateType.GONG

    def _handle_default(self, context: GameContext, action: MahjongAction, current_player_data: PlayerData) -> GameStateType:
        """
        默认动作处理器

        处理未定义的动作类型，抛出异常。

        Args:
            context: 游戏上下文
            action: 动作
            current_player_data: 玩家数据

        Raises:
            ValueError: 如果动作类型不允许
        """
        if action.action_type == ActionType.WIN:
            raise ValueError(
                f"WIN action is not allowed in MELD_DECISION state. "
                f"Player {current_player_data.player_id} just melded (chow/pong) and cannot win immediately. "
                f"Must discard a tile or kong first."
            )

        if action.action_type == ActionType.PASS:
            raise ValueError(
                f"PASS action is not allowed in MELD_DECISION state. "
                f"Player {current_player_data.player_id} must discard a tile or kong."
            )

        raise ValueError(
            f"Action {action.action_type.name} is not allowed in MELD_DECISION state. "
            f"Valid actions: DISCARD, KONG_SUPPLEMENT, KONG_CONCEALED, KONG_RED, KONG_SKIN, KONG_LAZY"
        )

    def exit(self, context: GameContext) -> None:
        """
        离开鸣牌后决策状态

        Args:
            context: 游戏上下文
        """
        pass


if __name__ == "__main__":
    # 测试代码
    pass
