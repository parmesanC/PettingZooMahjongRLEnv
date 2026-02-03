from typing import Dict, Callable, Optional, Union, List

import numpy as np

from src.mahjong_rl.core.constants import WinWay, GameStateType, ActionType
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData, Meld
from src.mahjong_rl.state_machine.base import GameState


class PlayerDecisionState(GameState):
    """
    玩家决策状态
    
    处理玩家的决策动作，包括：
    - 打牌
    - 各种杠牌（明杠、暗杠、补杠、红中杠、皮子杠、赖子杠）
    - 和牌（自摸）
    
    这是手动状态，需要agent传入MahjongAction对象。
    """

    def __init__(self, rule_engine, observation_builder):
        super().__init__(rule_engine, observation_builder)
        # 定义动作处理器映射
        self.action_handlers: Dict[ActionType, Callable] = {
            ActionType.DISCARD: self._handle_discard,
            ActionType.KONG_SUPPLEMENT: self._handle_supplement_kong,
            ActionType.KONG_CONCEALED: self._handle_concealed_kong,
            ActionType.KONG_RED: self._handle_red_kong,
            ActionType.KONG_SKIN: self._handle_skin_kong,
            ActionType.KONG_LAZY: self._handle_lazy_kong,
            ActionType.WIN: self._handle_win,
        }

    def enter(self, context: GameContext) -> None:
        """
        进入玩家决策状态

        生成当前玩家的观测和动作掩码。

        Args:
            context: 游戏上下文
        """
        context.current_state = GameStateType.PLAYER_DECISION

        # 立即生成观测和动作掩码（不再懒加载）
        # observation_builder 会自动为当前玩家生成可用的动作掩码
        self.build_observation(context)

    def step(self, context: GameContext, action: Union[MahjongAction, str]) -> GameStateType:
        """
        处理玩家决策动作

        Args:
            context: 游戏上下文
            action: MahjongAction对象

        Returns:
            下一个状态类型
        """
        # 类型验证
        if not isinstance(action, MahjongAction):
            raise ValueError(f"PlayerDecisionState expects MahjongAction, got {type(action)}")

        current_player_idx = context.current_player_idx
        current_player_data = context.players[current_player_idx]

        # ===== 新增：动作验证 =====
        # 获取可用动作列表
        available_actions = self._get_available_actions(context, current_player_data)

        # 验证动作是否合法
        if not self.validate_action(context, action, available_actions):
            # 非法动作：抛出异常，让环境返回负奖励
            raise ValueError(
                f"Invalid action {action.action_type.name} (param={action.parameter}) "
                f"in PLAYER_DECISION state for player {current_player_idx}. "
                f"Available actions: {[f'{a.action_type.name}({a.parameter})' for a in available_actions[:5]]}..."
            )

        # ===== 原有逻辑 =====
        # 保存杠牌动作到context（供GongState使用）
        if action.action_type in [ActionType.KONG_SUPPLEMENT, ActionType.KONG_CONCEALED,
                                ActionType.KONG_RED, ActionType.KONG_SKIN, ActionType.KONG_LAZY]:
            context.last_kong_action = action

        action_type = action.action_type

        # 使用策略模式处理动作
        handler = self.action_handlers.get(action_type, self._handle_default)
        return handler(context, action, current_player_data)

    def _handle_discard(self, context: GameContext, action: MahjongAction, current_player_data: PlayerData) -> GameStateType:
        """
        处理打牌动作

        记录玩家要打出的牌，实际打牌操作在DISCARDING状态中执行。

        Args:
            context: 游戏上下文
            action: 动作
            current_player_data: 玩家数据

        Returns:
            DISCARDING状态
        """
        discard_tile = action.parameter

        # ===== 新增：防御性验证 =====
        if discard_tile not in current_player_data.hand_tiles:
            raise ValueError(
                f"Player {current_player_data.player_id} cannot discard tile {discard_tile}: "
                f"not in hand. Hand: {current_player_data.hand_tiles}"
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
        # 将杠牌动作保存到context中，供GongState使用
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

        Raises:
            ValueError: 如果玩家没有4张相同的牌
        """
        kong_tile = action.parameter

        # ===== 新增：验证暗杠条件 =====
        # 检查手牌中是否有4张相同的牌
        tile_count = current_player_data.hand_tiles.count(kong_tile)
        if tile_count < 4:
            raise ValueError(
                f"Player {current_player_data.player_id} cannot concealed kong {kong_tile}: "
                f"only has {tile_count} tiles. Hand: {current_player_data.hand_tiles}"
            )

        # 将杠牌动作保存到context中，供GongState使用
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
        # 将杠牌动作保存到context中，供GongState使用
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
        # 将杠牌动作保存到context中，供GongState使用
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
        # 将杠牌动作保存到context中，供GongState使用
        context.pending_kong_action = action
        return GameStateType.GONG
    
    def _handle_win(self, context: GameContext, action: MahjongAction, current_player_data: PlayerData) -> GameStateType:
        """
        处理和牌动作

        Args:
            context: 游戏上下文
            action: 动作
            current_player_data: 玩家数据

        Returns:
            WIN状态

        Raises:
            ValueError: 如果不能胡牌
        """
        # ===== 新增：验证胡牌条件 =====
        from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.win_detector import WuhanMahjongWinChecker

        win_checker = WuhanMahjongWinChecker(context)

        # 构建临时手牌
        # 注意：last_drawn_tile 在 DRAWING 状态已经添加到 hand_tiles 中了，不需要再次添加
        temp_hand = current_player_data.hand_tiles.copy()

        # 创建临时玩家对象
        temp_player = PlayerData(
            player_id=current_player_data.player_id,
            hand_tiles=temp_hand,
            melds=current_player_data.melds.copy(),
            special_gangs=current_player_data.special_gangs.copy()
        )

        # 检查是否真的能胡
        win_result = win_checker.check_win(temp_player)
        if not win_result.can_win:
            raise ValueError(
                f"Player {current_player_data.player_id} cannot win: "
                f"hand={temp_hand}, melds={current_player_data.melds}"
            )

        # 设置游戏状态为和牌
        context.is_win = True
        context.winner_ids = [context.current_player_idx]
        context.win_way = WinWay.SELF_DRAW.value

        return GameStateType.WIN
    
    def _handle_default(self, context: GameContext, action: MahjongAction, current_player_data: PlayerData) -> GameStateType:
        """
        默认动作处理器

        Args:
            context: 游戏上下文
            action: 动作
            current_player_data: 玩家数据

        Returns:
            下一个状态类型

        Raises:
            ValueError: 对于未处理的动作类型（如 PASS）
        """
        # 在 PLAYER_DECISION 状态下，不应该有 PASS 动作
        if action.action_type == ActionType.PASS:
            raise ValueError(f"PASS action not allowed in PLAYER_DECISION state. Current player: {context.current_player_idx}, hand: {current_player_data.hand_tiles}")

        # 对于其他未处理的动作类型，抛出异常
        raise ValueError(f"Unhandled action type {action.action_type} in PLAYER_DECISION state")
    
    def exit(self, context: GameContext) -> None:
        """
        离开玩家决策状态

        Args:
            context: 游戏上下文
        """
        pass

    def _get_available_actions(self, context: GameContext, current_player_data: PlayerData) -> List[MahjongAction]:
        """
        获取当前玩家的可用动作列表

        Args:
            context: 游戏上下文
            current_player_data: 当前玩家数据

        Returns:
            可用动作列表
        """
        from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.action_validator import ActionValidator

        validator = ActionValidator(context)
        return validator.detect_available_actions_after_draw(
            current_player_data,
            context.last_drawn_tile
        )

    def _generate_action_mask(self, context) -> np.ndarray:
        """
        生成动作掩码
        
        Args:
            context: 游戏上下文
        
        Returns:
            动作掩码数组
        """
        pass


if __name__ == "__main__":
    # 创建游戏上下文
    context_init = GameContext()
    context_init.players = [PlayerData(player_id=i) for i in range(4)]
    context_init.round_info.dealer_position = 0  # 设置庄家位置
    from mahjong_rl.state_machine.states.initialize_state import InitialState
    from mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder
    from mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine

    # 进入初始状态
    builder = Wuhan7P4LObservationBuilder(context_init)
    engine = Wuhan7P4LRuleEngine(context_init)
    initial_state = InitialState(rule_engine=engine, observation_builder=builder)
    initial_state.enter(context_init)
    initial_state.step(context_init)
