"""
等待响应状态 - 单步收集模式

该状态逐个收集其他玩家对弃牌的响应（吃/碰/杠/胡/过）。
采用单步模式，每调用一次step()收集一个玩家响应。

功能:
1. 设置响应顺序
2. 逐个收集玩家响应（单步模式）
3. 使用ResponseCollector管理响应
4. 当所有玩家响应完成后，选择最佳响应
5. 根据最佳响应类型转换状态
"""

from typing import Union

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType, ActionType, ResponsePriority
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.observation.builder import IObservationBuilder
from src.mahjong_rl.rules.base import IRuleEngine
from src.mahjong_rl.state_machine.base import GameState


class WaitResponseState(GameState):
    """
    等待响应状态
    
    逐个收集其他玩家对弃牌的响应（吃/碰/杠/胡/过）。
    采用单步模式，每调用一次step()收集一个玩家响应。
    所有玩家响应后，使用ResponseCollector选择最佳响应。
    
    Attributes:
        rule_engine: 规则引擎实例
        observation_builder: 观测构建器实例
    """
    
    def __init__(self, rule_engine: IRuleEngine, observation_builder: IObservationBuilder):
        """
        初始化等待响应状态
        
        Args:
            rule_engine: 规则引擎实例
            observation_builder: 观测构建器实例
        """
        super().__init__(rule_engine, observation_builder)
    
    def enter(self, context: GameContext) -> None:
        """
        进入等待响应状态
        
        初始化响应收集器，设置响应顺序，并为第一个响应者生成观测。
        
        Args:
            context: 游戏上下文
        """
        context.current_state = GameStateType.WAITING_RESPONSE
        
        # 初始化响应收集器
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        from mahjong_rl.state_machine.ResponseCollector import ResponseCollector
        context.response_collector = ResponseCollector()
        
        # 确保响应顺序已设置
        if not context.response_order:
            context.setup_response_order(context.discard_player)
        
        # 重置当前响应者索引
        context.current_responder_idx = 0

        # 为第一个响应者生成观测
        current_responder = context.get_current_responder()
        if current_responder is not None:
            # 更新当前玩家索引到当前响应者
            context.current_player_idx = current_responder
            # 立即生成观测和动作掩码
            self.build_observation(context)

    def step(self, context: GameContext, action: Union[MahjongAction, str]) -> GameStateType:
        """
        收集一个玩家的响应
        
        处理当前玩家的响应，移动到下一个响应者。
        当所有玩家都响应后，选择最佳响应并转换状态。
        
        Args:
            context: 游戏上下文
            action: 玩家响应动作（MahjongAction对象）或'auto'
        
        Returns:
            WAITING_RESPONSE (继续收集) 或下一个状态
        
        Raises:
            ValueError: 如果响应玩家索引超出范围
        """
        current_responder = context.get_current_responder()
        
        if current_responder is None:
            # 所有玩家都已响应，选择最佳响应
            return self._select_best_response(context)
        
        # 处理当前玩家的响应
        if action == 'auto':
            # 如果是自动模式，默认PASS
            response_action = MahjongAction(ActionType.PASS, -1)
        else:
            response_action = action
        
        # 验证动作有效性
        if not self._is_action_valid(context, current_responder, response_action):
            # 无效动作默认为PASS
            response_action = MahjongAction(ActionType.PASS, -1)
        
        # 添加到响应收集器
        priority = self._get_action_priority(response_action.action_type)
        context.response_collector.add_response(
            current_responder,
            response_action.action_type,
            priority,
            response_action.parameter  # 添加参数
        )
        
        # 移动到下一个响应者
        context.move_to_next_responder()
        
        # 检查是否所有玩家都已响应
        if context.is_all_responded():
            return self._select_best_response(context)

        # 为下一个响应者生成观测
        next_responder = context.get_current_responder()
        if next_responder is not None:
            # 更新当前玩家索引到下一个响应者
            context.current_player_idx = next_responder
            # 立即生成观测和动作掩码
            self.build_observation(context)

        return GameStateType.WAITING_RESPONSE
    
    def exit(self, context: GameContext) -> None:
        """
        离开等待响应状态
        
        Args:
            context: 游戏上下文
        """
        pass
    
    def _select_best_response(self, context: GameContext) -> GameStateType:
        """
        选择最佳响应并转换状态

        使用ResponseCollector选择最佳响应（考虑优先级和距离），
        然后根据响应类型决定下一个状态。

        Args:
            context: 游戏上下文

        Returns:
            下一个状态类型 (WIN, GONG, PROCESSING_MELD, 或 DRAWING)
        """
        best_response = context.response_collector.get_best_response(context)

        if best_response is None:
            # 检查牌墙是否为空，如果为空则流局
            if len(context.wall) == 0:
                return GameStateType.FLOW_DRAW

            # 所有玩家都PASS，下一个玩家摸牌
            next_player = (context.discard_player + 1) % 4
            context.current_player_idx = next_player
            return GameStateType.DRAWING

        # 检查最佳响应是否为PASS（所有玩家都过牌）
        if best_response.action_type == ActionType.PASS:
            # 检查牌墙是否为空，如果为空则流局
            if len(context.wall) == 0:
                return GameStateType.FLOW_DRAW
            # 所有玩家都PASS，下一个玩家摸牌
            next_player = (context.discard_player + 1) % 4
            context.current_player_idx = next_player
            return GameStateType.DRAWING

        # 根据最佳响应类型转换状态
        if best_response.action_type == ActionType.WIN:
            # 和牌
            context.winner_ids = [best_response.player_id]
            context.is_win = True
            context.win_way = 3  # WinWay.DISCARD
            return GameStateType.WIN

        elif best_response.action_type == ActionType.KONG_EXPOSED:
            # 明杠
            context.selected_responder = best_response.player_id
            return GameStateType.GONG

        elif best_response.action_type in [ActionType.CHOW, ActionType.PONG]:
            # 吃或碰
            context.selected_responder = best_response.player_id
            return GameStateType.PROCESSING_MELD

        else:
            # 不应该到达这里
            raise ValueError(f"Unexpected action type in _select_best_response: {best_response.action_type}")
    
    def _is_action_valid(self, context: GameContext, player_id: int, action: MahjongAction) -> bool:
        """
        验证动作是否有效
        
        检查玩家的动作是否在可用动作列表中。
        
        Args:
            context: 游戏上下文
            player_id: 玩家ID
            action: 动作
        
        Returns:
            True如果动作有效
        """
        player = context.players[player_id]
        discard_tile = context.last_discarded_tile
        discard_player = context.discard_player
        
        # 获取该玩家的所有可用动作
        available_actions = self.rule_engine.detect_available_actions_after_discard(
            player, discard_tile, discard_player
        )
        
        # PASS总是有效
        if action.action_type == ActionType.PASS:
            return True
        
        # 检查动作是否在可用动作列表中
        # PONG, KONG_EXPOSED 的 parameter 可忽略（被自动确定）
        if action.action_type in [ActionType.PONG, ActionType.KONG_EXPOSED]:
            return any(a.action_type == action.action_type for a in available_actions)

        return any(
            a.action_type == action.action_type and a.parameter == action.parameter
            for a in available_actions
        )
    
    def _get_action_priority(self, action_type: ActionType) -> ResponsePriority:
        """
        获取动作优先级

        Args:
            action_type: 动作类型

        Returns:
            动作优先级
        """
        priority_map = {
            ActionType.WIN: ResponsePriority.WIN,
            ActionType.KONG_EXPOSED: ResponsePriority.KONG,
            ActionType.PONG: ResponsePriority.PONG,
            ActionType.CHOW: ResponsePriority.CHOW,
            ActionType.PASS: ResponsePriority.PASS,
        }
        return priority_map.get(action_type, ResponsePriority.PASS)

    def _can_only_pass(self, context: GameContext, player_id: int) -> bool:
        """
        检查玩家是否只能 PASS

        Args:
            context: 游戏上下文
            player_id: 玩家ID

        Returns:
            True 如果只能 PASS，False 如果有其他可用动作
        """
        player = context.players[player_id]
        discard_tile = context.last_discarded_tile
        discard_player = context.discard_player

        # 获取可用动作
        available_actions = self.rule_engine.detect_available_actions_after_discard(
            player, discard_tile, discard_player
        )

        # 检查是否有非 PASS 的动作
        for action in available_actions:
            if action.action_type != ActionType.PASS:
                return False

        return True
