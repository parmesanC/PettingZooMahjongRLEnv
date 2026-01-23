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

        职责：仅初始化，不执行任何状态转换逻辑
        - 初始化响应收集器
        - 构建响应者列表（区分需要决策和只能 PASS 的玩家）
        - 为需要决策的玩家生成观测

        设计原则：
        - SRP: enter() 只负责初始化
        - 状态转换逻辑由 should_auto_skip() 和 transition_to() 处理
        """
        context.current_state = GameStateType.WAITING_RESPONSE

        # 初始化响应收集器
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        from mahjong_rl.state_machine.ResponseCollector import ResponseCollector
        context.response_collector = ResponseCollector()

        # 确保响应顺序已设置（自动排除出牌者）
        if not context.response_order:
            context.setup_response_order(context.discard_player)

        # 构建响应者列表
        context.active_responders = []
        context.active_responder_idx = 0

        for responder_id in context.response_order:
            if not self._can_only_pass(context, responder_id):
                # 需要决策的玩家
                context.active_responders.append(responder_id)
            else:
                # 只能 PASS，自动添加响应
                context.response_collector.add_response(
                    responder_id,
                    ActionType.PASS,
                    ResponsePriority.PASS,
                    -1  # PASS 无参数
                )

        # 关键修改：如果所有玩家都只能 PASS，不在这里调用 _select_best_response
        # 而是通过 should_auto_skip() 让状态机处理自动跳过

        # 如果有需要决策的玩家，为第一个生成观测
        if context.active_responders:
            first_responder = context.active_responders[0]
            context.current_player_idx = first_responder
            self.build_observation(context)

    def step(self, context: GameContext, action: Union[MahjongAction, str]) -> GameStateType:
        """
        处理一个真实响应者的响应

        Args:
            context: 游戏上下文
            action: 玩家响应动作或'auto'

        Returns:
            WAITING_RESPONSE (继续) 或下一个状态
        """
        # 检查是否还有待处理的响应者
        if context.active_responder_idx >= len(context.active_responders):
            # 所有真实响应者都已处理
            return self._select_best_response(context)

        # 获取当前真实响应者
        current_responder = context.active_responders[context.active_responder_idx]

        # 处理响应
        if action == 'auto':
            response_action = MahjongAction(ActionType.PASS, -1)
        else:
            response_action = action

        # 验证动作有效性
        if not self._is_action_valid(context, current_responder, response_action):
            response_action = MahjongAction(ActionType.PASS, -1)

        # 添加到响应收集器
        priority = self._get_action_priority(response_action.action_type)
        context.response_collector.add_response(
            current_responder,
            response_action.action_type,
            priority,
            response_action.parameter
        )

        # 移动到下一个真实响应者
        context.active_responder_idx += 1

        # 检查是否还有待处理的响应者
        if context.active_responder_idx >= len(context.active_responders):
            # 所有真实响应者处理完毕
            return self._select_best_response(context)

        # 为下一个真实响应者生成观测
        next_responder = context.active_responders[context.active_responder_idx]
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
