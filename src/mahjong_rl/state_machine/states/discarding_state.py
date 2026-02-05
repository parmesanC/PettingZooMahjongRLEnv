"""
出牌状态 - 自动状态

该状态自动执行玩家打牌操作并等待其他玩家响应。
这是自动状态，传入'auto'即可自动完成打牌。

功能:
1. 自动从手牌移除玩家决策时指定的牌
2. 将牌添加到弃牌堆
3. 设置last_discarded_tile
4. 设置响应顺序（其他三个玩家）
5. 转换到WAITING_RESPONSE状态
"""

from typing import Optional, Union

import numpy as np

from src.mahjong_rl.core.GameData import GameContext, ActionRecord
from src.mahjong_rl.core.constants import GameStateType, ActionType, WinWay
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.observation.builder import IObservationBuilder
from src.mahjong_rl.rules.base import IRuleEngine
from src.mahjong_rl.state_machine.base import GameState


class DiscardingState(GameState):
    """
    出牌状态
    
    自动执行玩家打牌操作，等待其他玩家响应（吃/碰/杠/胡/过）。
    这是自动状态，传入'auto'即可自动完成打牌。
    待打出的牌由PLAYER_DECISION状态通过context.pending_discard_tile指定。
    
    Attributes:
        rule_engine: 规则引擎实例
        observation_builder: 观测构建器实例
    """
    
    def __init__(self, rule_engine: IRuleEngine, observation_builder: IObservationBuilder):
        """
        初始化出牌状态
        
        Args:
            rule_engine: 规则引擎实例
            observation_builder: 观测构建器实例
        """
        super().__init__(rule_engine, observation_builder)
    
    def enter(self, context: GameContext) -> None:
        """
        进入出牌状态
        
        设置当前状态为DISCARDING。
        
        Args:
            context: 游戏上下文
        """
        context.current_state = GameStateType.DISCARDING
        
        # 自动状态不需要生成观测和动作掩码
        context.observation = None
        context.action_mask = None
    
    def step(self, context: GameContext, action: Union[str, None] = None) -> GameStateType:
        """
        自动执行打牌动作
        
        从context.pending_discard_tile读取待打出的牌，自动从手牌移除并添加到弃牌堆，
        设置响应顺序，然后转换到WAITING_RESPONSE状态。
        
        Args:
            context: 游戏上下文
            action: 必须为'auto'（自动状态）
        
        Returns:
            WAITING_RESPONSE
        
        Raises:
            ValueError: 如果action不是'auto'或pending_discard_tile未设置
        """
        # 验证是自动状态
        if action != 'auto':
            raise ValueError(f"DiscardingState is an auto state, action should be 'auto', got {action}")
        
        # 检查是否有待打出的牌
        if not hasattr(context, 'pending_discard_tile') or context.pending_discard_tile is None:
            raise ValueError("No pending_discard_tile found in context")
        
        current_player = context.players[context.current_player_idx]
        discard_tile = context.pending_discard_tile
        
        # 验证牌在手牌中
        if discard_tile not in current_player.hand_tiles:
            raise ValueError(f"Tile {discard_tile} not in player {context.current_player_idx}'s hand")
        
        # 从手牌移除并添加到弃牌堆
        current_player.hand_tiles.remove(discard_tile)
        current_player.discard_tiles.append(discard_tile)
        context.discard_pile.append(discard_tile)
        context.last_discarded_tile = discard_tile
        context.discard_player = context.current_player_idx
        
        # 记录动作历史
        context.action_history.append(
            ActionRecord(
                action_type=MahjongAction(ActionType.DISCARD, discard_tile),
                tile=discard_tile,
                player_id=context.current_player_idx
            )
        )
        
        # 设置响应顺序（其他三个玩家）
        context.setup_response_order(context.current_player_idx)
        
        # 清理临时变量
        context.pending_discard_tile = None
        
        # 设置胡牌场景为点炮
        context.win_way = WinWay.DISCARD.value
        return GameStateType.WAITING_RESPONSE
    
    def exit(self, context: GameContext) -> None:
        """
        离开出牌状态
        
        Args:
            context: 游戏上下文
        """
        pass
