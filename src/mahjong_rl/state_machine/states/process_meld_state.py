"""
处理鸣牌状态 - 自动状态

该状态处理吃或碰牌操作，将牌组添加到玩家副露中。
这是自动状态，不需要agent输入动作。

功能:
1. 处理CHOW或PONG动作
2. 将弃牌和手牌中的牌添加到玩家副露中
3. 更新current_player为响应玩家
4. 转换到MELD_DECISION状态（鸣牌后决策，可杠/出牌，不能胡）
"""

from typing import Optional

from src.mahjong_rl.core.GameData import GameContext, ActionRecord
from src.mahjong_rl.core.PlayerData import PlayerData, Meld
from src.mahjong_rl.core.constants import GameStateType, ActionType
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.observation.builder import IObservationBuilder
from src.mahjong_rl.rules.base import IRuleEngine
from src.mahjong_rl.state_machine.base import GameState


class ProcessMeldState(GameState):
    """
    处理鸣牌状态
    
    处理吃或碰牌操作，将牌组添加到玩家副露中。
    这是自动状态，从WaitResponseState获取最佳响应信息。
    
    Attributes:
        rule_engine: 规则引擎实例
        observation_builder: 观测构建器实例
    """
    
    def __init__(self, rule_engine: IRuleEngine, observation_builder: IObservationBuilder):
        """
        初始化处理鸣牌状态
        
        Args:
            rule_engine: 规则引擎实例
            observation_builder: 观测构建器实例
        """
        super().__init__(rule_engine, observation_builder)
    
    def enter(self, context: GameContext) -> None:
        """
        进入处理鸣牌状态

        Args:
            context: 游戏上下文
        """
        context.current_state = GameStateType.PROCESSING_MELD
        # 自动状态不需要生成观测和动作掩码
        context.observation = None
        context.action_mask = None
    
    def step(self, context: GameContext, action: Optional[str] = None) -> GameStateType:
        """
        处理鸣牌
        
        根据最佳响应类型，处理吃或碰牌操作，
        将牌组添加到玩家副露中，并转换到PLAYER_DECISION状态。
        
        Args:
            context: 游戏上下文
            action: 必须为'auto'（自动状态）
        
        Returns:
            PLAYER_DECISION
        
        Raises:
            ValueError: 如果无法获取最佳响应
        """
        player_id = context.selected_responder
        player = context.players[player_id]
        discard_tile = context.last_discarded_tile
        
        # 获取该玩家的最佳响应
        best_response = context.response_collector.get_best_response(context)
        
        if best_response is None:
            raise ValueError("No best response found in ProcessMeldState")
        
        # 根据动作类型处理
        if best_response.action_type == ActionType.CHOW:
            # 吃牌
            chow_type = best_response.parameter  # 0=左吃, 1=中吃, 2=右吃
            self._process_chow(context, player, discard_tile, chow_type)
        elif best_response.action_type == ActionType.PONG:
            # 碰牌
            self._process_pong(context, player, discard_tile)
        
        # 更新当前玩家为响应玩家
        context.current_player_idx = player_id
        
        # 重置selected_responder
        context.selected_responder = None
        
        # 从弃牌堆移除被鸣的牌
        if discard_tile in context.discard_pile:
            context.discard_pile.remove(discard_tile)

        # 鸣牌后需要决策（可以杠或出牌），转换到 MELD_DECISION 状态
        # MELD_DECISION 与 PLAYER_DECISION 的区别：
        # - PLAYER_DECISION: 摸牌后，可以杠、胡、出牌
        # - MELD_DECISION: 鸣牌后，可以杠、出牌，但不能胡
        return GameStateType.MELD_DECISION
    
    def exit(self, context: GameContext) -> None:
        """
        离开处理鸣牌状态
        
        Args:
            context: 游戏上下文
        """
        pass
    
    def _process_chow(self, context: GameContext, player: PlayerData, discard_tile: int, chow_type: int):
        """
        处理吃牌
        
        Args:
            context: 游戏上下文
            player: 玩家数据
            discard_tile: 弃牌
            chow_type: 吃牌类型（0=左吃, 1=中吃, 2=右吃）
        """
        # 确定需要的两张牌
        discard_val = discard_tile % 9 + 1  # 1-9
        discard_suit = discard_tile // 9
        
        if chow_type == 0:  # 左吃 (val, val+1, val+2)
            needed_vals = [discard_val, discard_val + 1, discard_val + 2]
            # 从手牌移除 val+1 和 val+2
            tiles_to_remove = [needed_vals[1] + discard_suit * 9 - 1, needed_vals[2] + discard_suit * 9 - 1]
            meld_tiles = [discard_tile, tiles_to_remove[0], tiles_to_remove[1]]
        elif chow_type == 1:  # 中吃 (val-1, val, val+1)
            needed_vals = [discard_val - 1, discard_val, discard_val + 1]
            # 从手牌移除 val-1 和 val+1
            tiles_to_remove = [needed_vals[0] + discard_suit * 9 - 1, needed_vals[2] + discard_suit * 9 - 1]
            meld_tiles = [tiles_to_remove[0], discard_tile, tiles_to_remove[1]]
        else:  # 右吃 (val-2, val-1, val)
            needed_vals = [discard_val - 2, discard_val - 1, discard_val]
            # 从手牌移除 val-2 和 val-1
            tiles_to_remove = [needed_vals[0] + discard_suit * 9 - 1, needed_vals[1] + discard_suit * 9 - 1]
            meld_tiles = [tiles_to_remove[0], tiles_to_remove[1], discard_tile]
        
        # 从手牌移除两张牌
        for tile in tiles_to_remove:
            if tile in player.hand_tiles:
                player.hand_tiles.remove(tile)
        
        # 添加到副露
        meld = Meld(
            action_type=MahjongAction(ActionType.CHOW, chow_type),
            tiles=meld_tiles,
            from_player=context.discard_player
        )
        player.melds.append(meld)
        
        # 记录动作历史
        context.action_history.append(
            ActionRecord(
                action_type=MahjongAction(ActionType.CHOW, chow_type),
                tile=discard_tile,
                player_id=player.player_id
            )
        )
    
    def _process_pong(self, context: GameContext, player: PlayerData, discard_tile: int):
        """
        处理碰牌
        
        Args:
            context: 游戏上下文
            player: 玩家数据
            discard_tile: 弃牌
        """
        # 从手牌移除两张牌
        count = 0
        for tile in player.hand_tiles[:]:  # 遍历副本
            if tile == discard_tile and count < 2:
                player.hand_tiles.remove(tile)
                count += 1
        
        # 添加到副露
        meld = Meld(
            action_type=MahjongAction(ActionType.PONG, discard_tile),
            tiles=[discard_tile, discard_tile, discard_tile],
            from_player=context.discard_player
        )
        player.melds.append(meld)
        
        # 记录动作历史
        context.action_history.append(
            ActionRecord(
                action_type=MahjongAction(ActionType.PONG, discard_tile),
                tile=discard_tile,
                player_id=player.player_id
            )
        )
