"""
杠牌状态 - 半自动状态

该状态处理武汉麻将的所有杠牌类型（明杠、暗杠、补杠、红中杠、皮子杠、赖子杠）。
对于补杠，需要检查是否可以抢杠和。

功能:
1. 处理所有类型的杠牌（明杠、暗杠、补杠、红中杠、皮子杠、赖子杠）
2. 对于补杠，检查是否有其他玩家可以抢杠和
3. 如果可以抢杠和，转到WAIT_ROB_KONG状态
4. 如果不可以抢杠和，执行杠牌，转到DRAWING_AFTER_GONG状态
"""

from typing import Optional

from src.mahjong_rl.core.GameData import GameContext, ActionRecord
from src.mahjong_rl.core.PlayerData import PlayerData, Meld
from src.mahjong_rl.core.constants import GameStateType, ActionType, WinWay
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.observation.builder import IObservationBuilder
from src.mahjong_rl.rules.base import IRuleEngine
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.win_detector import WuhanMahjongWinChecker, WinCheckResult
from src.mahjong_rl.state_machine.base import GameState


class GongState(GameState):
    """
    杠牌状态

    处理武汉麻将的所有杠牌类型：
    - 明杠（冲杠）：玩家有3张，其他人打出第4张
    - 暗杠：玩家有4张在手中
    - 补杠：玩家已有碰牌，摸到第4张（可能被抢杠和）
    - 红中杠、皮子杠、赖子杠：特殊牌的杠牌

    对于补杠，需要先检查是否有其他玩家可以抢杠和。
    这是半自动状态，对于补杠可能需要等待其他玩家的响应。
    """

    def __init__(self, rule_engine: IRuleEngine, observation_builder: IObservationBuilder):
        """
        初始化杠牌状态

        Args:
            rule_engine: 规则引擎实例
            observation_builder: 观测构建器实例
        """
        super().__init__(rule_engine, observation_builder)

    def enter(self, context: GameContext) -> None:
        """
        进入杠牌状态

        Args:
            context: 游戏上下文
        """
        context.current_state = GameStateType.GONG
        # 半自动状态不需要生成观测和动作掩码
        context.observation = None
        context.action_mask = None

    def step(self, context: GameContext, action: Optional[str] = None) -> GameStateType:
        """
        处理杠牌

        根据杠牌来源和类型，执行相应的杠牌操作：
        - 如果从WaitResponseState来（selected_responder不为None），则是明杠
        - 如果从PLAYER_DECISION来（pending_kong_action不为None），则是其他类型的杠牌
        - 对于补杠，需要先检查是否可以抢杠和

        Args:
            context: 游戏上下文
            action: 必须为'auto'（半自动状态）

        Returns:
            WAIT_ROB_KONG 如果补杠可以被抢
            DRAWING_AFTER_GONG 如果杠牌执行成功
        """
        # 验证是自动状态
        if action != 'auto':
            raise ValueError(f"GongState is an auto state, action should be 'auto', got {action}")

        # 确定杠牌类型和玩家
        player_id = None
        kong_tile = None
        kong_type = None

        if hasattr(context, 'selected_responder') and context.selected_responder is not None:
            # 明杠（从WaitResponseState来的）
            player_id = context.selected_responder
            kong_tile = context.last_discarded_tile
            kong_type = ActionType.KONG_EXPOSED

            # 执行明杠
            self._handle_kong_exposed(context, player_id, kong_tile)
            context.selected_responder = None

            # 从弃牌堆移除
            if kong_tile in context.discard_pile:
                context.discard_pile.remove(kong_tile)

            # 明杠不能被抢，直接进入杠后补牌
            context.current_player_idx = player_id
            context.is_kong_draw = True
            return GameStateType.DRAWING_AFTER_GONG

        else:
            # 从PLAYER_DECISION来的其他类型杠牌
            player_id = context.current_player_idx
            player = context.players[player_id]

            # 获取杠牌动作（从context.pending_kong_action获取）
            if not hasattr(context, 'pending_kong_action') or context.pending_kong_action is None:
                raise ValueError("GongState: No pending_kong_action found in context")

            kong_action = context.pending_kong_action
            kong_type = kong_action.action_type
            kong_tile = kong_action.parameter

            # 对于补杠，需要先检查是否可以抢杠和
            if kong_type == ActionType.KONG_SUPPLEMENT:
                # 设置被抢杠的牌
                context.rob_kong_tile = kong_tile
                context.kong_player_idx = player_id

                # 保存杠牌动作，供后续使用
                context.saved_kong_action = kong_action

                # 清理临时变量
                context.pending_kong_action = None

                # 设置胡牌场景为抢杠和
                context.win_way = WinWay.ROB_KONG.value

                # 转到等待抢杠和状态
                return GameStateType.WAIT_ROB_KONG

            # 其他类型的杠牌（暗杠、红中杠、皮子杠、赖子杠）不能被抢
            # 执行杠牌
            if kong_type == ActionType.KONG_CONCEALED:
                self._handle_kong_concealed(context, player, kong_tile)
            elif kong_type == ActionType.KONG_RED:
                self._handle_special_kong(context, player, kong_tile, 'RED')
            elif kong_type == ActionType.KONG_SKIN:
                self._handle_special_kong(context, player, kong_tile, 'SKIN')
            elif kong_type == ActionType.KONG_LAZY:
                self._handle_special_kong(context, player, kong_tile, 'LAZY')
            else:
                raise ValueError(f"GongState: Unknown kong type {kong_type}")

            # 清理临时变量
            context.pending_kong_action = None

            # 设置杠后摸牌标记
            context.current_player_idx = player_id
            context.is_kong_draw = True
            return GameStateType.DRAWING_AFTER_GONG

    def exit(self, context: GameContext) -> None:
        """
        离开杠牌状态

        Args:
            context: 游戏上下文
        """
        pass

    def _handle_kong_exposed(self, context: GameContext, player_id: int, tile: int):
        """
        处理明杠（冲杠）

        玩家有3张牌，其他玩家打出第4张。

        Args:
            context: 游戏上下文
            player_id: 玩家ID
            tile: 杠牌
        """
        player = context.players[player_id]

        # 从手牌移除三张牌
        count = 0
        for i in range(len(player.hand_tiles) - 1, -1, -1):
            if player.hand_tiles[i] == tile and count < 3:
                player.hand_tiles.pop(i)
                count += 1

        # 添加到副露
        meld = Meld(
            action_type=MahjongAction(ActionType.KONG_EXPOSED, tile),
            tiles=[tile, tile, tile, tile],
            from_player=context.discard_player
        )
        player.melds.append(meld)

        # 记录动作历史
        context.action_history.append(
            ActionRecord(
                action_type=MahjongAction(ActionType.KONG_EXPOSED, tile),
                tile=tile,
                player_id=player_id
            )
        )

    def _handle_kong_concealed(self, context: GameContext, player: PlayerData, tile: int):
        """
        处理暗杠

        玩家有4张牌在手中。

        Args:
            context: 游戏上下文
            player: 玩家数据
            tile: 杠牌
        """
        # 从手牌移除四张牌
        for _ in range(4):
            player.hand_tiles.remove(tile)

        # 添加到副露
        meld = Meld(
            action_type=MahjongAction(ActionType.KONG_CONCEALED, tile),
            tiles=[tile, tile, tile, tile],
            from_player=player.player_id
        )
        player.melds.append(meld)

        # 记录动作历史
        context.action_history.append(
            ActionRecord(
                action_type=MahjongAction(ActionType.KONG_CONCEALED, tile),
                tile=tile,
                player_id=player.player_id
            )
        )

    def _handle_special_kong(self, context: GameContext, player: PlayerData, tile: int, kong_type: str):
        """
        处理特殊杠（红中、皮子、赖子）

        Args:
            context: 游戏上下文
            player: 玩家数据
            tile: 杠牌（对于KONG_SKIN使用，对于KONG_RED和KONG_LAZY会被忽略）
            kong_type: 'RED', 'SKIN', or 'LAZY'
        """
        # 确定要移除的牌ID
        if kong_type == 'RED':
            actual_tile = context.red_dragon
        elif kong_type == 'LAZY':
            actual_tile = context.lazy_tile
        elif kong_type == 'SKIN':
            actual_tile = tile  # 皮子杠使用传入的tile（因为有两张皮子）
        else:
            raise ValueError(f"Unknown kong_type: {kong_type}")

        # 从手牌移除牌
        player.hand_tiles.remove(actual_tile)

        # 更新special_gang计数
        if kong_type == 'RED':
            player.special_gangs[2] += 1  # 红中杠
        elif kong_type == 'SKIN':
            player.special_gangs[0] += 1  # 皮子杠
        elif kong_type == 'LAZY':
            player.special_gangs[1] += 1  # 赖子杠

        # 记录动作历史
        action_type_enum = ActionType.KONG_RED if kong_type == 'RED' else (
            ActionType.KONG_SKIN if kong_type == 'SKIN' else ActionType.KONG_LAZY
        )
        context.action_history.append(
            ActionRecord(
                action_type=MahjongAction(action_type_enum, actual_tile),
                tile=actual_tile,
                player_id=player.player_id
            )
        )
