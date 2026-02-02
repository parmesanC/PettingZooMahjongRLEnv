"""
杠后补牌状态 - 自动状态

该状态处理杠后摸牌，并检查武汉麻将的特殊规则（杠上开花）。
这是自动状态，不需要agent输入动作。

功能:
1. 杠后摸牌
2. 检查是否杠上开花
3. 转换到PLAYER_DECISION或WIN状态
"""

from typing import Optional

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import GameStateType, WinWay
from src.mahjong_rl.observation.builder import IObservationBuilder
from src.mahjong_rl.rules.base import IRuleEngine
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.win_detector import WuhanMahjongWinChecker, WinCheckResult
from src.mahjong_rl.state_machine.base import GameState


class DrawingAfterGongState(GameState):
    """
    杠后补牌状态

    杠后摸牌，并检查武汉麻将的特殊规则：
    - 杠上开花：自己摸到可以胡的牌

    这是自动状态，从GongState进入。
    """

    def __init__(self, rule_engine: IRuleEngine, observation_builder: IObservationBuilder):
        """
        初始化杠后补牌状态

        Args:
            rule_engine: 规则引擎实例
            observation_builder: 观测构建器实例
        """
        super().__init__(rule_engine, observation_builder)

    def enter(self, context: GameContext) -> None:
        """
        进入杠后补牌状态

        Args:
            context: 游戏上下文
        """
        context.current_state = GameStateType.DRAWING_AFTER_GONG
        # 自动状态不需要生成观测和动作掩码
        context.observation = None
        context.action_mask = None

    def step(self, context: GameContext, action: Optional[str] = None) -> GameStateType:
        """
        杠后摸牌

        从牌墙摸牌，检查杠上开花，然后决定下一个状态。

        Args:
            context: 游戏上下文
            action: 必须为'auto'（自动状态）

        Returns:
            FLOW_DRAW 如果牌墙为空
            WIN 如果杠上开花
            PLAYER_DECISION 正常情况

        Raises:
            ValueError: 如果action不是'auto'
        """
        # 验证是自动状态
        if action != 'auto':
            raise ValueError(f"DrawingAfterGongState is an auto state, action should be 'auto', got {action}")

        # 检查牌墙
        if len(context.wall) == 0:
            return GameStateType.FLOW_DRAW

        # 摸牌
        draw_tile = context.wall.pop()
        current_player = context.players[context.current_player_idx]
        current_player.hand_tiles.append(draw_tile)

        # 存储摸到的牌供PLAYER_DECISION状态使用
        context.last_drawn_tile = draw_tile

        # 检查杠上开花（自己胡这张牌）
        win_result = self._check_win(context, current_player)
        if win_result.can_win:
            context.win_way = WinWay.KONG_SELF_DRAW.value
            context.winner_ids = [context.current_player_idx]
            context.is_win = True
            return GameStateType.WIN

        return GameStateType.PLAYER_DECISION

    def exit(self, context: GameContext) -> None:
        """
        离开杠后补牌状态

        重置杠后摸牌标记。

        Args:
            context: 游戏上下文
        """
        # 重置杠后摸牌标记
        context.is_kong_draw = False

    def _check_win(self, context: GameContext, player: PlayerData) -> WinCheckResult:
        """
        检查自己是否胡牌（杠开）

        杠开：武汉麻将特殊规则，杠后摸牌并胡牌。

        Args:
            context: 游戏上下文
            player: 玩家数据

        Returns:
            胡牌检测结果
        """
        win_checker = WuhanMahjongWinChecker(context)
        return win_checker.check_win(player)
