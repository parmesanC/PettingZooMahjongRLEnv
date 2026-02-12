"""
摸牌状态 - 自动状态

该状态处理玩家摸牌逻辑，自动从牌墙抽取一张牌。
这是自动状态，不需要agent输入动作。

功能:
1. 从牌墙为当前玩家摸一张牌
2. 检查摸牌后是否杠上开花（如果is_kong_draw为True）
3. 转换到PLAYER_DECISION状态
"""

from typing import Optional

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import GameStateType, WinWay
from src.mahjong_rl.observation.builder import IObservationBuilder
from src.mahjong_rl.rules.base import IRuleEngine
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.win_detector import WuhanMahjongWinChecker, WinCheckResult
from src.mahjong_rl.state_machine.base import GameState


class DrawingState(GameState):
    """
    摸牌状态
    
    处理玩家摸牌逻辑，自动从牌墙抽取一张牌。
    如果是杠后摸牌（is_kong_draw=True），还需要检查是否杠上开花。
    
    Attributes:
        rule_engine: 规则引擎实例
        observation_builder: 观测构建器实例
    """
    
    def __init__(self, rule_engine: IRuleEngine, observation_builder: IObservationBuilder):
        """
        初始化摸牌状态

        Args:
            rule_engine: 规则引擎实例
            observation_builder: 观测构建器实例
        """
        super().__init__(rule_engine, observation_builder)

        # 性能优化：缓存的 WinChecker
        self._cached_win_checker = None

    def set_cached_components(self, validator=None, win_checker=None) -> None:
        """
        设置缓存的组件（由状态机调用）

        Args:
            validator: 缓存的 ActionValidator 实例（此状态不使用）
            win_checker: 缓存的 WuhanMahjongWinChecker 实例
        """
        if win_checker is not None:
            self._cached_win_checker = win_checker
    
    def enter(self, context: GameContext) -> None:
        """
        进入摸牌状态

        设置当前状态为DRAWING。自动状态不需要生成观测和动作掩码。

        Args:
            context: 游戏上下文
        """
        context.current_state = GameStateType.DRAWING
        # 自动状态不需要生成观测和动作掩码
        context.observation = None
        context.action_mask = None
    
    def step(self, context: GameContext, action: Optional[str] = None) -> GameStateType:
        """
        执行摸牌动作
        
        从牌墙摸一张牌给当前玩家，检查特殊条件，并决定下一个状态。
        
        Args:
            context: 游戏上下文
            action: 必须为'auto'（自动状态）
        
        Returns:
            WIN 如果杠上开花
            PLAYER_DECISION 正常情况
        
        Raises:
            ValueError: 如果action不是'auto'
        """
        # 验证是自动状态
        if action != 'auto':
            raise ValueError(f"DrawingState is an auto state, action should be 'auto', got {action}")
        
        # 从牌墙摸牌（假设调用时墙一定有牌，墙空检查已在WaitResponseState处理）
        draw_tile = context.wall.popleft()
        current_player = context.players[context.current_player_idx]
        current_player.hand_tiles.append(draw_tile)

        # 存储摸到的牌供PLAYER_DECISION状态使用
        context.last_drawn_tile = draw_tile

        # 设置胡牌场景为自摸
        context.win_way = WinWay.SELF_DRAW.value
        return GameStateType.PLAYER_DECISION
    
    def exit(self, context: GameContext) -> None:
        """
        离开摸牌状态
        
        Args:
            context: 游戏上下文
        """
        pass
    
    def _check_win(self, context: GameContext, player: PlayerData) -> WinCheckResult:
        """
        检查是否胡牌（杠上开花）
        
        Args:
            context: 游戏上下文
            player: 玩家数据
        
        Returns:
            胡牌检测结果
        """
        # 使用缓存的 WinChecker 或降级创建新实例
        if self._cached_win_checker is not None:
            win_checker = self._cached_win_checker
        else:
            win_checker = WuhanMahjongWinChecker(context)
        return win_checker.check_win(player)
