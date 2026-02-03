"""
和牌状态 - 终端状态

该状态处理游戏结束时的和牌逻辑。
这是终端状态，不再有状态转换。

功能:
1. 标记游戏结束
2. 计算分数
3. 记录胡牌信息
4. 生成最终观测
"""

from typing import Optional

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType
from src.mahjong_rl.observation.builder import IObservationBuilder
from src.mahjong_rl.rules.base import IRuleEngine
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.score_calculator import MahjongScoreSettler
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.win_detector import WuhanMahjongWinChecker
from src.mahjong_rl.state_machine.base import GameState


class WinState(GameState):
    """
    和牌状态
    
    处理游戏结束时的和牌逻辑，计算分数并更新玩家信息。
    这是终端状态，执行step()后不再转换状态。
    
    Attributes:
        rule_engine: 规则引擎实例
        observation_builder: 观测构建器实例
    """
    
    def __init__(self, rule_engine: IRuleEngine, observation_builder: IObservationBuilder):
        """
        初始化和牌状态
        
        Args:
            rule_engine: 规则引擎实例
            observation_builder: 观测构建器实例
        """
        super().__init__(rule_engine, observation_builder)
    
    def enter(self, context: GameContext) -> None:
        """
        进入和牌状态

        标记游戏为和牌，计算所有玩家的分数。
        终端状态不需要生成观测和动作掩码。

        Args:
            context: 游戏上下文
        """
        context.current_state = GameStateType.WIN
        context.is_win = True
        # 终端状态不需要生成观测和动作掩码
        context.observation = None
        context.action_mask = None

        # 计算分数
        self._calculate_scores(context)
    
    def step(self, context: GameContext, action: Optional[str] = None) -> Optional[GameStateType]:
        """
        和牌状态不执行任何动作
        
        这是终端状态，step()方法返回None表示游戏结束。
        
        Args:
            context: 游戏上下文
            action: 忽略
        
        Returns:
            None (终端状态）
        """
        return None  # 终端状态
    
    def exit(self, context: GameContext) -> None:
        """
        离开和牌状态
        
        Args:
            context: 游戏上下文
        """
        pass
    
    def _calculate_scores(self, context: GameContext):
        """
        计算分数

        使用WuhanMahjongWinChecker和MahjongScoreSettler
        计算每个玩家的胡牌分数。

        Args:
            context: 游戏上下文
        """
        win_checker = WuhanMahjongWinChecker(context)
        score_calculator = MahjongScoreSettler(False)

        # 计算胡牌玩家的分数
        for winner_id in context.winner_ids:
            player = context.players[winner_id]
            win_result = win_checker.check_win(player)

            if win_result.can_win:
                # 计算分数
                score_list = score_calculator.settle(win_result, context)
                # 保存完整分数列表到 context（新增）
                context.final_scores = score_list
                # 同时保留赢家得分（兼容旧逻辑）
                player.fan_count = max(score_list)
