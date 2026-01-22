"""
荒牌流局状态 - 终端状态

该状态处理牌墙耗尽时的流局逻辑。
这是终端状态，不再有状态转换。

功能:
1. 标记游戏为荒牌
2. 处理庄家连庄规则
3. 记录流局信息
"""

from typing import Optional

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType
from src.mahjong_rl.observation.builder import IObservationBuilder
from src.mahjong_rl.rules.base import IRuleEngine
from src.mahjong_rl.state_machine.base import GameState


class FlushState(GameState):
    """
    荒牌流局状态
    
    处理牌墙耗尽时的流局逻辑。
    根据武汉麻将规则，流局时庄家保留。
    这是终端状态，执行step()后不再转换状态。
    
    Attributes:
        rule_engine: 规则引擎实例
        observation_builder: 观测构建器实例
    """
    
    def __init__(self, rule_engine: IRuleEngine, observation_builder: IObservationBuilder):
        """
        初始化荒牌状态
        
        Args:
            rule_engine: 规则引擎实例
            observation_builder: 观测构建器实例
        """
        super().__init__(rule_engine, observation_builder)
    
    def enter(self, context: GameContext) -> None:
        """
        进入荒牌状态

        标记游戏为荒牌，并设置庄家连庄规则。
        终端状态不需要生成观测和动作掩码。

        Args:
            context: 游戏上下文
        """
        context.current_state = GameStateType.FLOW_DRAW
        context.is_flush = True
        # 终端状态不需要生成观测和动作掩码
        context.observation = None
        context.action_mask = None

        # 确定下一局庄家（庄家连庄）
        # 规则：流局时，庄家保留
        # 已经在GameContext.create_new_round中处理
        # 这里不需要额外处理，只需要标记为流局
    
    def step(self, context: GameContext, action: Optional[str] = None) -> Optional[GameStateType]:
        """
        荒牌状态不执行任何动作
        
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
        离开荒牌状态
        
        Args:
            context: 游戏上下文
        """
        pass
