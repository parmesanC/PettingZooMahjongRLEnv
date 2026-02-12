"""
Mahjong游戏状态机 - 协调所有状态转换和自动推进

该模块提供了完整的状态机实现，用于管理武汉麻将七皮四赖子游戏的所有游戏阶段。
状态机负责:
1. 管理所有状态实例的注册和转换
2. 状态历史快照管理，支持回滚
3. 与PettingZoo AECEnv集成
4. 自动状态推进协调
5. 状态转换日志记录
"""

import time
from copy import deepcopy
from typing import Dict, List, Optional, Union

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.observation.builder import IObservationBuilder
from src.mahjong_rl.rules.base import IRuleEngine

# 日志系统导入
from src.mahjong_rl.logging.base import ILogger
from src.mahjong_rl.logging.formatters import LogLevel, LogType


class StateLogger:
    """
    状态转换日志记录器
    
    记录所有状态转换和关键动作，便于调试和问题追踪
    """
    
    def __init__(self):
        """初始化日志记录器"""
        self.history: List[Dict] = []
    
    def log_transition(self, from_state: Optional[GameStateType], to_state: GameStateType, context: GameContext):
        """
        记录状态转换
        
        Args:
            from_state: 源状态类型
            to_state: 目标状态类型
            context: 游戏上下文
        """
        log_entry = {
            'timestamp': time.time(),
            'type': 'transition',
            'from_state': from_state.name if from_state else None,
            'to_state': to_state.name,
            'current_player': context.current_player_idx
        }
        self.history.append(log_entry)
    
    def log_action(self, player_id: int, action: MahjongAction, context: GameContext):
        """
        记录玩家动作
        
        Args:
            player_id: 玩家ID
            action: 动作
            context: 游戏上下文
        """
        log_entry = {
            'timestamp': time.time(),
            'type': 'action',
            'player_id': player_id,
            'action_type': action.action_type.name,
            'action_parameter': action.parameter,
            'state': context.current_state.name if context.current_state else None
        }
        self.history.append(log_entry)
    
    def log(self, message: str):
        """
        记录自定义日志消息
        
        Args:
            message: 日志消息
        """
        log_entry = {
            'timestamp': time.time(),
            'type': 'log',
            'message': message
        }
        self.history.append(log_entry)
    
    def log_error(self, message: str):
        """
        记录错误信息
        
        Args:
            message: 错误消息
        """
        log_entry = {
            'timestamp': time.time(),
            'type': 'error',
            'message': message
        }
        self.history.append(log_entry)
    
    def get_history(self) -> List[Dict]:
        """
        获取完整日志历史
        
        Returns:
            日志历史列表
        """
        return self.history.copy()
    
    def clear(self):
        """清空日志历史"""
        self.history.clear()


class MahjongStateMachine:
    """
    Mahjong游戏状态机
    
    协调所有游戏状态的转换和执行，支持状态回滚和日志记录。
    实现了与PettingZoo AECEnv的集成，支持自动状态推进。
    
    Attributes:
        rule_engine: 规则引擎实例 (Wuhan7P4LRuleEngine)
        observation_builder: 观测构建器实例 (Wuhan7P4LObservationBuilder)
        states: 状态注册表，映射GameStateType到对应的状态实例
        current_state: 当前状态实例
        current_state_type: 当前状态类型
        state_history: 状态历史快照列表（支持回滚）
        logger: 状态转换日志记录器
        context: 游戏上下文引用
    """
    
    def __init__(
        self,
        rule_engine: IRuleEngine,
        observation_builder: IObservationBuilder,
        logger: Optional[ILogger] = None,
        enable_logging: bool = True
    ):
        """
        初始化状态机

        Args:
            rule_engine: 规则引擎实例，用于验证动作和检测可用动作
            observation_builder: 观测构建器实例，用于生成RL观测
            logger: 外部日志器（ILogger 实例）
            enable_logging: 是否启用内部 StateLogger（用于兼容）
        """
        self.rule_engine = rule_engine
        self.observation_builder = observation_builder

        # 日志系统：优先使用外部 logger，否则使用内部 StateLogger
        self.external_logger = logger
        self.internal_logger = StateLogger() if enable_logging else None
        
        # 注册所有状态
        self._register_states()
        
        # 当前状态
        self.current_state = None
        self.current_state_type = None
        
        # 状态历史（用于回滚）
        self.state_history: List[Dict] = []
        
        # 游戏上下文引用（需要在reset时设置）
        self.context: Optional[GameContext] = None
    
    def _register_states(self):
        """注册所有状态到状态机"""
        # 导入所有状态类
        from src.mahjong_rl.state_machine.states.initialize_state import InitialState
        from src.mahjong_rl.state_machine.states.drawing_state import DrawingState
        from src.mahjong_rl.state_machine.states.discarding_state import DiscardingState
        from src.mahjong_rl.state_machine.states.player_decision_state import PlayerDecisionState
        from src.mahjong_rl.state_machine.states.meld_decision_state import MeldDecisionState
        from src.mahjong_rl.state_machine.states.wait_response_state import WaitResponseState
        from src.mahjong_rl.state_machine.states.process_meld_state import ProcessMeldState
        from src.mahjong_rl.state_machine.states.gong_state import GongState
        from src.mahjong_rl.state_machine.states.drawing_after_gong_state import DrawingAfterGongState
        from src.mahjong_rl.state_machine.states.wait_rob_kong_state import WaitRobKongState
        from src.mahjong_rl.state_machine.states.win_state import WinState
        from src.mahjong_rl.state_machine.states.flush_state import FlushState

        # 创建状态注册表
        self.states = {
            GameStateType.INITIAL: InitialState(self.rule_engine, self.observation_builder),
            GameStateType.DRAWING: DrawingState(self.rule_engine, self.observation_builder),
            GameStateType.DISCARDING: DiscardingState(self.rule_engine, self.observation_builder),
            GameStateType.PLAYER_DECISION: PlayerDecisionState(self.rule_engine, self.observation_builder),
            GameStateType.MELD_DECISION: MeldDecisionState(self.rule_engine, self.observation_builder),
            GameStateType.WAITING_RESPONSE: WaitResponseState(self.rule_engine, self.observation_builder),
            GameStateType.PROCESSING_MELD: ProcessMeldState(self.rule_engine, self.observation_builder),
            GameStateType.GONG: GongState(self.rule_engine, self.observation_builder),
            GameStateType.DRAWING_AFTER_GONG: DrawingAfterGongState(self.rule_engine, self.observation_builder),
            GameStateType.WAIT_ROB_KONG: WaitRobKongState(self.rule_engine, self.observation_builder),
            GameStateType.WIN: WinState(self.rule_engine, self.observation_builder),
            GameStateType.FLOW_DRAW: FlushState(self.rule_engine, self.observation_builder),
        }
    
    def set_context(self, context: GameContext):
        """
        设置游戏上下文引用

        Args:
            context: 游戏上下文
        """
        self.context = context

    def set_cached_components(self, validator=None, win_checker=None) -> None:
        """
        设置缓存的组件（由环境调用）

        将缓存的 ActionValidator 和 WuhanMahjongWinChecker 传递给所有状态。

        Args:
            validator: 缓存的 ActionValidator 实例
            win_checker: 缓存的 WuhanMahjongWinChecker 实例
        """
        # 存储到状态机（用于后续新状态）
        self._cached_validator = validator
        self._cached_win_checker = win_checker

        # 传递给当前状态（如果支持）
        if self.current_state and hasattr(self.current_state, 'set_cached_components'):
            self.current_state.set_cached_components(validator, win_checker)

        # 传递给所有已注册的状态
        for state in self.states.values():
            if hasattr(state, 'set_cached_components'):
                state.set_cached_components(validator, win_checker)
    
    def transition_to(self, new_state_type: GameStateType, context: GameContext):
        """
        转换到新状态

        执行状态转换的完整流程：
        1. 退出当前状态
        2. 设置新状态
        3. 保存快照（不包括终端状态）
        4. 记录转换日志
        5. 进入新状态
        6. 【新增】检查是否需要自动跳过

        Args:
            new_state_type: 目标状态类型
            context: 游戏上下文
        """
        # 退出当前状态
        if self.current_state:
            self.current_state.exit(context)

        # 设置新状态（先设置，确保快照中的state_type正确）
        old_state_type = self.current_state_type
        self.current_state_type = new_state_type
        self.current_state = self.states[new_state_type]

        # 记录快照（不包括终端状态）
        if new_state_type not in [GameStateType.WIN, GameStateType.FLOW_DRAW]:
            self._save_snapshot(context)

        # 记录状态转换日志（外部 logger 和内部 logger）
        if self.external_logger:
            self.external_logger.log_state_transition(old_state_type, new_state_type, context)

        if self.internal_logger:
            self.internal_logger.log_transition(old_state_type, new_state_type, context)

        # 进入新状态
        self.current_state.enter(context)

        # 【新增】检查是否需要自动跳过
        if self.current_state.should_auto_skip(context):
            self._auto_skip_state(context)

    def _auto_skip_state(self, context: GameContext) -> None:
        """
        自动跳过当前状态

        当状态的 should_auto_skip() 返回 True 时调用，
        使用 'auto' 动作执行 step()，触发状态转换。

        设计意图：
        - 统一处理自动跳过逻辑
        - 避免 enter() 中包含状态转换代码
        - 支持递归自动跳过（跳过后的状态也可能需要跳过）

        Args:
            context: 游戏上下文
        """
        if self.external_logger:
            self.external_logger.log_info(f"Auto-skipping state {self.current_state_type.name}")

        # 执行 step()，传入 'auto' 动作
        next_state_type = self.current_state.step(context, 'auto')

        # 如果需要转换状态
        if next_state_type is not None and next_state_type != self.current_state_type:
            self.transition_to(next_state_type, context)

    def step(self, context: GameContext, action: Union[MahjongAction, str] = 'auto') -> Optional[GameStateType]:
        """
        执行一步游戏
        
        这是状态机的核心方法，根据当前状态和传入的动作执行相应的逻辑。
        
        Args:
            context: 游戏上下文，包含所有游戏状态数据
            action: agent动作（MahjongAction对象）或'auto'（自动状态）
                    手动状态需要agent传入MahjongAction对象
                    自动状态使用'auto'标记
        
        Returns:
            下一个状态类型，如果是终端状态则返回None
        
        Raises:
            RuntimeError: 如果状态机未初始化（未调用transition_to）
        """
        if not self.current_state:
            raise RuntimeError("State machine not initialized. Call transition_to() first.")
        
        # 记录动作日志（仅手动状态）
        if action != 'auto':
            if self.external_logger:
                self.external_logger.log_action(context.current_player_idx, action, context)
            if self.internal_logger:
                self.internal_logger.log_action(context.current_player_idx, action, context)
        
        # 执行当前状态的step方法
        next_state_type = self.current_state.step(context, action)
        
        # 如果需要转换状态
        if next_state_type is not None and next_state_type != self.current_state_type:
            self.transition_to(next_state_type, context)
        
        return next_state_type
    
    def is_terminal(self) -> bool:
        """
        检查是否在终端状态
        
        Returns:
            True如果当前是WIN或FLOW_DRAW状态，False否则
        """
        return self.current_state_type in [GameStateType.WIN, GameStateType.FLOW_DRAW]
    
    def get_current_agent(self) -> str:
        """
        获取当前agent名称（用于AECEnv的agent_selection）
        
        Returns:
            当前agent名称，格式为'player_0', 'player_1'等
        """
        return f"player_{self.get_current_player_id()}"
    
    def get_current_player_id(self) -> int:
        """
        获取当前玩家ID
        
        Returns:
            当前玩家索引（0-3），如果未初始化则返回-1
        """
        if self.context is None:
            return -1
        return self.context.current_player_idx
    
    def _save_snapshot(self, context: GameContext):
        """
        保存当前状态快照（用于回滚）
        
        保存当前游戏上下文和状态信息的深拷贝，以便后续回滚。
        
        Args:
            context: 游戏上下文
        """
        snapshot = {
            'state_type': self.current_state_type,
            'context': deepcopy(context),
            'timestamp': time.time()
        }
        self.state_history.append(snapshot)
        
        # 限制历史记录大小（最多保存100个快照以控制内存）
        if len(self.state_history) > 100:
            self.state_history.pop(0)
    
    def rollback(self, steps: int = 1) -> GameContext:
        """
        回滚到之前的状态
        
        通过保存的状态快照，可以回滚到之前的任何状态。
        
        Args:
            steps: 回滚步数（默认为1，回退一个状态）
        
        Returns:
            回滚后的游戏上下文
        
        Raises:
            ValueError: 如果回滚步数超过历史记录长度
        """
        if steps > len(self.state_history):
            raise ValueError(
                f"Cannot rollback {steps} steps, history has {len(self.state_history)} steps"
            )
        
        # 获取目标快照
        snapshot = self.state_history[-(steps + 1)]
        
        # 截断历史记录
        self.state_history = self.state_history[:-(steps + 1)]

        # 恢复状态
        self.current_state_type = snapshot['state_type']
        self.current_state = self.states[self.current_state_type]

        if self.internal_logger:
            self.internal_logger.log(f"Rolled back {steps} steps to {self.current_state_type.name}")

        return snapshot['context']
    
    def get_history(self) -> List[Dict]:
        """
        获取状态历史快照
        
        Returns:
            状态历史快照列表
        """
        return self.state_history.copy()
    
    def clear_history(self):
        """清空状态历史"""
        self.state_history.clear()
        if self.internal_logger:
            self.internal_logger.log("State history cleared")
    
    def get_logger(self) -> Optional[Union[ILogger, StateLogger]]:
        """
        获取日志记录器

        优先返回外部 logger，如果没有则返回内部 StateLogger

        Returns:
            日志记录器实例，如果未启用则返回None
        """
        return self.external_logger or self.internal_logger
