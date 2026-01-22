"""
MahjongEnv状态机集成示例

展示如何在MahjongEnv中集成MahjongStateMachine，实现完整的PettingZoo AECEnv。
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType, ActionType
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.state_machine.machine import MahjongStateMachine
from src.mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder
from src.mahjong_rl.rules.round_info import RoundInfo

# 日志系统导入
from src.mahjong_rl.logging.base import ILogger
from src.mahjong_rl.logging import (
    CompositeLogger,
    FileLogger,
    GameRecorder,
    PerfMonitor,
    LogLevel,
    LogFormatter
)


class WuhanMahjongEnv(AECEnv):
    """
    武汉麻将七皮四赖子环境 - 状态机集成版本
    
    该环境集成了完整的MahjongStateMachine，实现了PettingZoo AECEnv接口。
    状态机自动处理所有游戏逻辑，环境只需负责agent交互和观测传递。
    """

    metadata = {'render_modes': ['human', 'ansi'], 'name': 'wuhan_mahjong_v1.0'}
    
    # 动作空间：参数化动作
    ACTION_SPACE = len(ActionType)  # 11种动作类型
    PARAM_SPACE = 35  # 34种牌 + 1个通配符

    # 默认日志配置
    DEFAULT_LOG_CONFIG = {
        "file_logger": {
            "enabled": True,
            "log_dir": "logs",
            "level": "INFO"
        },
        "game_recorder": {
            "enabled": False,
            "replay_dir": "replays"
        },
        "perf_monitor": {
            "enabled": False,
            "perf_dir": "performance"
        }
    }

    def __init__(
        self,
        render_mode=None,
        training_phase=3,
        enable_logging=True,
        log_config=None,
        logger=None,
        enable_perf_monitor=False
    ):
        """
        初始化环境

        Args:
            render_mode: 渲染模式
            training_phase: 训练阶段（影响信息可见度）
            enable_logging: 是否启用日志系统
            log_config: 日志配置字典（覆盖默认配置）
            logger: 自定义日志器（如果提供，则忽略 log_config）
            enable_perf_monitor: 是否启用性能监控
        """
        super().__init__()

        self.training_phase = training_phase
        self.num_players = 4
        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        self.agents = self.possible_agents[:]
        self.agents_name_mapping = {name: i for i, name in enumerate(self.agents)}

        # 定义观测空间
        self.observation_spaces = self._create_observation_spaces()
        self.action_spaces = {
            agent: spaces.Tuple((
                spaces.Discrete(self.ACTION_SPACE),  # 动作类型
                spaces.Discrete(self.PARAM_SPACE)  # 动作参数
            )) for agent in self.possible_agents
        }

        # 初始化日志系统
        self.logger = self._setup_logger(logger, log_config, enable_logging, enable_perf_monitor)
        self.current_game_id: Optional[str] = None
        self._game_logged = False  # 跟踪游戏是否已记录结束

        # 初始化状态机
        self.context: GameContext = None
        self.state_machine: MahjongStateMachine = None
        self.enable_logging = enable_logging

        # 初始化轮次信息（Env 内部管理）
        self.round_info = RoundInfo()

        self.render_mode = render_mode
    
    def _create_observation_spaces(self) -> Dict[str, spaces.Dict]:
        """创建观测空间"""
        observation_spaces = {}
        
        for agent in self.possible_agents:
            observation_spaces[agent] = spaces.Dict({
                'global_hand': spaces.MultiDiscrete([6] * (4 * 34)),
                'private_hand': spaces.MultiDiscrete([6] * 34),
                'discard_pool_total': spaces.MultiDiscrete([6] * 34),
                'wall': spaces.MultiDiscrete([35] * 82),
                'melds': spaces.Dict({
                    'action_types': spaces.MultiDiscrete([11] * 16),
                    'tiles': spaces.MultiDiscrete([35] * 256),
                    'group_indices': spaces.MultiDiscrete([4] * 32),
                }),
                'action_history': spaces.Dict({
                    'types': spaces.MultiDiscrete([11] * 80),
                    'params': spaces.MultiDiscrete([35] * 80),
                    'players': spaces.MultiDiscrete([4] * 80),
                }),
                'special_gangs': spaces.MultiDiscrete([8, 4, 5] * 4),
                'current_player': spaces.MultiDiscrete([4]),
                'fan_counts': spaces.MultiDiscrete([600] * 4),
                'special_indicators': spaces.MultiDiscrete([34, 34]),
                'remaining_tiles': spaces.Discrete(137),
                'dealer': spaces.MultiDiscrete([4]),
                'current_phase': spaces.Discrete(8),
                'action_mask': spaces.MultiBinary(145),  # 扁平化动作掩码（145位）
            })
        
        return observation_spaces

    def _setup_logger(
        self,
        logger: Optional[ILogger],
        log_config: Optional[Dict],
        enable_logging: bool,
        enable_perf_monitor: bool
    ) -> Optional[ILogger]:
        """
        设置日志系统

        Args:
            logger: 自定义日志器
            log_config: 日志配置
            enable_logging: 是否启用日志
            enable_perf_monitor: 是否启用性能监控

        Returns:
            配置好的日志器，如果禁用则返回 None
        """
        # 如果禁用日志，返回 None
        if not enable_logging:
            return None

        # 如果提供了自定义日志器，直接使用
        if logger is not None:
            return logger

        # 合并默认配置和用户配置
        config = self.DEFAULT_LOG_CONFIG.copy()
        if log_config is not None:
            for key, value in log_config.items():
                if key in config and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value

        # 强制启用性能监控（如果指定）
        if enable_perf_monitor:
            config["perf_monitor"]["enabled"] = True

        # 创建组合日志器
        loggers = []

        # 文件日志器
        if config["file_logger"]["enabled"]:
            file_config = config["file_logger"]
            level = LogLevel[file_config.get("level", "INFO")]
            loggers.append(FileLogger(
                log_dir=file_config.get("log_dir", "logs"),
                level=level
            ))

        # 对局记录器
        if config["game_recorder"]["enabled"]:
            recorder_config = config["game_recorder"]
            loggers.append(GameRecorder(
                replay_dir=recorder_config.get("replay_dir", "replays")
            ))

        # 性能监控器
        if config["perf_monitor"]["enabled"]:
            perf_config = config["perf_monitor"]
            loggers.append(PerfMonitor(
                perf_dir=perf_config.get("perf_dir", "performance"),
                enabled=True
            ))

        # 如果没有任何日志器，返回 None
        if not loggers:
            return None

        # 创建组合日志器
        if len(loggers) == 1:
            return loggers[0]
        return CompositeLogger(loggers)

    def _get_total_steps(self) -> int:
        """
        获取当前游戏的总步数

        Returns:
            总步数
        """
        if self.context and hasattr(self.context, 'action_history'):
            return len(self.context.action_history)
        return 0

    def reset(self, seed=None, options=None):
        """
        重置环境

        创建新局游戏，初始化状态机，开始游戏。

        Returns:
            第一个玩家的观测和空info字典
        """
        # 设置随机种子
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)

        # 重置agents到初始状态
        self.agents = self.possible_agents[:]

        # 重置所有状态
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        # Initialize cumulative rewards for PettingZoo AECEnv compatibility
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}

        # 设置初始agent selection
        self.agent_selection = self.possible_agents[0]

        # 生成游戏 ID
        self.current_game_id = LogFormatter.generate_game_id()

        # 创建空 GameContext，并传入 round_info（由 InitialState 完成初始化）
        self.context = GameContext()
        self.context.round_info = self.round_info

        # 初始化规则引擎和观测构建器
        rule_engine = Wuhan7P4LRuleEngine(self.context)
        observation_builder = Wuhan7P4LObservationBuilder(self.context)

        # 创建状态机，传递 logger
        self.state_machine = MahjongStateMachine(
            rule_engine=rule_engine,
            observation_builder=observation_builder,
            logger=self.logger,
            enable_logging=self.enable_logging
        )
        self.state_machine.set_context(self.context)

        # 记录游戏开始
        if self.logger:
            self.logger.start_game(self.current_game_id, {
                "seed": seed,
                "training_phase": self.training_phase,
                "num_players": self.num_players
            })

        # 重置游戏结束标志
        self._game_logged = False
        
        # 转换到初始状态
        self.state_machine.transition_to(GameStateType.INITIAL, self.context)
        
        # 执行初始状态（自动）
        self.state_machine.step(self.context, 'auto')
        
        # 设置当前agent
        self.agent_selection = self.state_machine.get_current_agent()
        
        # 返回第一个玩家的观测
        return self.observe(self.agent_selection), {}

    def agent_iter(self, num_steps: int = 0):
        """
        重载 PettingZoo 的 agent_iter 方法

        麻将的玩家轮转由状态机控制，而不是简单的列表迭代。
        该方法只产生当前 agent_selection 指向的玩家，玩家轮转
        完全由 step() 方法和状态机配合完成。

        Args:
            num_steps: 最大迭代步数（0 表示无限迭代直到游戏结束）

        Yields:
            当前需要动作的 agent 名称

        设计说明：
            - 游戏进行中：产生 agent_selection（由状态机根据
              current_player_idx 确定）
            - WAITING_RESPONSE 状态：状态机逐个更新响应者，
              agent_iter 对应产生每个响应者
            - 游戏结束：agents 列表为空，迭代自动终止

        PettingZoo 兼容性：
            完全兼容 AECEnv 规范，可以用于标准游戏循环：

            for agent in env.agent_iter():
                obs, reward, terminated, truncated, info = env.last()
                if terminated or truncated:
                    action = None
                else:
                    action = policy(obs)
                env.step(action)
        """
        if num_steps == 0:
            # 无限迭代模式：迭代直到 agents 列表为空
            while self.agents:
                yield self.agent_selection
        else:
            # 限制步数模式：最多产生 num_steps 个 agent
            for _ in range(num_steps):
                if not self.agents:
                    break
                yield self.agent_selection

    def step(self, action):
        """
        执行一步游戏

        环境职责：
        1. 接收并执行agent的动作
        2. 自动推进自动状态
        3. 在需要agent动作的状态停止
        4. 不做任何AI决策（由外部代码负责）

        Args:
            action: agent的动作（(action_type, parameter)元组）

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        current_agent = self.agent_selection

        # Reset cumulative rewards for the current agent
        if current_agent in self._cumulative_rewards:
            self._cumulative_rewards[current_agent] = 0.0

        agent_idx = self.agents_name_mapping[current_agent]

        # 转换action为MahjongAction对象
        try:
            mahjong_action = self._convert_action(action)
        except ValueError as e:
            print(f"无效动作: {e}")
            return self.observe(current_agent), -1.0, False, False, {'error': str(e)}

        # 执行状态机step
        try:
            next_state_type = self.state_machine.step(self.context, mahjong_action)
        except Exception as e:
            print(f"状态机执行错误: {e}")
            import traceback
            traceback.print_exc()
            return self.observe(current_agent), -1.0, True, False, {'error': str(e)}

        # 状态转换后立即更新 agent_selection（确保 observation 同步）
        if not self.state_machine.is_terminal():
            self.agent_selection = self.state_machine.get_current_agent()

        # 自动推进：只处理自动状态，在需要agent动作的状态停止
        while not self.state_machine.is_terminal():
            current_state = self.state_machine.current_state_type

            # 需要agent动作的四个状态 - 停止自动推进
            if current_state in [
                GameStateType.PLAYER_DECISION,
                GameStateType.MELD_DECISION,
                GameStateType.WAITING_RESPONSE,
                GameStateType.WAIT_ROB_KONG
            ]:
                break

            # 其他状态都是自动状态，使用'auto'推进
            try:
                next_state_type = self.state_machine.step(self.context, 'auto')
                if next_state_type is None:
                    # 终端状态
                    break
            except Exception as e:
                print(f"自动推进错误: {e}")
                break

        # 更新agent_selection
        if not self.state_machine.is_terminal():
            self.agent_selection = self.state_machine.get_current_agent()
        
        # 获取观测
        observation = self.observe(self.agent_selection)

        # 计算奖励（简化版本，实际应根据游戏结果计算）
        reward = self._calculate_reward(agent_idx)

        # 记录对局步骤（如果有 GameRecorder）
        if self.logger:
            # 获取 GameRecorder（支持直接是 GameRecorder 或在 CompositeLogger 中）
            game_recorder = None
            if isinstance(self.logger, GameRecorder):
                game_recorder = self.logger
            elif isinstance(self.logger, CompositeLogger):
                game_recorder = self.logger.get_logger(GameRecorder)

            if game_recorder:
                prev_observation = self.observe(current_agent)
                game_recorder.record_step(
                    agent=current_agent,
                    observation=prev_observation,
                    action={"type": action[0], "param": action[1]},
                    reward=reward,
                    next_observation=observation,
                    info={}
                )

        # 更新 rewards 字典（PettingZoo AECEnv 要求）
        self.rewards[current_agent] = reward

        # 检查是否结束
        terminated = self.state_machine.is_terminal()
        truncated = False
        info = self._get_info(agent_idx)

        # 游戏结束时更新 round_info（用于下一局确定庄家）并记录结果
        if terminated:
            self._update_round_info()

            # 记录游戏结束
            if self.logger and not self._game_logged:
                result = {
                    "winners": list(self.context.winner_ids) if self.context.is_win else [],
                    "is_flush": self.context.is_flush,
                    "win_way": self.context.win_way if self.context.is_win else None,
                    "total_steps": self._get_total_steps()
                }
                self.logger.end_game(result)
                self._game_logged = True

        # PettingZoo AEC规范：终端状态下移除所有agents
        if terminated or truncated:
            # 将所有agents标记为终止，并从agents列表中移除
            for agent in self.agents[:]:
                self.terminations[agent] = True
                self.agents.remove(agent)
            self.agent_selection = None  # 清空agent选择

        # Accumulate rewards for PettingZoo AECEnv compatibility
        self._accumulate_rewards()

        return observation, reward, terminated, truncated, info
    
    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        """
        为指定agent生成观测
        
        Args:
            agent: agent名称（如'player_0'）
        
        Returns:
            观测字典
        """
        agent_idx = self.agents_name_mapping[agent]
        
        # 懒加载观测
        if self.context.observation is None:
            self.context.observation = self.state_machine.observation_builder.build(
                agent_idx,
                self.context
            )
        
        # 应用信息可见度掩码
        observation = self._apply_visibility_mask(
            self.context.observation.copy(),
            agent_idx
        )
        
        return observation
    
    def _convert_action(self, action: Tuple[int, int]) -> MahjongAction:
        """
        转换action为MahjongAction对象
        
        Args:
            action: (action_type, parameter)元组
        
        Returns:
            MahjongAction对象
        
        Raises:
            ValueError: 如果action格式无效
        """
        if not isinstance(action, tuple) or len(action) != 2:
            raise ValueError(f"Action must be tuple of length 2, got {action}")
        
        action_type, parameter = action
        
        # 验证action_type
        if action_type < 0 or action_type >= self.ACTION_SPACE:
            raise ValueError(f"Invalid action_type: {action_type}")
        
        # 验证parameter
        if parameter < -1 or parameter >= self.PARAM_SPACE:
            raise ValueError(f"Invalid parameter: {parameter}")
        
        return MahjongAction(ActionType(action_type), parameter)
    
    def _apply_visibility_mask(self, observation: Dict[str, np.ndarray], agent_id: int) -> Dict[str, np.ndarray]:
        """
        应用信息可见度掩码
        
        根据训练阶段，屏蔽一些信息（如对手手牌）。
        
        Args:
            observation: 原始观测
            agent_id: 当前agent ID
        
        Returns:
            掩码后的观测
        """
        if self.training_phase == 1:
            # 阶段1：只保留私有手牌
            observation['global_hand'] = np.zeros(4 * 34, dtype=np.int8)
            observation['wall'].fill(34)
        
        elif self.training_phase == 3:
            # 阶段3：屏蔽对手手牌
            global_hand = observation['global_hand'].copy()
            for i in range(4):
                if i != agent_id:
                    global_hand[i * 34:(i + 1) * 34] = 0
            observation['global_hand'] = global_hand
            observation['wall'].fill(34)
        
        # 阶段2：随机屏蔽（这里简化处理）
        elif self.training_phase == 2:
            # 简化：使用阶段3的逻辑
            global_hand = observation['global_hand'].copy()
            for i in range(4):
                if i != agent_id:
                    global_hand[i * 34:(i + 1) * 34] = 0
            observation['global_hand'] = global_hand
            observation['wall'].fill(34)
        
        return observation
    
    def _calculate_reward(self, agent_id: int) -> float:
        """
        计算奖励
        
        简化版本，实际应根据游戏结果、分数、对手分数等计算。
        
        Args:
            agent_id: agent ID
        
        Returns:
            奖励值
        """
        # 简化奖励：
        # +1: 赢牌
        # -1: 输牌
        # 0: 其他
        
        if self.state_machine.current_state_type == GameStateType.WIN:
            if agent_id in self.context.winner_ids:
                return 1.0
            else:
                return -1.0
        elif self.state_machine.current_state_type == GameStateType.FLOW_DRAW:
            return 0.0
        else:
            return 0.0
    
    def _get_info(self, agent_id: int) -> Dict[str, Any]:
        """
        获取额外信息
        
        Args:
            agent_id: agent ID
        
        Returns:
            信息字典
        """
        info = {
            'current_player': self.context.current_player_idx,
            'is_terminal': self.state_machine.is_terminal(),
            'current_state': self.state_machine.current_state_type.name,
        }
        
        # 添加游戏结果
        if self.state_machine.is_terminal():
            if self.context.is_win:
                info['winners'] = self.context.winner_ids
                info['win_way'] = self.context.win_way
            elif self.context.is_flush:
                info['flush'] = True
        
        return info

    def _update_round_info(self):
        """
        游戏结束时更新 round_info，用于下一局确定庄家

        轮庄规则：
        - 庄家胡牌 → 庄家连庄
        - 荒庄（流局） → 庄家连庄
        - 闲家胡牌 → 闲家成为新庄
        """
        if self.context.is_win and self.context.winner_ids:
            # 有人胡牌
            win_position = self.context.winner_ids[0]
            is_dealer_win = (win_position == self.context.dealer_idx)
            self.round_info.advance_round(win_position, is_dealer_win)
        else:
            # 荒庄（流局）
            self.round_info.advance_round(None, False)

    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            # 简化：打印当前状态
            print(f"\n当前状态: {self.state_machine.current_state_type.name}")
            print(f"当前玩家: {self.context.current_player_idx}")
            print(f"牌墙剩余: {len(self.context.wall)}张")
            for i, player in enumerate(self.context.players):
                print(f"玩家{i}: {len(player.hand_tiles)}张牌, {len(player.melds)}组副露")
        elif self.render_mode == 'ansi':
            # ANSI渲染模式
            pass
    
    def close(self):
        """
        关闭环境

        确保即使游戏没有正常结束也记录日志。
        """
        # 如果游戏没有正常结束，记录未完成的游戏
        if self.logger and not self._game_logged and self.context is not None:
            result = {
                "winners": [],
                "is_flush": self.context.is_flush if hasattr(self.context, 'is_flush') else False,
                "incomplete": True,
                "total_steps": self._get_total_steps(),
                "reason": "env_closed_without_termination"
            }
            self.logger.end_game(result)
            self._game_logged = True

        # 关闭所有日志器
        if self.logger and hasattr(self.logger, 'close_all'):
            self.logger.close_all()
        elif self.logger and hasattr(self.logger, 'close'):
            self.logger.close()


# 使用示例
if __name__ == "__main__":
    print("MahjongEnv状态机集成示例")
    print("=" * 60)
    
    # 创建环境
    env = WuhanMahjongEnv(render_mode='human', training_phase=3)
    
    # 重置环境
    print("\n重置环境...")
    observation, info = env.reset(seed=42)
    print(f"初始agent: {env.agent_selection}")
    print(f"观测键: {list(observation.keys())}")
    
    # 模拟几步游戏
    print("\n" + "=" * 60)
    print("模拟游戏...")
    print("=" * 60)
    
    for i in range(5):
        print(f"\n第{i+1}轮:")
        print(f"  当前agent: {env.agent_selection}")
        
        # 获取可用动作（简化：假设DISCARD总是可用）
        # 实际应根据action_mask选择
        action = (ActionType.DISCARD.value, 0)  # 简化：总是打出第一张牌
        
        # 执行动作
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"  奖励: {reward}")
        print(f"  是否结束: {terminated}")
        
        if terminated or truncated:
            print("\n游戏结束！")
            break
        
        # 渲染
        env.render()
    
    # 关闭环境
    env.close()
    print("\n示例完成！")
