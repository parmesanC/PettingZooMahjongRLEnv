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
from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import (
    Wuhan7P4LObservationBuilder,
)
from src.mahjong_rl.rules.round_info import RoundInfo

# 性能优化：缓存组件
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.action_validator import (
    ActionValidator,
)
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.win_detector import (
    WuhanMahjongWinChecker,
)
from src.mahjong_rl.optimization.mask_cache import ActionMaskCache

# 日志系统导入
from src.mahjong_rl.logging.base import ILogger
from src.mahjong_rl.logging import (
    CompositeLogger,
    FileLogger,
    GameRecorder,
    PerfMonitor,
    LogLevel,
    LogFormatter,
)


class WuhanMahjongEnv(AECEnv):
    """
    武汉麻将七皮四赖子环境 - 状态机集成版本（支持性能优化和向量化训练）

    该环境集成了完整的MahjongStateMachine，实现了PettingZoo AECEnv接口。
    状态机自动处理所有游戏逻辑，环境只需负责agent交互和观测传递。

    ## 主要特性

    ### 1. 状态机架构
    - 完整的状态转换管理（11种游戏状态）
    - 支持自动状态推进和手动决策状态
    - 内置游戏规则验证和动作掩码生成

    ### 2. 性能优化选项

    #### fast_mode (快速模式)
    - **启用**: fast_mode=True
    - **效果**: 禁用状态历史快照，减少深拷贝开销
    - **性能提升**: 预计减少状态转换开销20-40%
    - **限制**: 无法使用 rollback() 功能
    - **适用场景**: 纯训练模式，无需调试

    #### 向量化训练支持
    - **启用**: 配合 NFSPTrainer(use_vectorized_env=True, num_envs=N)
    - **效果**: 并行运行多个环境实例，提升数据收集效率
    - **适用场景**: 大规模训练，多GPU/多进程部署

    #### 日志和性能监控
    - **内置日志**: enable_logging=True/False
    - **性能监控**: enable_perf_monitor=True/False
    - **自定义日志器**: logger=自定义ILogger实例

    ### 3. 课程学习支持
    - **训练阶段**: training_phase=1/2/3
    - **阶段进度**: phase2_progress=0.0-1.0 (仅阶段2有效)
    - **S曲线掩码**: 渐进式信息可见度调整

    ## 推荐配置

    ### 训练模式（最优性能）
    ```python
    env = WuhanMahjongEnv(
        fast_mode=True,          # 禁用快照，提升速度
        enable_logging=False,    # 禁用日志，减少开销
        enable_perf_monitor=False  # 训练时可关闭监控
    )
    ```

    ### 调试模式（完整功能）
    ```python
    env = WuhanMahjongEnv(
        fast_mode=False,         # 启用快照，支持回滚
        enable_logging=True,     # 启用日志，便于追踪
        enable_perf_monitor=True  # 监控性能指标
    )
    ```

    ### 向量化训练
    ```python
    from src.drl.trainer import NFSPTrainer

    trainer = NFSPTrainer(
        use_vectorized_env=True,  # 启用向量化环境
        num_envs=4,               # 并行环境数量
        # ... 其他配置
    )
    ```

    ### 课程学习示例
    ```python
    # 阶段1：全知视角
    env = WuhanMahjongEnv(training_phase=1)

    # 阶段2：渐进式掩码，进度50%
    env = WuhanMahjongEnv(training_phase=2, phase2_progress=0.5)

    # 阶段3：真实状态
    env = WuhanMahjongEnv(training_phase=3)
    ```

    ## 性能指标

    启用性能监控后，系统会记录以下指标：
    - **step_time_ms**: 完整步骤执行时间
    - **state_transition_ms**: 状态转换时间
    - **observation_build_ms**: 观测构建时间
    - **memory_mb**: 内存使用峰值
    - **gc_count**: 垃圾回收次数
    - **throughput_steps_sec**: 每秒执行步数

    性能数据保存至 `performance/` 目录，可用于分析和优化。
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "wuhan_mahjong_v1.0"}

    # 动作空间：参数化动作
    ACTION_SPACE = len(ActionType)  # 11种动作类型
    PARAM_SPACE = 35  # 34种牌 + 1个通配符

    # 默认日志配置
    DEFAULT_LOG_CONFIG = {
        "file_logger": {"enabled": True, "log_dir": "logs", "level": "INFO"},
        "game_recorder": {"enabled": False, "replay_dir": "replays"},
        "perf_monitor": {"enabled": False, "perf_dir": "performance"},
    }

    def __init__(
        self,
        render_mode=None,
        training_phase=3,
        phase2_progress=0.0,  # 新增：阶段2的课程学习进度（0.0-1.0）
        enable_logging=True,
        log_config=None,
        logger=None,
        enable_perf_monitor=False,
        fast_mode=False,  # 新增：快速模式，禁用快照以提升性能
    ):
        """
        初始化麻将强化学习环境

        该环境实现了武汉麻将七皮四赖子玩法，基于状态机架构，完全兼容PettingZoo AECEnv接口。
        提供多种性能优化选项，包括fast_mode（禁用快照）和向量化训练支持。

        Args:
            render_mode (str, optional): 渲染模式，支持 'human'（命令行）或 'ansi' 格式，默认 None
            training_phase (int, optional): 训练阶段（1-3），影响观测信息可见度：
                                           1=全知视角，2=渐进式掩码，3=真实状态，默认 3
            phase2_progress (float, optional): 阶段2的课程学习进度（0.0-1.0），用于S曲线掩码概率，
                                              仅在 training_phase=2 时有效，默认 0.0
            enable_logging (bool, optional): 是否启用内置日志系统，默认 True
            log_config (dict, optional): 日志配置字典，覆盖默认配置 DEFAULT_LOG_CONFIG
            logger (ILogger, optional): 自定义日志器实例，如果提供则忽略 log_config，默认 None
            enable_perf_monitor (bool, optional): 是否启用性能监控器，记录详细性能指标，默认 False
            fast_mode (bool, optional): 快速模式，禁用状态快照以显著提升训练性能（减少20-40%开销），默认 False

        Note:
            - fast_mode=True 时无法使用 rollback() 功能，适用于纯训练场景
            - 推荐训练配置：fast_mode=True, enable_logging=False
            - 推荐调试配置：fast_mode=False, enable_logging=True
            - 向量化训练需配合 NFSPTrainer(use_vectorized_env=True, num_envs=N) 使用

        Example:
            >>> # 最快训练模式
            >>> env = WuhanMahjongEnv(fast_mode=True, enable_logging=False)
            >>>
            >>> # 完整功能调试模式
            >>> env = WuhanMahjongEnv(fast_mode=False, enable_logging=True)
            >>>
            >>> # 课程学习阶段2，进度50%
            >>> env = WuhanMahjongEnv(training_phase=2, phase2_progress=0.5)
        """
        super().__init__()

        # 验证配置一致性
        self._validate_config(
            training_phase=training_phase,
            phase2_progress=phase2_progress,
            enable_logging=enable_logging,
            enable_perf_monitor=enable_perf_monitor,
            fast_mode=fast_mode,
        )

        self.training_phase = training_phase
        self.phase2_progress = max(
            0.0, min(1.0, float(phase2_progress))
        )  # 限制在0-1范围
        self.num_players = 4
        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        self.agents = self.possible_agents[:]
        self.agents_name_mapping = {name: i for i, name in enumerate(self.agents)}

        # 定义观测空间
        self.observation_spaces = self._create_observation_spaces()
        self.action_spaces = {
            agent: spaces.Tuple(
                (
                    spaces.Discrete(self.ACTION_SPACE),  # 动作类型
                    spaces.Discrete(self.PARAM_SPACE),  # 动作参数
                )
            )
            for agent in self.possible_agents
        }

        # 初始化日志系统
        self.logger = self._setup_logger(
            logger, log_config, enable_logging, enable_perf_monitor
        )
        self.current_game_id: Optional[str] = None
        self._game_logged = False  # 跟踪游戏是否已记录结束

        # 初始化状态机
        self.context: GameContext = None
        self.state_machine: MahjongStateMachine = None
        self.enable_logging = enable_logging
        self.fast_mode = fast_mode  # 保存 fast_mode 设置

        # 性能优化：缓存组件（在 reset() 中创建）
        self._cached_validator: Optional[ActionValidator] = None
        self._cached_win_checker: Optional[WuhanMahjongWinChecker] = None
        self._cached_mask_cache: Optional[ActionMaskCache] = None

        # 初始化轮次信息（Env 内部管理）
        self.round_info = RoundInfo()

        self.render_mode = render_mode

    def _create_observation_spaces(self) -> Dict[str, spaces.Dict]:
        """创建观测空间"""
        observation_spaces = {}

        for agent in self.possible_agents:
            observation_spaces[agent] = spaces.Dict(
                {
                    "global_hand": spaces.MultiDiscrete([6] * (4 * 34)),
                    "private_hand": spaces.MultiDiscrete([6] * 34),
                    "discard_pool_total": spaces.MultiDiscrete([6] * 34),
                    "wall": spaces.MultiDiscrete([35] * 82),
                    "melds": spaces.Dict(
                        {
                            "action_types": spaces.MultiDiscrete([11] * 16),
                            "tiles": spaces.MultiDiscrete([35] * 256),
                            "group_indices": spaces.MultiDiscrete([4] * 32),
                        }
                    ),
                    "action_history": spaces.Dict(
                        {
                            "types": spaces.MultiDiscrete([11] * 80),
                            "params": spaces.MultiDiscrete([35] * 80),
                            "players": spaces.MultiDiscrete([4] * 80),
                        }
                    ),
                    "special_gangs": spaces.MultiDiscrete([8, 4, 5] * 4),
                    "current_player": spaces.MultiDiscrete([4]),
                    "fan_counts": spaces.MultiDiscrete([600] * 4),
                    "special_indicators": spaces.MultiDiscrete([34, 34]),
                    "remaining_tiles": spaces.Discrete(137),
                    "dealer": spaces.MultiDiscrete([4]),
                    "current_phase": spaces.Discrete(8),
                    "action_mask": spaces.MultiBinary(145),  # 扁平化动作掩码（145位）
                }
            )

        return observation_spaces

    def _setup_logger(
        self,
        logger: Optional[ILogger],
        log_config: Optional[Dict],
        enable_logging: bool,
        enable_perf_monitor: bool,
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
            loggers.append(
                FileLogger(log_dir=file_config.get("log_dir", "logs"), level=level)
            )

        # 对局记录器
        if config["game_recorder"]["enabled"]:
            recorder_config = config["game_recorder"]
            loggers.append(
                GameRecorder(replay_dir=recorder_config.get("replay_dir", "replays"))
            )

        # 性能监控器
        if config["perf_monitor"]["enabled"]:
            perf_config = config["perf_monitor"]
            loggers.append(
                PerfMonitor(
                    perf_dir=perf_config.get("perf_dir", "performance"), enabled=True
                )
            )

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
        if self.context and hasattr(self.context, "action_history"):
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

        # 创建状态机，传递 logger 和 fast_mode
        self.state_machine = MahjongStateMachine(
            rule_engine=rule_engine,
            observation_builder=observation_builder,
            logger=self.logger,
            enable_logging=self.enable_logging,
            fast_mode=self.fast_mode,  # 传递 fast_mode 参数
        )
        self.state_machine.set_context(self.context)

        # 性能优化：创建缓存的组件（每个 episode 一次）
        self._cached_validator = ActionValidator(self.context)
        self._cached_win_checker = WuhanMahjongWinChecker(self.context)
        self._cached_mask_cache = ActionMaskCache()

        # 注入缓存的组件到状态机和观测构建器
        self.state_machine.set_cached_components(
            validator=self._cached_validator, win_checker=self._cached_win_checker
        )
        observation_builder.set_cached_validator(self._cached_validator)

        # 注入缓存的组件到状态机和观测构建器
        self.state_machine.set_cached_components(
            validator=self._cached_validator, win_checker=self._cached_win_checker
        )
        observation_builder.set_cached_validator(self._cached_validator)

        # 记录游戏开始
        if self.logger:
            self.logger.start_game(
                self.current_game_id,
                {
                    "seed": seed,
                    "training_phase": self.training_phase,
                    "num_players": self.num_players,
                },
            )

        # 重置游戏结束标志
        self._game_logged = False

        # 转换到初始状态
        self.state_machine.transition_to(GameStateType.INITIAL, self.context)

        # 执行初始状态（自动）
        self.state_machine.step(self.context, "auto")

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
            return self.observe(current_agent), -1.0, False, False, {"error": str(e)}

        # 验证action_mask：确保agent选择的动作在可用动作范围内
        obs = self.observe(current_agent)
        action_mask = obs["action_mask"]

        if not self._is_action_mask_valid(mahjong_action, action_mask):
            # 记录警告
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Agent {current_agent} chose action not in action_mask: "
                f"{mahjong_action.action_type.name}({mahjong_action.parameter}). "
                f"Returning negative reward."
            )

            # 返回负奖励
            return obs, -1.0, False, False, {"error": "action not in mask"}

        # 执行状态机step
        # 注意：自动跳过逻辑（如所有玩家只能 PASS）已由状态机的 auto-skip pattern 处理
        # 不需要在此处额外检测和转换
        try:
            next_state_type = self.state_machine.step(self.context, mahjong_action)
        except Exception as e:
            print(f"状态机执行错误: {e}")
            import traceback

            traceback.print_exc()
            return self.observe(current_agent), -1.0, True, False, {"error": str(e)}

        # 状态转换后更新 agent_selection
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
                GameStateType.WAIT_ROB_KONG,
            ]:
                break

            # 其他状态都是自动状态，使用'auto'推进
            try:
                next_state_type = self.state_machine.step(self.context, "auto")
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
                    info={},
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

            # 计算所有玩家的最终得分作为 reward
            if self.context.final_scores:
                # 将实际分数归一化为 reward（除以100使范围在 [-5, +5]）
                for i, score in enumerate(self.context.final_scores):
                    agent_name = f"player_{i}"
                    self.rewards[agent_name] = score / 100.0
            else:
                # 兼容：如果没有final_scores，使用旧的简化逻辑
                if self.context.is_win:
                    for agent in self.possible_agents:
                        agent_idx = self.agents_name_mapping[agent]
                        if agent_idx in self.context.winner_ids:
                            self.rewards[agent] = 1.0
                        else:
                            self.rewards[agent] = -1.0
                else:
                    for agent in self.possible_agents:
                        self.rewards[agent] = 0.0

            # 记录游戏结束
            if self.logger and not self._game_logged:
                result = {
                    "winners": list(self.context.winner_ids)
                    if self.context.is_win
                    else [],
                    "is_flush": self.context.is_flush,
                    "win_way": self.context.win_way if self.context.is_win else None,
                    "total_steps": self._get_total_steps(),
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

        # 每次都为当前 agent 重新构建观测（不使用缓存）
        # 原因：不同 agent 需要不同的观测（private_hand 不同）
        observation = self.state_machine.observation_builder.build(
            agent_idx, self.context
        )

        # 应用信息可见度掩码
        observation = self._apply_visibility_mask(observation, agent_idx)

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

        # 对于WIN动作，强制参数为-1（与action_validator保持一致）
        if action_type == ActionType.WIN.value:
            parameter = -1

        return MahjongAction(ActionType(action_type), parameter)

    def _action_to_index(self, action: MahjongAction) -> int:
        """
        将 MahjongAction 转换为 action_mask 的索引

        Args:
            action: 动作对象

        Returns:
            action_mask 索引 (0-144)，如果动作类型无效则返回 -1
        """
        action_type = action.action_type
        parameter = action.parameter

        # DISCARD: 0-33
        if action_type == ActionType.DISCARD:
            return parameter

        # CHOW: 34-36
        elif action_type == ActionType.CHOW:
            return 34 + parameter

        # PONG: 37
        elif action_type == ActionType.PONG:
            return 37

        # KONG_EXPOSED: 38
        elif action_type == ActionType.KONG_EXPOSED:
            return 38

        # KONG_SUPPLEMENT: 39-72
        elif action_type == ActionType.KONG_SUPPLEMENT:
            return 39 + parameter

        # KONG_CONCEALED: 73-106
        elif action_type == ActionType.KONG_CONCEALED:
            return 73 + parameter

        # KONG_RED: 107
        elif action_type == ActionType.KONG_RED:
            return 107

        # KONG_SKIN: 108-141
        elif action_type == ActionType.KONG_SKIN:
            return 108 + parameter

        # KONG_LAZY: 142
        elif action_type == ActionType.KONG_LAZY:
            return 142

        # WIN: 143
        elif action_type == ActionType.WIN:
            return 143

        # PASS: 144
        elif action_type == ActionType.PASS:
            return 144

        else:
            return -1

    def _is_action_mask_valid(
        self, action: MahjongAction, action_mask: np.ndarray
    ) -> bool:
        """
        检查动作是否在 action_mask 中被标记为可用

        Args:
            action: 动作对象
            action_mask: 动作掩码 (145位)

        Returns:
            True 如果 action_mask 对应位置为1，False 否则
        """
        # 将动作转换为 mask 索引
        action_index = self._action_to_index(action)

        # 检查索引是否在范围内
        if action_index < 0 or action_index >= len(action_mask):
            return False

        return action_mask[action_index] == 1

    def _get_masking_probability(self) -> float:
        """
        根据progress计算S曲线掩码概率

        S曲线特性：
        - progress=0.0 时，概率接近 0（接近阶段1全知）
        - progress=0.5 时，概率=0.5（过渡中点）
        - progress=1.0 时，概率接近 1（接近阶段3完全掩码）

        Returns:
            掩码概率 (0.0-1.0)
        """
        if self.training_phase != 2:
            return 0.0 if self.training_phase == 1 else 1.0

        # S曲线：sigmoid函数，6*(x-0.5) 让曲线在中间变化更明显
        import math

        sigmoid = 1 / (1 + math.exp(-6 * (self.phase2_progress - 0.5)))
        return sigmoid

    def _apply_visibility_mask(
        self, observation: Dict[str, np.ndarray], agent_id: int
    ) -> Dict[str, np.ndarray]:
        """
        应用信息可见度掩码（课程学习）

        阶段设计：
        - 阶段1（全知视角）：所有信息可见
        - 阶段2（渐进式）：随progress逐渐增加掩码
        - 阶段3（真实状态）：只可见自己手牌和公共信息

        特殊值约定：
        - global_hand: 5 表示"未知"
        - wall: 34 表示"未知"
        - melds.tiles: 暗杠的牌设为 34（与wall一致）

        Args:
            observation: 原始观测
            agent_id: 当前agent ID

        Returns:
            掩码后的观测
        """
        if self.training_phase == 1:
            # 阶段1：全知视角，不做任何掩码
            pass

        elif self.training_phase == 2:
            # 阶段2：渐进式随机掩码（关联决策）
            mask_prob = self._get_masking_probability()

            # 单一随机决策：决定是否应用所有掩码（保持状态一致性）
            if np.random.random() < mask_prob:
                # 掩码对手手牌
                global_hand = observation["global_hand"].copy()
                for i in range(4):
                    if i != agent_id:
                        global_hand[i * 34 : (i + 1) * 34] = 5
                observation["global_hand"] = global_hand

                # 掩码牌墙
                observation["wall"].fill(34)

                # 掩码对手暗杠的牌
                tiles = observation["melds"]["tiles"].copy()
                action_types = observation["melds"]["action_types"]
                KONG_CONCEALED = 5  # ActionType.KONG_CONCEALED.value

                for player_id in range(4):
                    if player_id == agent_id:
                        continue
                    for meld_idx in range(4):
                        idx = player_id * 4 + meld_idx
                        if action_types[idx] == KONG_CONCEALED:
                            # 将这个暗杠的4张牌位置设为34
                            base_tile_idx = (player_id * 4 * 4 + meld_idx * 4) * 34
                            for tile_pos in range(4):
                                tile_start = base_tile_idx + tile_pos * 34
                                tiles[tile_start : tile_start + 34] = 34
                observation["melds"]["tiles"] = tiles

        elif self.training_phase == 3:
            # 阶段3：真实状态，完全掩码
            # 掩码对手手牌
            global_hand = observation["global_hand"].copy()
            for i in range(4):
                if i != agent_id:
                    global_hand[i * 34 : (i + 1) * 34] = 5
            observation["global_hand"] = global_hand

            # 掩码牌墙
            observation["wall"].fill(34)

            # 掩码对手暗杠的牌
            tiles = observation["melds"]["tiles"].copy()
            action_types = observation["melds"]["action_types"]
            KONG_CONCEALED = 5

            for player_id in range(4):
                if player_id == agent_id:
                    continue
                for meld_idx in range(4):
                    idx = player_id * 4 + meld_idx
                    if action_types[idx] == KONG_CONCEALED:
                        base_tile_idx = (player_id * 4 * 4 + meld_idx * 4) * 34
                        for tile_pos in range(4):
                            tile_start = base_tile_idx + tile_pos * 34
                            tiles[tile_start : tile_start + 34] = 34
            observation["melds"]["tiles"] = tiles

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
            "current_player": self.context.current_player_idx,
            "is_terminal": self.state_machine.is_terminal(),
            "current_state": self.state_machine.current_state_type.name,
        }

        # 添加游戏结果
        if self.state_machine.is_terminal():
            if self.context.is_win:
                info["winners"] = self.context.winner_ids
                info["win_way"] = self.context.win_way
            elif self.context.is_flush:
                info["flush"] = True

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
            is_dealer_win = win_position == self.context.dealer_idx
            self.round_info.advance_round(win_position, is_dealer_win)
        else:
            # 荒庄（流局）
            self.round_info.advance_round(None, False)

    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            # 简化：打印当前状态
            print(f"\n当前状态: {self.state_machine.current_state_type.name}")
            print(f"当前玩家: {self.context.current_player_idx}")
            print(f"牌墙剩余: {len(self.context.wall)}张")
            for i, player in enumerate(self.context.players):
                print(
                    f"玩家{i}: {len(player.hand_tiles)}张牌, {len(player.melds)}组副露"
                )
        elif self.render_mode == "ansi":
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
                "is_flush": self.context.is_flush
                if hasattr(self.context, "is_flush")
                else False,
                "incomplete": True,
                "total_steps": self._get_total_steps(),
                "reason": "env_closed_without_termination",
            }
            self.logger.end_game(result)
            self._game_logged = True

        # 关闭所有日志器
        if self.logger and hasattr(self.logger, "close_all"):
            self.logger.close_all()
        elif self.logger and hasattr(self.logger, "close"):
            self.logger.close()

    def _validate_config(
        self,
        training_phase: int,
        phase2_progress: float,
        enable_logging: bool,
        enable_perf_monitor: bool,
        fast_mode: bool,
    ) -> None:
        """
        验证环境配置的一致性，防止不合理组合

        Args:
            training_phase: 训练阶段 (1-3)
            phase2_progress: 阶段2进度 (0.0-1.0)
            enable_logging: 是否启用日志
            enable_perf_monitor: 是否启用性能监控
            fast_mode: 是否启用快速模式

        Raises:
            ValueError: 如果配置存在不合理组合
            Warning: 如果配置可能影响性能但不会导致错误
        """
        import warnings

        # 1. 基本参数范围验证
        if not 1 <= training_phase <= 3:
            raise ValueError(
                f"training_phase 必须在 1-3 范围内，当前为 {training_phase}"
            )

        if not 0.0 <= phase2_progress <= 1.0:
            raise ValueError(
                f"phase2_progress 必须在 0.0-1.0 范围内，当前为 {phase2_progress}"
            )

        # 2. 逻辑一致性验证
        if training_phase == 2:
            if phase2_progress == 0.0:
                warnings.warn(
                    "phase2_progress=0.0 且 training_phase=2：这相当于全知视角（阶段1）。"
                    "考虑使用 training_phase=1 以获得明确语义。",
                    UserWarning,
                )
            elif phase2_progress == 1.0:
                warnings.warn(
                    "phase2_progress=1.0 且 training_phase=2：这相当于真实状态（阶段3）。"
                    "考虑使用 training_phase=3 以获得明确语义。",
                    UserWarning,
                )

        # 3. 性能配置建议（只打印一次）
        if fast_mode and not enable_logging:
            # 最优训练配置，打印信息但不警告（仅第一次）
            if not hasattr(WuhanMahjongEnv, '_config_printed'):
                print(f"[OK] 性能配置：fast_mode=True, enable_logging=False (最优训练性能)")
                WuhanMahjongEnv._config_printed = True
        elif fast_mode and enable_logging:
            warnings.warn(
                "fast_mode=True 但 enable_logging=True：日志会降低训练速度。"
                "如果不需要调试日志，建议使用 enable_logging=False。",
                UserWarning,
            )
        elif not fast_mode and enable_logging:
            if not hasattr(WuhanMahjongEnv, '_debug_config_printed'):
                print(f"[!] 配置：fast_mode=False, enable_logging=True (完整功能调试模式)")
                WuhanMahjongEnv._debug_config_printed = True

        # 4. 性能监控配置建议
        if enable_perf_monitor and not enable_logging:
            warnings.warn(
                "enable_perf_monitor=True 但 enable_logging=False："
                "性能监控器需要日志系统支持才能记录数据。"
                "建议同时启用 enable_logging=True 或禁用性能监控。",
                UserWarning,
            )
        elif enable_perf_monitor and fast_mode:
            warnings.warn(
                "enable_perf_monitor=True 且 fast_mode=True："
                "性能监控会增加一些开销，可能影响 fast_mode 的优化效果。"
                "长期训练时可定期启用以检测性能问题。",
                UserWarning,
            )

        # 5. 阶段2进度与阶段不匹配警告
        if training_phase != 2 and phase2_progress != 0.0:
            warnings.warn(
                f"phase2_progress={phase2_progress} 但 training_phase={training_phase}："
                "phase2_progress 仅在 training_phase=2 时有效，当前设置将被忽略。",
                UserWarning,
            )


# 使用示例
if __name__ == "__main__":
    print("MahjongEnv状态机集成示例")
    print("=" * 60)

    # 创建环境
    # training_phase: 1=全知视角, 2=渐进式掩码(需配合phase2_progress), 3=真实状态
    env = WuhanMahjongEnv(render_mode="human", training_phase=3)

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
        print(f"\n第{i + 1}轮:")
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
