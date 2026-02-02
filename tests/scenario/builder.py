"""
场景测试框架 - 流式构建器

提供链式调用的流式接口用于构建测试场景。
"""

from typing import List, Optional, Dict, Any, Callable, TYPE_CHECKING
from tests.scenario.context import ScenarioContext, StepConfig
from src.mahjong_rl.core.constants import GameStateType, ActionType

if TYPE_CHECKING:
    from tests.scenario.builder import ScenarioBuilder


class StepBuilder:
    """步骤构建器

    用于配置单个测试步骤，支持链式调用。
    """

    def __init__(
        self,
        scenario_builder: 'ScenarioBuilder',
        scenario_context: ScenarioContext,
        step_number: int,
        description: str
    ):
        """初始化步骤构建器

        Args:
            scenario_builder: 场景构建器引用（用于代理方法）
            scenario_context: 场景上下文
            step_number: 步骤编号
            description: 步骤描述
        """
        self.scenario_builder = scenario_builder
        self.scenario = scenario_context
        self.step_config = StepConfig(
            step_number=step_number,
            description=description
        )

    def action(self, player: int, action_type: ActionType, param: int = -1) -> 'StepBuilder':
        """指定玩家动作

        Args:
            player: 玩家索引
            action_type: 动作类型
            param: 动作参数

        Returns:
            self，支持链式调用
        """
        self.step_config.is_action = True
        self.step_config.is_auto = False
        self.step_config.player = player
        self.step_config.action_type = action_type
        self.step_config.parameter = param
        return self

    def auto_advance(self) -> 'StepBuilder':
        """自动推进（用于自动状态）

        Returns:
            self，支持链式调用
        """
        self.step_config.is_action = False
        self.step_config.is_auto = True
        return self

    def expect_state(self, state: GameStateType) -> 'StepBuilder':
        """预期下一个状态

        Args:
            state: 预期状态

        Returns:
            self，支持链式调用
        """
        self.step_config.expect_state = state
        return self

    def expect_action_mask(self, available_actions: List[ActionType]) -> 'StepBuilder':
        """预期可用的动作类型

        Args:
            available_actions: 预期可用的动作类型列表

        Returns:
            self，支持链式调用
        """
        self.step_config.expect_action_mask_contains = available_actions
        return self

    def verify(self, description: str, validator: Callable) -> 'StepBuilder':
        """添加自定义验证条件

        Args:
            description: 验证描述
            validator: 验证函数，接收 GameContext，返回 bool

        Returns:
            self，支持链式调用
        """
        self.step_config.validators.append(validator)
        return self

    def verify_hand(self, player: int, expected_tiles: List[int]) -> 'StepBuilder':
        """验证玩家手牌

        Args:
            player: 玩家索引
            expected_tiles: 预期手牌（包含关系）

        Returns:
            self，支持链式调用
        """
        if self.step_config.verify_hand_tiles is None:
            self.step_config.verify_hand_tiles = {}
        self.step_config.verify_hand_tiles[player] = expected_tiles
        return self

    def verify_wall_count(self, expected: int) -> 'StepBuilder':
        """验证牌墙剩余数量

        Args:
            expected: 预期数量

        Returns:
            self，支持链式调用
        """
        self.step_config.verify_wall_count = expected
        return self

    def step(self, step_number: int, description: str) -> 'StepBuilder':
        """代理到 ScenarioBuilder.step()，支持链式调用

        Args:
            step_number: 步骤编号
            description: 步骤描述

        Returns:
            新的 StepBuilder 实例
        """
        return self.scenario_builder.step(step_number, description)

    def run(self) -> 'TestResult':
        """代理到 ScenarioBuilder.run()，执行测试场景

        Returns:
            TestResult 测试结果
        """
        return self.scenario_builder.run()

    def __enter__(self):
        """支持 with 语句"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 语句时将步骤添加到场景"""
        self.scenario.steps.append(self.step_config)
        return False


class ScenarioBuilder:
    """场景构建器

    用于流式构建测试场景。
    """

    def __init__(self, name: str):
        """初始化场景构建器

        Args:
            name: 场景名称
        """
        self.context = ScenarioContext(name=name)
        self._current_step_builder: Optional[StepBuilder] = None

    def description(self, desc: str) -> 'ScenarioBuilder':
        """设置场景描述

        Args:
            desc: 描述文本

        Returns:
            self，支持链式调用
        """
        self.context.description = desc
        return self

    def with_wall(self, tiles: List[int]) -> 'ScenarioBuilder':
        """设置牌墙顺序

        Args:
            tiles: 牌ID列表

        Returns:
            self，支持链式调用
        """
        self.context.wall = tiles.copy()
        return self

    def with_special_tiles(
        self,
        lazy: Optional[int] = None,
        skins: Optional[List[int]] = None
    ) -> 'ScenarioBuilder':
        """设置特殊牌

        Args:
            lazy: 赖子牌ID
            skins: 皮子牌ID列表 [skin1, skin2]

        Returns:
            self，支持链式调用
        """
        special_tiles = {}
        if lazy is not None:
            special_tiles['lazy'] = lazy
        if skins is not None and len(skins) >= 2:
            special_tiles['skins'] = [skins[0], skins[1]]

        if special_tiles:
            self.context.special_tiles = special_tiles
        return self

    def step(
        self,
        step_number: int,
        description: str
    ) -> 'StepBuilder':
        """开始一个新步骤

        Args:
            step_number: 步骤编号
            description: 步骤描述

        Returns:
            StepBuilder 实例

        Usage:
            builder.step(1, "第一步")
                .action(0, ActionType.DISCARD, 5)
                .expect_state(GameStateType.WAITING_RESPONSE)
        """
        # 自动添加上一个未添加的步骤（如果存在）
        self._add_pending_step()

        # 创建新的步骤构建器并保存引用，传递 self 引用用于代理
        self._current_step_builder = StepBuilder(self, self.context, step_number, description)
        return self._current_step_builder

    def expect_winner(self, winners: List[int]) -> 'ScenarioBuilder':
        """设置预期获胜者

        Args:
            winners: 获胜玩家索引列表

        Returns:
            self，支持链式调用
        """
        self.context.expect_winner = winners
        return self

    def with_initial_state(self, config: Dict[str, Any]) -> 'ScenarioBuilder':
        """设置自定义初始状态，绕过 InitialState 的自动初始化

        这允许用户完全控制游戏初始状态，包括：
        - 庄家位置和当前玩家
        - 每个玩家的手牌
        - 牌墙顺序
        - 特殊牌（赖子、皮子）
        - 庄家刚摸的牌（用于 PLAYER_DECISION 状态）

        Args:
            config: 初始状态配置字典
                - dealer_idx (int): 庄家位置 (0-3)
                - current_player_idx (int): 当前玩家 (0-3)
                - hands (Dict[int, List[int]]): 玩家手牌 {player_id: [tiles]}
                - wall (List[int]): 牌墙
                - special_tiles (Dict): 特殊牌 {lazy: int, skins: [int, int]}
                - last_drawn_tile (int, optional): 庄家刚摸的牌

        Returns:
            self，支持链式调用

        Example:
            ```python
            result = (
                ScenarioBuilder("自定义状态测试")
                .with_initial_state({
                    'dealer_idx': 0,
                    'current_player_idx': 0,
                    'hands': {
                        0: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6],  # 13张
                        1: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                        2: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                        3: [33, 33, 33, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    },
                    'wall': [11, 12, 13, ...],  # 剩余牌墙
                    'special_tiles': {'lazy': 8, 'skins': [7, 9]},
                    'last_drawn_tile': 6,
                })
                .run()
            )
            ```
        """
        self.context.initial_config = config
        return self

    def _add_pending_step(self):
        """添加当前待处理的步骤到场景

        这是方案 C 的核心实现：
        在创建新步骤或运行测试时，自动将上一个步骤添加到场景中。
        """
        if self._current_step_builder is not None:
            step_config = self._current_step_builder.step_config
            # 只添加不在场景中的步骤
            if step_config not in self.context.steps:
                self.context.steps.append(step_config)

    def run(self) -> 'TestResult':
        """执行测试场景

        Returns:
            TestResult 测试结果
        """
        from tests.scenario.executor import TestExecutor

        # 自动添加最后一个未添加的步骤
        self._add_pending_step()

        executor = TestExecutor(self.context)
        return executor.run()
