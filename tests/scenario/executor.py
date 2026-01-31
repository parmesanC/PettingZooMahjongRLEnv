"""
场景测试框架 - 测试执行器

负责执行配置好的测试场景，验证状态转换和游戏状态。
"""

from typing import Optional
from copy import deepcopy
from tests.scenario.context import ScenarioContext, StepConfig, TestResult
from src.mahjong_rl.core.constants import GameStateType, ActionType
from src.mahjong_rl.core.mahjong_action import MahjongAction


class TestExecutor:
    """测试执行器

    执行测试场景，收集验证结果。
    """

    def __init__(self, scenario: ScenarioContext):
        """初始化执行器

        Args:
            scenario: 测试场景配置
        """
        self.scenario = scenario
        self.env = None
        self.result = TestResult(scenario_name=scenario.name, success=False)

    def run(self) -> TestResult:
        """执行测试场景

        Returns:
            测试结果
        """
        try:
            # 延迟导入避免循环依赖
            from example_mahjong_env import WuhanMahjongEnv

            # 创建环境
            self.env = WuhanMahjongEnv(
                render_mode=None,
                training_phase=3,  # 完整信息
                enable_logging=False  # 测试时关闭日志
            )

            # 重置环境
            self.env.reset(seed=self.scenario.seed)

            # 配置牌墙
            if self.scenario.wall:
                self.env.context.wall.clear()
                self.env.context.wall.extend(self.scenario.wall)

            # 配置特殊牌
            if self.scenario.special_tiles:
                if 'lazy' in self.scenario.special_tiles:
                    self.env.context.lazy_tile = self.scenario.special_tiles['lazy']
                if 'skins' in self.scenario.special_tiles:
                    skins = self.scenario.special_tiles['skins']
                    if len(skins) >= 2:
                        self.env.context.skin_tile = [skins[0], skins[1]]

            self.result.total_steps = len(self.scenario.steps)

            # 执行每个步骤
            for step_config in self.scenario.steps:
                self._execute_step(step_config)
                self.result.executed_steps += 1

            # 执行最终验证
            if self.scenario.final_validators:
                for validator in self.scenario.final_validators:
                    if not validator(self.env.context):
                        raise AssertionError(f"最终验证失败: {validator.__name__}")

            # 验证获胜者
            if self.scenario.expect_winner is not None:
                if set(self.env.context.winner_ids) != set(self.scenario.expect_winner):
                    raise AssertionError(
                        f"获胜者验证失败: 预期 {self.scenario.expect_winner}, "
                        f"实际 {self.env.context.winner_ids}"
                    )

            self.result.success = True
            self.result.final_state = self.env.state_machine.current_state_type

        except Exception as e:
            self.result.success = False
            self.result.failed_step = self.result.executed_steps + 1
            self.result.failure_message = str(e)

            # 保存快照用于调试
            if self.env and self.env.context:
                self.result.final_context_snapshot = self._create_snapshot()

        return self.result

    def _execute_step(self, step: StepConfig):
        """执行单个步骤

        Args:
            step: 步骤配置

        Raises:
            AssertionError: 验证失败
            Exception: 执行错误
        """
        print(f"\n步骤 {step.step_number}: {step.description}")

        if step.is_auto:
            # 自动步骤：状态机自动推进
            self._auto_advance(step)
        elif step.is_action:
            # 动作步骤：执行指定动作
            self._execute_action(step)

        # 执行验证
        self._run_validations(step)

    def _execute_action(self, step: StepConfig):
        """执行动作步骤

        Args:
            step: 步骤配置

        Raises:
            AssertionError: 验证失败
        """
        # 构造动作
        action = (step.action_type.value, step.parameter)

        # 执行动作
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 打印执行结果
        print(f"  玩家 {step.player} 执行: {step.action_type.name}({step.parameter})")
        print(f"  当前状态: {self.env.state_machine.current_state_type.name}")

    def _auto_advance(self, step: StepConfig):
        """自动推进步骤

        Args:
            step: 步骤配置

        Raises:
            AssertionError: 验证失败
        """
        # 自动状态已由 env.step() 内部处理
        # 这里只需要验证当前状态
        print(f"  自动推进到: {self.env.state_machine.current_state_type.name}")

    def _run_validations(self, step: StepConfig):
        """运行所有验证

        Args:
            step: 步骤配置

        Raises:
            AssertionError: 验证失败
        """
        context = self.env.context

        # 验证状态
        if step.expect_state:
            actual = self.env.state_machine.current_state_type
            if actual != step.expect_state:
                raise AssertionError(
                    f"状态验证失败: 预期 {step.expect_state.name}, "
                    f"实际 {actual.name if actual else None}"
                )

        # 验证 action_mask
        if step.expect_action_mask_contains:
            mask = context.action_mask
            for action_type in step.expect_action_mask_contains:
                action = MahjongAction(action_type, -1)
                index = self.env._action_to_index(action)
                if index < 0 or index >= len(mask) or mask[index] != 1:
                    raise AssertionError(
                        f"action_mask 验证失败: {action_type.name} 不在可用动作中"
                    )

        # 执行自定义验证器
        for validator in step.validators:
            if not validator(context):
                raise AssertionError(f"验证器失败: {validator.__name__}")

        # 快捷验证：手牌
        if step.verify_hand_tiles:
            for player_id, tiles in step.verify_hand_tiles.items():
                validator = hand_contains(player_id, tiles)
                if not validator(context):
                    raise AssertionError(f"手牌验证失败: 玩家 {player_id}")

        # 快捷验证：牌墙数量
        if step.verify_wall_count is not None:
            validator = wall_count_equals(step.verify_wall_count)
            if not validator(context):
                raise AssertionError(f"牌墙数量验证失败")

        # 快捷验证：弃牌堆
        if step.verify_discard_pile_contains:
            for tile in step.verify_discard_pile_contains:
                validator = discard_pile_contains(tile)
                if not validator(context):
                    raise AssertionError(f"弃牌堆验证失败: 牌 {tile}")

    def _create_snapshot(self) -> dict:
        """创建上下文快照用于调试

        Returns:
            快照字典
        """
        context = self.env.context
        return {
            'current_state': context.current_state.name if context.current_state else None,
            'current_player': context.current_player_idx,
            'wall_count': len(context.wall),
            'discard_pile': context.discard_pile[-10:] if context.discard_pile else [],  # 最后10张
            'player_hand_counts': [len(p.hand_tiles) for p in context.players],
            'winner_ids': context.winner_ids if hasattr(context, 'winner_ids') else [],
        }


# 导入验证器函数用于快捷验证
from tests.scenario.validators import hand_contains, wall_count_equals, discard_pile_contains
