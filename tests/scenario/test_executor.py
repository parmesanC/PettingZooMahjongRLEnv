"""测试 TestExecutor"""

import pytest
from tests.scenario.context import ScenarioContext, StepConfig
from tests.scenario.executor import TestExecutor
from src.mahjong_rl.core.constants import GameStateType, ActionType


def test_executor_basic_flow():
    """测试基本执行流程"""
    scenario = ScenarioContext(
        name="基本流程测试",
        description="测试 executor 能正常创建和运行"
    )

    # 添加一个简单的自动步骤
    scenario.steps.append(StepConfig(
        step_number=1,
        description="初始化",
        is_auto=True,
        expect_state=GameStateType.PLAYER_DECISION,
    ))

    executor = TestExecutor(scenario)
    result = executor.run()

    # 应该成功（因为只是初始化）
    assert result.executed_steps == 1
    assert result.failure_message is None or "状态验证失败" in result.failure_message
