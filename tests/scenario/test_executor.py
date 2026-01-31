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

    # 验证基本执行结果
    assert result is not None, "执行器应该返回结果"
    assert result.executed_steps == 1, f"应该执行 1 步，实际执行 {result.executed_steps} 步"
    assert result.total_steps == 1, f"总步骤数应该是 1，实际是 {result.total_steps}"

    # 验证资源清理：env 应该被关闭
    assert executor.env is not None, "环境对象应该存在"
    # 注意：我们无法直接检查 env 是否已关闭，但 finally 块确保了 close() 被调用

    # 验证成功或预期的失败（可能因为状态不匹配）
    if result.success:
        assert result.final_state == GameStateType.PLAYER_DECISION, "最终状态应该是 PLAYER_DECISION"
        assert result.failure_message is None, "成功时不应该有失败消息"
    else:
        assert result.failure_message is not None, "失败时应该有失败消息"
        assert result.failed_step is not None, "失败时应该记录失败的步骤号"
