"""
测试 fast_mode 功能

验证快速模式下的行为：
- fast_mode=True 时不保存快照
- fast_mode=True 时 rollback 抛出异常
- fast_mode=False 时保留完整功能
"""

import pytest
import numpy as np

from example_mahjong_env import WuhanMahjongEnv


def test_fast_mode_disables_snapshots():
    """验证 fast_mode=True 时不保存快照"""
    env = WuhanMahjongEnv(fast_mode=True)
    env.reset()

    # 执行几步
    for _ in range(10):
        action = env.action_space(env.agent_selection)
        env.step(action)

    # 验证历史记录为空
    assert len(env.state_machine.get_history()) == 0, \
        f"Expected empty history in fast_mode, got {len(env.state_machine.get_history())} snapshots"


def test_fast_mode_rollback_raises_error():
    """验证 fast_mode=True 时 rollback 抛出异常"""
    env = WuhanMahjongEnv(fast_mode=True)
    env.reset()

    # 执行几步
    for _ in range(5):
        action = env.action_space(env.agent_selection)
        env.step(action)

    # 尝试回滚应该抛出 RuntimeError
    with pytest.raises(RuntimeError, match="fast_mode.*snapshots are disabled"):
        env.state_machine.rollback(1)


def test_normal_mode_keeps_snapshots():
    """验证普通模式仍保留快照功能"""
    env = WuhanMahjongEnv(fast_mode=False)
    env.reset()

    # 执行几步
    for _ in range(5):
        action = env.action_space(env.agent_selection)
        env.step(action)

    # 验证有历史记录
    history = env.state_machine.get_history()
    assert len(history) > 0, \
        f"Expected non-empty history in normal mode, got {len(history)} snapshots"

    # 验证快照结构
    snapshot = history[0]
    assert 'state_type' in snapshot
    assert 'context' in snapshot
    assert 'timestamp' in snapshot


def test_normal_mode_rollback_works():
    """验证普通模式下 rollback 功能正常"""
    env = WuhanMahjongEnv(fast_mode=False)
    env.reset()

    # 执行几步并记录状态
    initial_player = env.agent_selection

    for _ in range(3):
        action = env.action_space(env.agent_selection)
        env.step(action)

    # 回滚 1 步
    context = env.state_machine.rollback(1)

    # 验证返回了 GameContext
    assert context is not None
    assert hasattr(context, 'current_player_idx')
    assert hasattr(context, 'players')


def test_fast_mode_backward_compatibility():
    """验证向后兼容性：默认情况下 fast_mode=False"""
    env = WuhanMahjongEnv()  # 不传递 fast_mode
    env.reset()

    # 执行几步
    for _ in range(3):
        action = env.action_space(env.agent_selection)
        env.step(action)

    # 验证默认行为是保留快照
    history = env.state_machine.get_history()
    assert len(history) > 0, "Default behavior should keep snapshots"


def test_fast_mode_with_debug_mode():
    """验证 fast_mode 与 debug_mode 的兼容性"""
    # fast_mode=True, debug_mode=False
    env1 = WuhanMahjongEnv(fast_mode=True, enable_logging=False)
    env1.reset()
    assert env1.fast_mode is True
    assert env1.enable_logging is False

    # fast_mode=False, debug_mode=True
    env2 = WuhanMahjongEnv(fast_mode=False, enable_logging=True)
    env2.reset()
    assert env2.fast_mode is False
    assert env2.enable_logging is True

    # 快速模式应该没有快照
    for _ in range(5):
        action = env1.action_space(env1.agent_selection)
        env1.step(action)
    assert len(env1.state_machine.get_history()) == 0

    # 普通模式应该有快照
    for _ in range(5):
        action = env2.action_space(env2.agent_selection)
        env2.step(action)
    assert len(env2.state_machine.get_history()) > 0


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])
