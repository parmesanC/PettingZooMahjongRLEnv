"""
测试 agent_iter() 的实现

测试 WuhanMahjongEnv.agent_iter() 方法是否正确实现了
麻将玩家轮转规则。
"""

import pytest
import random
from example_mahjong_env import WuhanMahjongEnv


def test_agent_iter_produces_current_selection():
    """测试 agent_iter() 产生当前的 agent_selection"""
    env = WuhanMahjongEnv()
    obs, info = env.reset()

    # 第一次迭代应该产生初始 agent_selection
    agents = list(env.agent_iter(num_steps=1))
    assert len(agents) == 1
    assert agents[0] == env.agent_selection, f"Expected {env.agent_selection}, got {agents[0]}"


def test_agent_iter_matches_selection_after_reset():
    """测试 reset 后 agent_iter() 产生正确的 agent"""
    env = WuhanMahjongEnv()
    obs, info = env.reset(seed=42)

    # agent_iter 应该产生当前 agent_selection
    agent = next(env.agent_iter(num_steps=1))
    assert agent == env.agent_selection


def test_agent_iter_updates_after_step():
    """测试 step() 后 agent_iter() 产生更新的 agent"""
    env = WuhanMahjongEnv()
    obs, info = env.reset()
    initial_agent = env.agent_selection

    # 执行一步
    action = (0, 0)  # 简化动作
    obs, reward, terminated, truncated, info = env.step(action)

    # 如果游戏未结束，agent_iter 应该产生新的 agent
    if not terminated and not truncated:
        next_agent = next(env.agent_iter(num_steps=1))
        assert next_agent == env.agent_selection


def test_agent_iter_terminates_when_game_ends():
    """测试游戏结束时 agent_iter() 终止"""
    env = WuhanMahjongEnv()
    obs, info = env.reset()

    # 模拟游戏直到结束
    agent_count = 0
    for agent in env.agent_iter():
        agent_count += 1
        obs, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            # 游戏结束，action 为 None
            break
        else:
            # 简化：随机选择动作
            action = (random.randint(0, 10), random.randint(0, 34))

        env.step(action)

        # 安全退出
        if terminated or truncated or agent_count > 1000:
            break

    # 游戏结束后 agents 列表应该为空
    if terminated or truncated:
        assert len(env.agents) == 0, f"Expected empty agents, got {env.agents}"


def test_agent_iter_with_num_steps():
    """测试带 num_steps 参数的 agent_iter()"""
    env = WuhanMahjongEnv()
    obs, info = env.reset()

    # 请求 5 步
    agents = list(env.agent_iter(num_steps=5))
    assert len(agents) == 5, f"Expected 5 agents, got {len(agents)}"

    # 所有产生的 agent 都应该是当前 agent_selection
    # （因为游戏状态在第一次迭代后不会改变）
    for agent in agents:
        assert agent in env.possible_agents


def test_full_game_loop_with_agent_iter():
    """测试完整的游戏循环"""
    env = WuhanMahjongEnv(render_mode=None)  # 不使用渲染模式
    obs, info = env.reset(seed=42)

    step_count = 0
    max_steps = 200  # 限制最大步数

    for agent in env.agent_iter():
        step_count += 1

        # 获取观测和奖励
        obs, reward, terminated, truncated, info = env.last()

        # 验证 agent 与 agent_selection 一致
        assert agent == env.agent_selection, \
            f"Agent mismatch: iter={agent}, selection={env.agent_selection}"

        if terminated or truncated:
            # 游戏结束
            break
        else:
            # 简化：随机选择动作
            action = (random.randint(0, 10), random.randint(0, 34))

        # 执行动作
        env.step(action)

        if step_count >= max_steps:
            break

    print(f"测试完成，共 {step_count} 步")
    assert step_count > 0


def test_agent_iter_empty_when_terminated():
    """测试游戏终止后 agent_iter() 不产生任何 agent"""
    env = WuhanMahjongEnv()
    obs, info = env.reset()

    # 运行到游戏结束
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            # 游戏结束，不再执行 step
            break
        else:
            action = (random.randint(0, 10), random.randint(0, 34))
            env.step(action)

    # 游戏结束后，agents 列表应该为空
    # 此时 agent_iter() 应该不产生任何 agent
    agents_after_end = list(env.agent_iter(num_steps=1))
    assert len(agents_after_end) == 0, \
        f"Expected no agents after game end, got {agents_after_end}"


def test_agent_iter_returns_player_names():
    """测试 agent_iter() 返回正确的玩家名称格式"""
    env = WuhanMahjongEnv()
    obs, info = env.reset()

    # 检查返回的 agent 名称格式
    agent = next(env.agent_iter(num_steps=1))
    assert agent in ["player_0", "player_1", "player_2", "player_3"], \
        f"Invalid agent name: {agent}"


if __name__ == "__main__":
    # 运行测试
    print("运行 agent_iter() 测试...")
    print("=" * 60)

    test_agent_iter_produces_current_selection()
    print("✓ test_agent_iter_produces_current_selection")

    test_agent_iter_matches_selection_after_reset()
    print("✓ test_agent_iter_matches_selection_after_reset")

    test_agent_iter_updates_after_step()
    print("✓ test_agent_iter_updates_after_step")

    test_agent_iter_with_num_steps()
    print("✓ test_agent_iter_with_num_steps")

    test_agent_iter_returns_player_names()
    print("✓ test_agent_iter_returns_player_names")

    test_full_game_loop_with_agent_iter()
    print("✓ test_full_game_loop_with_agent_iter")

    print("=" * 60)
    print("所有测试通过！")
