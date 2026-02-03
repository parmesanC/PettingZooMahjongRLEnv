"""
测试reward系统 - 验证游戏结束时所有玩家都能获得正确的reward
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from example_mahjong_env import WuhanMahjongEnv


def test_all_players_get_reward_on_win():
    """测试胡牌时所有4个玩家都能获得reward"""
    env = WuhanMahjongEnv(render_mode=None, training_phase=3, enable_logging=False)
    env.reset(seed=42)

    # 运行游戏直到结束（使用简单的随机策略）
    terminated = False
    step_count = 0
    max_steps = 1000

    while not terminated and step_count < max_steps:
        current_agent = env.agent_selection
        obs = env.observe(current_agent)
        action_mask = obs['action_mask']

        # 找到第一个可用动作
        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            break

        action_idx = valid_actions[0]

        # 将action_idx转换为(action_type, parameter)
        if action_idx < 34:  # DISCARD
            action = (0, action_idx)
        elif action_idx == 143:  # WIN
            action = (10, -1)
        elif action_idx == 144:  # PASS
            action = (11, -1)
        else:
            action = (0, 0)  # 默认DISCARD第一张

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

    # 验证：游戏应该结束
    assert terminated or truncated, "游戏应该在最大步数内结束"

    # 验证：所有4个玩家都应该有reward
    for agent in env.possible_agents:
        assert agent in env.rewards, f"{agent} 应该有reward"
        # reward不应该全是初始值0（除非流局且没有人听牌）
        # 这里我们只验证key存在

    # 验证：如果是胡牌，rewards总和应该为0（零和）
    if env.context.is_win and env.context.final_scores:
        reward_sum = sum(env.rewards.values())
        assert abs(reward_sum) < 0.01, f"胡牌时reward总和应为0，实际为{reward_sum}"

    print(f"游戏在 {step_count} 步后结束")
    print(f"最终rewards: {env.rewards}")
    if env.context.final_scores:
        print(f"原始分数: {env.context.final_scores}")

    env.close()


def test_reward_matches_final_scores():
    """测试reward应该等于final_scores除以100"""
    env = WuhanMahjongEnv(render_mode=None, training_phase=3, enable_logging=False)
    env.reset(seed=123)

    # 运行游戏直到结束
    terminated = False
    step_count = 0
    max_steps = 1000

    while not terminated and step_count < max_steps:
        current_agent = env.agent_selection
        obs = env.observe(current_agent)
        action_mask = obs['action_mask']

        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            break

        action_idx = valid_actions[0]

        if action_idx < 34:
            action = (0, action_idx)
        elif action_idx == 143:
            action = (10, -1)
        elif action_idx == 144:
            action = (11, -1)
        else:
            action = (0, 0)

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

    # 验证：如果有final_scores，reward应该匹配
    if env.context.final_scores:
        for i, score in enumerate(env.context.final_scores):
            agent_name = f"player_{i}"
            expected_reward = score / 100.0
            actual_reward = env.rewards.get(agent_name, 0.0)
            assert abs(actual_reward - expected_reward) < 0.01, \
                f"{agent_name} reward不匹配: 期望{expected_reward}, 实际{actual_reward}"

    env.close()


if __name__ == "__main__":
    print("测试1: 验证所有玩家都能获得reward")
    test_all_players_get_reward_on_win()
    print("✓ 测试1通过\n")

    print("测试2: 验证reward与final_scores匹配")
    test_reward_matches_final_scores()
    print("✓ 测试2通过\n")

    print("所有测试通过！")
