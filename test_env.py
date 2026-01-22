#!/usr/bin/env python3
"""
简单测试脚本 - 验证环境是否能正常工作
"""

import sys
sys.path.insert(0, '.')

from example_mahjong_env import WuhanMahjongEnv

def test_env():
    """测试环境基本功能"""
    print("测试1: 环境初始化")
    try:
        env = WuhanMahjongEnv(render_mode=None, training_phase=3, enable_logging=False)
        print("  ✓ 环境创建成功")
        print(f"  ✓ agent数量: {len(env.agents)}")
        print(f"  ✓ action_space: {env.action_spaces}")
    except Exception as e:
        print(f"  ✗ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n测试2: 环境重置")
    try:
        obs, info = env.reset(seed=42)
        print("  ✓ 环境重置成功")
        print(f"  ✓ agent_selection: {env.agent_selection}")
        print(f"  ✓ observation keys: {list(obs.keys())}")
    except Exception as e:
        print(f"  ✗ 环境重置失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n测试3: 获取观测")
    try:
        obs = env.observe(env.agent_selection)
        print("  ✓ 获取观测成功")
        for key, value in obs.items():
            if isinstance(value, (list, dict)):
                print(f"  ✓ {key}: {type(value).__name__}")
            else:
                print(f"  ✓ {key}: {type(value).__name__} (shape: {getattr(value, 'shape', 'N/A')})")
    except Exception as e:
        print(f"  ✗ 获取观测失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n测试4: 执行动作")
    try:
        obs, reward, terminated, truncated, info = env.last()
        print("  ✓ last()调用成功")
        print(f"  ✓ observation keys: {list(obs.keys())}")
        print(f"  ✓ reward: {reward}")
        print(f"  ✓ terminated: {terminated}")
        print(f"  ✓ truncated: {truncated}")
        print(f"  ✓ info keys: {list(info.keys())}")
    except Exception as e:
        print(f"  ✗ last()调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n测试5: 执行step")
    try:
        action = (0, 5)
        obs, reward, terminated, truncated, info = env.step(action)
        print("  ✓ step()调用成功")
        print(f"  ✓ agent_selection: {env.agent_selection}")
    except Exception as e:
        print(f"  ✗ step()调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    env.close()
    print("\n所有测试完成！")
    return True

if __name__ == '__main__':
    test_env()
