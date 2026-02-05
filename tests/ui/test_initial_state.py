#!/usr/bin/env python3
"""
测试初始状态管理器功能
"""

import sys
sys.path.insert(0, '.')

from src.mahjong_rl.web.initial_state_manager import InitialStateManager


def test_initial_state_manager():
    """测试初始状态管理器"""
    print("\n" + "=" * 60)
    print("测试初始状态管理器")
    print("=" * 60)
    
    manager = InitialStateManager()
    
    # 测试1：初始状态为空
    print("\n测试1: 初始状态为空")
    html, action_mask = manager.get_initial_state()
    assert html is None, "初始HTML应为None"
    assert action_mask is None, "action_mask应为None"
    print("  ✓ 初始状态为空")
    
    # 测试2：设置初始状态
    print("\n测试2: 设置初始状态")
    test_html = "<html><body>测试内容</body></html>"
    test_mask = {"types": [True, False], "params": [0, 1]}
    
    manager.set_initial_state(test_html, test_mask)
    assert manager.initial_html == test_html, "HTML未正确保存"
    assert manager.action_mask == test_mask, "action_mask未正确保存"
    assert manager.is_initialized, "is_initialized应为True"
    print("  ✓ 初始状态设置成功")
    
    # 测试3：获取初始状态
    print("\n测试3: 获取初始状态")
    html, action_mask = manager.get_initial_state()
    assert html == test_html, "HTML未正确获取"
    assert action_mask == test_mask, "action_mask未正确获取"
    print("  ✓ 初始状态获取成功")
    
    # 测试4：清除状态
    print("\n测试4: 清除状态")
    manager.clear()
    assert manager.initial_html is None, "HTML未清除"
    assert manager.action_mask is None, "action_mask未清除"
    assert not manager.is_initialized, "is_initialized应为False"
    print("  ✓ 状态清除成功")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过")
    print("=" * 60)


if __name__ == '__main__':
    try:
        test_initial_state_manager()
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
