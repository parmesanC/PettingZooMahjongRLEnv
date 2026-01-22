#!/usr/bin/env python3
"""
快速测试FastAPI服务器（简化版）
"""

import sys
sys.path.insert(0, '.')

from src.mahjong_rl.web.websocket_manager import WebSocketManager
from src.mahjong_rl.web.initial_state_manager import InitialStateManager
from src.mahjong_rl.web.fastapi_server import MahjongFastAPIServer
from example_mahjong_env import WuhanMahjongEnv
from src.mahjong_rl.manual_control.cli_controller import CLIManualController


def test_server_simple():
    """简化版服务器测试"""
    print("\n" + "=" * 60)
    print("简化版FastAPI服务器测试")
    print("=" * 60)
    
    try:
        # 创建环境
        print("\n1. 创建环境...")
        env = WuhanMahjongEnv(render_mode=None, training_phase=3, enable_logging=False)
        print("  ✓ 环境创建成功")
        
        # 创建控制器（用于初始化）
        print("\n2. 创建控制器...")
        controller = CLIManualController(env=env, max_episodes=0)
        print("  ✓ 控制器创建成功")
        
        # 创建初始状态管理器
        print("\n3. 创建初始状态管理器...")
        initial_state_manager = InitialStateManager()
        print("  ✓ 初始状态管理器创建成功")
        
        # 初始化游戏
        print("\n4. 初始化游戏（手动）...")
        print("  - 调用env.reset()...")
        env.reset()
        print("  ✓ env.reset()调用成功")
        
        print("\n5. 检查context...")
        print(f"  - context类型: {type(env.context)}")
        print(f"  - context有players属性: {hasattr(env.context, 'players')}")
        if hasattr(env.context, 'players'):
            print(f"  - context.players类型: {type(env.context.players)}")
            print(f"  - context.players长度: {len(env.context.players) if env.context.players else 'N/A'}")
        
        # 获取action_mask
        print("\n6. 获取observation和action_mask...")
        obs = env.observe(env.agent_selection)
        print(f"  ✓ observation keys: {list(obs.keys())}")
        action_mask = obs.get('action_mask', {})
        print(f"  ✓ action_mask类型: {type(action_mask)}")
        
        # 生成初始HTML
        print("\n7. 生成初始HTML...")
        from src.mahjong_rl.visualization.web_renderer import WebRenderer
        renderer = WebRenderer()
        initial_html = renderer.render(
            env.context, 
            env.agent_selection
        )
        print(f"  ✓ 初始HTML生成成功（长度：{len(initial_html)}字符）")
        
        # 设置初始状态
        print("\n8. 设置初始状态...")
        initial_state_manager.set_initial_state(initial_html, action_mask)
        print("  ✓ 初始状态设置成功")
        
        # 验证获取
        print("\n9. 验证获取初始状态...")
        html, mask = initial_state_manager.get_initial_state()
        print(f"  ✓ HTML长度: {len(html) if html else 0}字符")
        print(f"  ✓ action_mask: {mask is not None}")
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_server_simple()
