#!/usr/bin/env python3
"""
快速测试FastAPI服务器
只启动服务器，不运行游戏
"""

import sys
sys.path.insert(0, '.')

from src.mahjong_rl.web.fastapi_server import MahjongFastAPIServer


def test_server():
    """测试服务器启动"""
    print("\n" + "=" * 60)
    print("FastAPI服务器快速测试")
    print("=" * 60)
    
    try:
        from example_mahjong_env import WuhanMahjongEnv
        from src.mahjong_rl.manual_control.cli_controller import CLIManualController
        
        # 创建环境（不运行游戏）
        env = WuhanMahjongEnv(render_mode=None, training_phase=3, enable_logging=False)
        
        # 创建控制器（用于测试）
        controller = CLIManualController(env=env, max_episodes=0)
        
        # 创建FastAPI服务器
        server = MahjongFastAPIServer(
            env=env,
            controller=controller,
            port=8000
        )
        
        # 启动服务器（阻塞）
        server.start()
        
    except ImportError as e:
        print(f"错误: 缺少依赖 - {e}")
        print("\n请运行以下命令安装依赖：")
        print("  pip install fastapi uvicorn")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    test_server()
