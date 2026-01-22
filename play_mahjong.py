#!/usr/bin/env python3
"""
麻将游戏主入口
支持PettingZoo标准的命令行和网页两种可视化方式
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='武汉麻将PettingZoo手动控制')
    parser.add_argument('--mode', choices=['human_vs_ai', 'four_human', 'observation'],
                       default='human_vs_ai', help='游戏模式')
    parser.add_argument('--renderer', choices=['cli', 'web'],
                       default='cli', help='可视化方式: cli(命令行) 或 web(网页)')
    parser.add_argument('--episodes', type=int, default=1, help='回合数')
    parser.add_argument('--human-player', type=int, default=0, choices=[0,1,2,3],
                       help='人类玩家位置')
    parser.add_argument('--port', type=int, default=8000, help='网页服务器端口')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    
    args = parser.parse_args()
    
    try:
        from example_mahjong_env import WuhanMahjongEnv
        from src.mahjong_rl.manual_control.cli_controller import CLIManualController
        from src.mahjong_rl.manual_control.web_controller import WebManualController
        from src.mahjong_rl.agents.human.manual_strategy import ManualPlayerStrategy
        from src.mahjong_rl.agents.ai.random_strategy import RandomStrategy
    except ImportError as e:
        print(f"错误: 无法导入必要模块: {e}")
        print("请确保:")
        print("1. 位于项目根目录")
        print("2. example_mahjong_env.py 文件存在")
        sys.exit(1)
    
    env = WuhanMahjongEnv(render_mode=None, training_phase=3, enable_logging=False)
    
    if args.mode == 'human_vs_ai':
        strategies = [
            ManualPlayerStrategy(None) if i == args.human_player
            else RandomStrategy()
            for i in range(4)
        ]
    elif args.mode == 'four_human':
        strategies = [ManualPlayerStrategy(None) for _ in range(4)]
    else:
        strategies = [RandomStrategy() for _ in range(4)]
    
    if args.renderer == 'web':
        controller = WebManualController(env=env, max_episodes=args.episodes, port=args.port, strategies=strategies)
    else:
        controller = CLIManualController(env=env, max_episodes=args.episodes, strategies=strategies)
    
    for strategy in strategies:
        if isinstance(strategy, ManualPlayerStrategy):
            strategy.controller = controller
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n游戏中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
        except:
            pass
        print("\n感谢游玩武汉麻将！")


if __name__ == '__main__':
    main()
