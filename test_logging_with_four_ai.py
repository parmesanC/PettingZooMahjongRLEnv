#!/usr/bin/env python3
"""
日志系统集成测试

基于 test_four_ai.py 的测试模式，验证日志系统是否正常工作。
"""

import sys
import json
import shutil
from pathlib import Path

sys.path.insert(0, '.')

from example_mahjong_env import WuhanMahjongEnv
from src.mahjong_rl.agents.ai.random_strategy import RandomStrategy
from src.mahjong_rl.logging import (
    LogLevel,
    CompositeLogger,
    FileLogger,
    GameRecorder,
    PerfMonitor,
    LogFormatter
)


def cleanup_test_dirs():
    """清理测试目录"""
    test_dirs = [
        Path("test_logs_full"),
        Path("test_replays_full"),
        Path("test_perf_full"),
    ]
    for dir_path in test_dirs:
        if dir_path.exists():
            shutil.rmtree(dir_path)


def verify_log_files(log_dir: Path, expected_files: int = 1):
    """验证日志文件"""
    if not log_dir.exists():
        print(f"    ✗ 日志目录不存在: {log_dir}")
        return False

    log_files = list(log_dir.glob("*.json"))
    print(f"    日志文件数: {len(log_files)} (期望: {expected_files})")

    if len(log_files) < expected_files:
        print(f"    ✗ 日志文件数量不足")
        return False

    # 检查文件内容
    for f in log_files:
        with open(f, 'r', encoding='utf-8') as file:
            count = sum(1 for _ in file)
        print(f"    - {f.name}: {count} 条日志")

    return True


def verify_replay_files(replay_dir: Path, game_id: str):
    """验证回放文件"""
    if not replay_dir.exists():
        print(f"    ✗ 回放目录不存在: {replay_dir}")
        return False

    replay_file = replay_dir / f"{game_id}.json"
    if not replay_file.exists():
        print(f"    ✗ 回放文件不存在: {replay_file}")
        return False

    with open(replay_file, 'r', encoding='utf-8') as file:
        game_data = json.load(file)

    print(f"    ✓ 回放文件存在: {replay_file.name}")
    print(f"    - 游戏配置: {game_data.get('config', {})}")
    print(f"    - 步骤数: {len(game_data.get('steps', []))}")
    print(f"    - 游戏结果: {game_data.get('result', {})}")

    return True


def verify_perf_files(perf_dir: Path):
    """验证性能文件"""
    if not perf_dir.exists():
        print(f"    ✗ 性能目录不存在: {perf_dir}")
        return False

    perf_files = list(perf_dir.glob("*.jsonl"))
    print(f"    性能文件数: {len(perf_files)}")

    if len(perf_files) == 0:
        print(f"    ✗ 没有性能文件")
        return False

    # 读取最新的性能文件
    with open(perf_files[-1], 'r', encoding='utf-8') as file:
        for line in file:
            perf_entry = json.loads(line)
            if 'summary' in perf_entry:
                print(f"    ✓ 性能摘要: {perf_entry['summary']}")
            break

    return True


def run_game_with_logging(env_name: str, env, strategies, max_steps=1000):
    """运行一局游戏"""
    print(f"\n{'=' * 60}")
    print(f"测试: {env_name}")
    print('=' * 60)

    # 重置环境
    obs, info = env.reset(seed=42)
    game_id = env.current_game_id
    print(f"游戏 ID: {game_id}")
    print(f"初始 agent: {env.agent_selection}")

    step_count = 0

    # 游戏主循环
    for agent in env.agent_iter(num_steps=max_steps):
        step_count += 1

        obs, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            action = None
        else:
            # AI选择动作
            action_mask = obs['action_mask']
            agent_idx = env.agents_name_mapping[agent]
            strategy = strategies[agent_idx]
            action = strategy.choose_action(obs, action_mask)

        # 执行动作
        env.step(action)

        # 检查游戏是否结束
        if terminated or truncated:
            break

    # 打印结果
    print(f"\n游戏结束:")
    print(f"  总步数: {step_count}")
    print(f"  剩余牌墙: {len(env.context.wall)}张")

    if env.context.is_win:
        print(f"  获胜者: {list(env.context.winner_ids)}")
        print(f"  胜利方式: {env.context.win_way}")
    else:
        print(f"  结果: 荒牌流局")

    # 关闭环境（触发日志记录）
    env.close()

    return game_id, step_count


def test_file_logger():
    """测试 1: 文件日志器"""
    print("\n" + "=" * 60)
    print("测试 1: 文件日志器")
    print("=" * 60)

    env = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,
        log_config={
            "file_logger": {
                "enabled": True,
                "log_dir": "test_logs_full",
                "level": "INFO"
            },
            "game_recorder": {"enabled": False},
            "perf_monitor": {"enabled": False}
        }
    )

    strategies = [RandomStrategy() for _ in range(4)]
    game_id, step_count = run_game_with_logging("文件日志器", env, strategies)

    # 验证日志文件
    print(f"\n验证日志文件:")
    if verify_log_files(Path("test_logs_full"), expected_files=1):
        print("  ✓ 文件日志器测试通过")
    else:
        print("  ✗ 文件日志器测试失败")


def test_game_recorder():
    """测试 2: 对局记录器"""
    print("\n" + "=" * 60)
    print("测试 2: 对局记录器")
    print("=" * 60)

    env = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,
        log_config={
            "file_logger": {"enabled": False},
            "game_recorder": {
                "enabled": True,
                "replay_dir": "test_replays_full"
            },
            "perf_monitor": {"enabled": False}
        }
    )

    strategies = [RandomStrategy() for _ in range(4)]
    game_id, step_count = run_game_with_logging("对局记录器", env, strategies)

    # 验证回放文件
    print(f"\n验证回放文件:")
    if verify_replay_files(Path("test_replays_full"), game_id):
        print("  ✓ 对局记录器测试通过")
    else:
        print("  ✗ 对局记录器测试失败")


def test_perf_monitor():
    """测试 3: 性能监控器"""
    print("\n" + "=" * 60)
    print("测试 3: 性能监控器")
    print("=" * 60)

    env = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,
        enable_perf_monitor=True,
        log_config={
            "file_logger": {"enabled": False},
            "game_recorder": {"enabled": False},
            "perf_monitor": {
                "enabled": True,
                "perf_dir": "test_perf_full"
            }
        }
    )

    strategies = [RandomStrategy() for _ in range(4)]
    game_id, step_count = run_game_with_logging("性能监控器", env, strategies)

    # 验证性能文件
    print(f"\n验证性能文件:")
    if verify_perf_files(Path("test_perf_full")):
        print("  ✓ 性能监控器测试通过")
    else:
        print("  ✗ 性能监控器测试失败")


def test_composite_logger():
    """测试 4: 组合日志器"""
    print("\n" + "=" * 60)
    print("测试 4: 组合日志器（全部启用）")
    print("=" * 60)

    env = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,
        log_config={
            "file_logger": {
                "enabled": True,
                "log_dir": "test_logs_full",
                "level": "INFO"
            },
            "game_recorder": {
                "enabled": True,
                "replay_dir": "test_replays_full"
            },
            "perf_monitor": {
                "enabled": True,
                "perf_dir": "test_perf_full"
            }
        }
    )

    strategies = [RandomStrategy() for _ in range(4)]
    game_id, step_count = run_game_with_logging("组合日志器", env, strategies)

    # 验证所有文件
    print(f"\n验证所有文件:")
    log_ok = verify_log_files(Path("test_logs_full"))
    replay_ok = verify_replay_files(Path("test_replays_full"), game_id)
    perf_ok = verify_perf_files(Path("test_perf_full"))

    if log_ok and replay_ok and perf_ok:
        print("  ✓ 组合日志器测试通过")
    else:
        print("  ✗ 组合日志器测试失败")


def test_custom_logger():
    """测试 5: 自定义日志器"""
    print("\n" + "=" * 60)
    print("测试 5: 自定义日志器")
    print("=" * 60)

    # 创建自定义组合日志器
    custom_logger = CompositeLogger([
        FileLogger(log_dir="test_logs_full", level=LogLevel.INFO, enable_async=False),
        GameRecorder(replay_dir="test_replays_full"),
    ])

    env = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,
        logger=custom_logger
    )

    strategies = [RandomStrategy() for _ in range(4)]
    game_id, step_count = run_game_with_logging("自定义日志器", env, strategies)

    # 验证文件
    print(f"\n验证文件:")
    log_ok = verify_log_files(Path("test_logs_full"))
    replay_ok = verify_replay_files(Path("test_replays_full"), game_id)

    if log_ok and replay_ok:
        print("  ✓ 自定义日志器测试通过")
    else:
        print("  ✗ 自定义日志器测试失败")


def test_multiple_games():
    """测试 6: 多局游戏"""
    print("\n" + "=" * 60)
    print("测试 6: 多局游戏日志")
    print("=" * 60)

    num_games = 3

    for game_num in range(num_games):
        print(f"\n第 {game_num + 1} 局:")

        env = WuhanMahjongEnv(
            render_mode=None,
            training_phase=3,
            log_config={
                "file_logger": {"enabled": False},
                "game_recorder": {
                    "enabled": True,
                    "replay_dir": "test_replays_full"
                },
                "perf_monitor": {"enabled": False}
            }
        )

        strategies = [RandomStrategy() for _ in range(4)]
        game_id, step_count = run_game_with_logging(f"第{game_num + 1}局", env, strategies)

    # 验证回放文件
    print(f"\n验证回放文件:")
    replay_dir = Path("test_replays_full")
    if replay_dir.exists():
        replay_files = list(replay_dir.glob("*.json"))
        print(f"  回放文件数: {len(replay_files)} (期望: {num_games})")
        if len(replay_files) == num_games:
            print("  ✓ 多局游戏测试通过")
        else:
            print("  ✗ 多局游戏测试失败")
    else:
        print("  ✗ 回放目录不存在")


def test_log_levels():
    """测试 7: 日志级别"""
    print("\n" + "=" * 60)
    print("测试 7: 日志级别过滤")
    print("=" * 60)

    # INFO 级别
    env_info = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,
        log_config={
            "file_logger": {
                "enabled": True,
                "log_dir": "test_logs_full/info",
                "level": "INFO"
            },
            "game_recorder": {"enabled": False},
            "perf_monitor": {"enabled": False}
        }
    )

    strategies = [RandomStrategy() for _ in range(4)]
    run_game_with_logging("INFO级别", env_info, strategies)

    # DEBUG 级别
    env_debug = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,
        log_config={
            "file_logger": {
                "enabled": True,
                "log_dir": "test_logs_full/debug",
                "level": "DEBUG"
            },
            "game_recorder": {"enabled": False},
            "perf_monitor": {"enabled": False}
        }
    )

    run_game_with_logging("DEBUG级别", env_debug, strategies)

    # 比较
    print(f"\n比较日志条目:")
    info_dir = Path("test_logs_full/info")
    debug_dir = Path("test_logs_full/debug")

    info_count = 0
    debug_count = 0

    if info_dir.exists():
        for f in info_dir.glob("*.json"):
            with open(f, 'r', encoding='utf-8') as file:
                info_count += sum(1 for _ in file)

    if debug_dir.exists():
        for f in debug_dir.glob("*.json"):
            with open(f, 'r', encoding='utf-8') as file:
                debug_count += sum(1 for _ in file)

    print(f"  INFO级别: {info_count} 条")
    print(f"  DEBUG级别: {debug_count} 条")

    if info_count > 0 and debug_count >= info_count:
        print("  ✓ 日志级别测试通过")
    else:
        print("  ✗ 日志级别测试失败")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("日志系统集成测试套件（基于 test_four_ai.py）")
    print("=" * 60)

    try:
        cleanup_test_dirs()

        test_file_logger()
        test_game_recorder()
        test_perf_monitor()
        test_composite_logger()
        test_custom_logger()
        test_multiple_games()
        test_log_levels()

        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # 清理测试文件
        print("清理测试文件...")
        cleanup_test_dirs()
        print("  ✓ 清理完成\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='日志系统集成测试')
    parser.add_argument('--test', type=str, choices=[
        'file', 'recorder', 'perf', 'composite', 'custom', 'multi', 'levels', 'all'
    ], default='all', help='要运行的测试')

    args = parser.parse_args()

    if args.test == 'all':
        run_all_tests()
    elif args.test == 'file':
        cleanup_test_dirs()
        test_file_logger()
        cleanup_test_dirs()
    elif args.test == 'recorder':
        cleanup_test_dirs()
        test_game_recorder()
        cleanup_test_dirs()
    elif args.test == 'perf':
        cleanup_test_dirs()
        test_perf_monitor()
        cleanup_test_dirs()
    elif args.test == 'composite':
        cleanup_test_dirs()
        test_composite_logger()
        cleanup_test_dirs()
    elif args.test == 'custom':
        cleanup_test_dirs()
        test_custom_logger()
        cleanup_test_dirs()
    elif args.test == 'multi':
        cleanup_test_dirs()
        test_multiple_games()
        cleanup_test_dirs()
    elif args.test == 'levels':
        cleanup_test_dirs()
        test_log_levels()
        cleanup_test_dirs()

    print("\n测试完成！")
