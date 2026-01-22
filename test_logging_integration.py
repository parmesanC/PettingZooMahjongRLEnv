"""
日志系统集成测试

验证日志系统与环境和状态机的完整集成。
"""

import os
import sys
import json
import shutil
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from example_mahjong_env import WuhanMahjongEnv
from src.mahjong_rl.logging import (
    LogLevel,
    CompositeLogger,
    FileLogger,
    GameRecorder,
    PerfMonitor,
    LogFormatter
)
from src.mahjong_rl.core.constants import ActionType


def cleanup_test_dirs():
    """清理测试目录"""
    test_dirs = [
        Path("test_logs_integration"),
        Path("test_replays_integration"),
        Path("test_perf_integration"),
    ]
    for dir_path in test_dirs:
        if dir_path.exists():
            shutil.rmtree(dir_path)


def test_default_logging():
    """测试 1: 默认日志配置（文件日志）"""
    print("\n" + "=" * 60)
    print("测试 1: 默认日志配置")
    print("=" * 60)

    env = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,
        enable_logging=True
    )

    # 运行几步
    obs, info = env.reset(seed=42)
    print(f"  游戏 ID: {env.current_game_id}")

    for i in range(3):
        action = (ActionType.DISCARD.value, 0)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  步骤 {i+1}: reward={reward}")

    # 验证日志文件
    log_dir = Path("logs")
    if log_dir.exists():
        log_files = list(log_dir.glob("*.json"))
        print(f"  ✓ 创建了 {len(log_files)} 个日志文件")
        for f in log_files:
            print(f"    - {f.name}")
    else:
        print(f"  ! 未找到日志目录")

    print("  ✓ 默认日志测试完成\n")


def test_disabled_logging():
    """测试 2: 禁用日志"""
    print("\n" + "=" * 60)
    print("测试 2: 禁用日志")
    print("=" * 60)

    env = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,
        enable_logging=False
    )

    # 运行几步
    obs, info = env.reset(seed=42)
    print(f"  Logger: {env.logger}")
    assert env.logger is None, "禁用日志时 logger 应该是 None"

    for i in range(2):
        action = (ActionType.DISCARD.value, 0)
        obs, reward, terminated, truncated, info = env.step(action)

    print("  ✓ 禁用日志测试完成\n")


def test_custom_logger():
    """测试 3: 自定义日志器"""
    print("\n" + "=" * 60)
    print("测试 3: 自定义日志器")
    print("=" * 60)

    # 创建自定义日志器
    custom_logger = FileLogger(
        log_dir="test_logs_integration",
        level=LogLevel.DEBUG,
        enable_async=False
    )

    env = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,
        logger=custom_logger
    )

    # 运行几步
    obs, info = env.reset(seed=42)
    print(f"  使用自定义 logger: {type(env.logger).__name__}")

    for i in range(2):
        action = (ActionType.DISCARD.value, 0)
        obs, reward, terminated, truncated, info = env.step(action)

    # 验证日志文件
    log_dir = Path("test_logs_integration")
    if log_dir.exists():
        log_files = list(log_dir.glob("*.json"))
        print(f"  ✓ 创建了 {len(log_files)} 个日志文件")
        # 读取并显示日志内容
        for f in log_files:
            with open(f, 'r', encoding='utf-8') as file:
                for line in file:
                    log_entry = json.loads(line)
                    print(f"    - {log_entry['type']}: {log_entry.get('level', 'N/A')}")
    else:
        print(f"  ! 未找到日志目录")

    print("  ✓ 自定义日志器测试完成\n")


def test_game_recorder():
    """测试 4: 对局记录器"""
    print("\n" + "=" * 60)
    print("测试 4: 对局记录器")
    print("=" * 60)

    env = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,
        log_config={
            "file_logger": {"enabled": False},
            "game_recorder": {
                "enabled": True,
                "replay_dir": "test_replays_integration"
            }
        }
    )

    # 运行几步
    obs, info = env.reset(seed=42)
    print(f"  游戏 ID: {env.current_game_id}")

    for i in range(3):
        action = (ActionType.DISCARD.value, i % 10)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break

    # 验证回放文件
    replay_dir = Path("test_replays_integration")
    if replay_dir.exists():
        replay_files = list(replay_dir.glob("*.json"))
        print(f"  ✓ 创建了 {len(replay_files)} 个回放文件")
        for f in replay_files:
            print(f"    - {f.name}")
            # 读取并验证内容
            with open(f, 'r', encoding='utf-8') as file:
                game_data = json.load(file)
                print(f"    游戏配置: {game_data.get('config')}")
                print(f"    步骤数: {len(game_data.get('steps', []))}")
    else:
        print(f"  ! 未找到回放目录")

    print("  ✓ 对局记录器测试完成\n")


def test_composite_logger():
    """测试 5: 组合日志器"""
    print("\n" + "=" * 60)
    print("测试 5: 组合日志器")
    print("=" * 60)

    # 创建组合日志器
    composite = CompositeLogger([
        FileLogger(log_dir="test_logs_integration", enable_async=False),
        GameRecorder(replay_dir="test_replays_integration"),
        PerfMonitor(perf_dir="test_perf_integration", enabled=True)
    ])

    env = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,
        logger=composite
    )

    # 运行几步
    obs, info = env.reset(seed=42)
    print(f"  使用组合日志器: {type(env.logger).__name__}")

    for i in range(3):
        action = (ActionType.DISCARD.value, i % 10)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break

    # 验证各类文件
    log_dir = Path("test_logs_integration")
    replay_dir = Path("test_replays_integration")
    perf_dir = Path("test_perf_integration")

    print(f"  日志文件: {len(list(log_dir.glob('*.json'))) if log_dir.exists() else 0}")
    print(f"  回放文件: {len(list(replay_dir.glob('*.json'))) if replay_dir.exists() else 0}")
    print(f"  性能文件: {len(list(perf_dir.glob('*.jsonl'))) if perf_dir.exists() else 0}")

    print("  ✓ 组合日志器测试完成\n")


def test_state_machine_logging():
    """测试 6: 状态机日志"""
    print("\n" + "=" * 60)
    print("测试 6: 状态机日志")
    print("=" * 60)

    env = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,
        log_config={
            "file_logger": {
                "enabled": True,
                "log_dir": "test_logs_integration"
            }
        }
    )

    # 运行几步
    obs, info = env.reset(seed=42)

    for i in range(2):
        action = (ActionType.DISCARD.value, i % 10)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break

    # 检查状态机日志
    machine_logger = env.state_machine.get_logger()
    print(f"  状态机日志器类型: {type(machine_logger).__name__}")

    if hasattr(machine_logger, 'get_history'):
        history = machine_logger.get_history()
        print(f"  状态转换历史: {len(history)} 条")
        for entry in history[:5]:  # 显示前5条
            print(f"    - {entry.get('type')}: {entry.get('from_state', 'N/A')} -> {entry.get('to_state', 'N/A')}")

    print("  ✓ 状态机日志测试完成\n")


def test_game_cycle_with_logging():
    """测试 7: 完整游戏周期（带日志）"""
    print("\n" + "=" * 60)
    print("测试 7: 完整游戏周期（带日志）")
    print("=" * 60)

    env = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,
        log_config={
            "file_logger": {"enabled": False},
            "game_recorder": {"enabled": True, "replay_dir": "test_replays_integration"}
        }
    )

    # 完整游戏循环
    obs, info = env.reset(seed=42)
    game_id = env.current_game_id
    print(f"  游戏 ID: {game_id}")
    print(f"  初始 agent: {env.agent_selection}")

    step_count = 0
    max_steps = 20

    for agent in env.agent_iter(max_steps):
        obs, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            action = None
        else:
            # 简单策略：打出第一张牌
            action_mask = obs.get('action_mask', [])
            if len(action_mask) > 34:
                # 找第一个可打出的牌
                for i in range(34):
                    if action_mask[i] > 0:
                        action = (ActionType.DISCARD.value, i)
                        break
                else:
                    action = (ActionType.PASS.value, 0)
            else:
                action = (ActionType.PASS.value, 0)

        step_count += 1
        print(f"  步骤 {step_count}: {agent} -> {action}")

        env.step(action)

        if terminated or truncated:
            break

    print(f"  总步数: {step_count}")
    print(f"  游戏结束: terminated={terminated}, truncated={truncated}")

    # 验证回放文件内容
    replay_dir = Path("test_replays_integration")
    if replay_dir.exists():
        replay_files = list(replay_dir.glob(f"{game_id}.json"))
        if replay_files:
            with open(replay_files[0], 'r', encoding='utf-8') as file:
                game_data = json.load(file)
                print(f"  记录的步数: {len(game_data.get('steps', []))}")
                print(f"  游戏结果: {game_data.get('result', {})}")

    print("  ✓ 完整游戏周期测试完成\n")


def test_log_levels():
    """测试 8: 日志级别过滤"""
    print("\n" + "=" * 60)
    print("测试 8: 日志级别过滤")
    print("=" * 60)

    # 创建不同级别的日志器
    info_logger = FileLogger(
        log_dir="test_logs_integration/info",
        level=LogLevel.INFO,
        enable_async=False
    )

    debug_logger = FileLogger(
        log_dir="test_logs_integration/debug",
        level=LogLevel.DEBUG,
        enable_async=False
    )

    env1 = WuhanMahjongEnv(logger=info_logger)
    env2 = WuhanMahjongEnv(logger=debug_logger)

    # 运行几步
    env1.reset(seed=42)
    for _ in range(2):
        env1.step((ActionType.DISCARD.value, 0))

    env2.reset(seed=42)
    for _ in range(2):
        env2.step((ActionType.DISCARD.value, 0))

    # 比较日志数量
    info_dir = Path("test_logs_integration/info")
    debug_dir = Path("test_logs_integration/debug")

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

    print(f"  INFO 级别日志条目: {info_count}")
    print(f"  DEBUG 级别日志条目: {debug_count}")

    print("  ✓ 日志级别过滤测试完成\n")


def test_performance_monitoring():
    """测试 9: 性能监控"""
    print("\n" + "=" * 60)
    print("测试 9: 性能监控")
    print("=" * 60)

    env = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,
        enable_perf_monitor=True
    )

    # 运行几步
    obs, info = env.reset(seed=42)
    print(f"  启用性能监控")

    for i in range(3):
        action = (ActionType.DISCARD.value, i % 10)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break

    # 检查性能数据
    perf_dir = Path("performance")
    if perf_dir.exists():
        perf_files = list(perf_dir.glob("*.jsonl"))
        print(f"  性能文件数: {len(perf_files)}")
        if perf_files:
            with open(perf_files[0], 'r', encoding='utf-8') as file:
                for line in file:
                    perf_entry = json.loads(line)
                    if 'summary' in perf_entry:
                        print(f"  性能摘要: {perf_entry['summary']}")
                    break

    print("  ✓ 性能监控测试完成\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("日志系统集成测试套件")
    print("=" * 60)

    try:
        cleanup_test_dirs()

        test_default_logging()
        test_disabled_logging()
        test_custom_logger()
        test_game_recorder()
        test_composite_logger()
        test_state_machine_logging()
        test_game_cycle_with_logging()
        test_log_levels()
        test_performance_monitoring()

        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

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
    run_all_tests()
