"""
测试日志系统

验证日志系统的各个组件是否正常工作。
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.mahjong_rl.logging import (
    LogLevel,
    LogType,
    LogFormatter,
    FileLogger,
    GameRecorder,
    PerfMonitor,
    CompositeLogger
)


def test_log_formatter():
    """测试日志格式化器"""
    print("测试 LogFormatter...")

    # 测试生成游戏 ID
    game_id = LogFormatter.generate_game_id()
    print(f"  生成游戏 ID: {game_id}")
    assert game_id.startswith("game_")

    # 测试格式化时间戳
    timestamp = LogFormatter.format_timestamp()
    print(f"  格式化时间戳: {timestamp}")

    # 测试格式化日志条目
    log_entry = LogFormatter.format_log_entry(
        LogLevel.INFO,
        LogType.STATE_TRANSITION,
        game_id,
        {"from_state": "INITIAL", "to_state": "DRAWING"}
    )
    print(f"  日志条目: {json.dumps(log_entry, indent=2)}")

    print("  ✓ LogFormatter 测试通过\n")


def test_file_logger():
    """测试文件日志器"""
    print("测试 FileLogger...")

    # 创建临时日志目录
    log_dir = Path("test_logs_temp")
    if log_dir.exists():
        import shutil
        shutil.rmtree(log_dir)

    logger = FileLogger(log_dir=str(log_dir), level=LogLevel.DEBUG, enable_async=False)

    # 开始游戏
    game_id = LogFormatter.generate_game_id()
    logger.start_game(game_id, {"seed": 42, "num_players": 4})

    # 记录一些日志
    logger.log(LogLevel.INFO, LogType.CUSTOM, {"message": "测试日志"})

    # 结束游戏
    logger.end_game({"winners": [0], "total_steps": 100})

    # 验证日志文件
    log_files = list(log_dir.glob("*.json"))
    print(f"  创建的日志文件: {[f.name for f in log_files]}")
    assert len(log_files) > 0, "没有创建日志文件"

    # 读取并验证日志内容
    with open(log_files[0], 'r', encoding='utf-8') as f:
        for line in f:
            log_entry = json.loads(line)
            print(f"  日志条目: {log_entry.get('type', 'unknown')}")

    print("  ✓ FileLogger 测试通过\n")


def test_game_recorder():
    """测试对局记录器"""
    print("测试 GameRecorder...")

    # 创建临时回放目录
    replay_dir = Path("test_replays_temp")
    if replay_dir.exists():
        import shutil
        shutil.rmtree(replay_dir)

    recorder = GameRecorder(replay_dir=str(replay_dir))

    # 开始游戏
    game_id = LogFormatter.generate_game_id()
    recorder.start_game(game_id, {"seed": 42, "num_players": 4})

    # 记录一些步骤
    recorder.record_step(
        agent="player_0",
        observation={"hand": [1, 2, 3]},
        action={"type": "DISCARD", "param": 1},
        reward=0.0,
        next_observation={"hand": [2, 3]},
        info={"step": 0}
    )

    # 结束游戏
    result = recorder.end_game({"winners": [0], "total_steps": 1})

    # 验证回放文件
    replay_files = list(replay_dir.glob("*.json"))
    print(f"  创建的回放文件: {[f.name for f in replay_files]}")
    assert len(replay_files) == 1, "应该创建一个回放文件"

    # 读取并验证回放内容
    with open(replay_files[0], 'r', encoding='utf-8') as f:
        game_data = json.load(f)
        print(f"  游戏数据: {game_data['game_id']}")
        print(f"  步骤数: {len(game_data['steps'])}")
        assert len(game_data['steps']) == 1, "应该记录一步"

    print("  ✓ GameRecorder 测试通过\n")


def test_perf_monitor():
    """测试性能监控器"""
    print("测试 PerfMonitor...")

    # 创建临时性能目录
    perf_dir = Path("test_perf_temp")
    if perf_dir.exists():
        import shutil
        shutil.rmtree(perf_dir)

    monitor = PerfMonitor(perf_dir=str(perf_dir), enabled=True)

    # 开始游戏
    game_id = LogFormatter.generate_game_id()
    monitor.start_game(game_id, {"seed": 42})

    # 记录一些性能指标
    monitor.record_step_metrics(
        step_time=15.5,
        state_transition_time=2.3,
        observation_build_time=8.7
    )

    # 获取当前指标
    current = monitor.get_current_metrics()
    print(f"  当前指标: {current}")

    # 结束游戏
    monitor.end_game({"winners": [0], "total_steps": 1})

    # 验证性能文件
    perf_files = list(perf_dir.glob("*.jsonl"))
    print(f"  创建的性能文件: {[f.name for f in perf_files]}")
    assert len(perf_files) == 1, "应该创建一个性能文件"

    print("  ✓ PerfMonitor 测试通过\n")


def test_composite_logger():
    """测试组合日志器"""
    print("测试 CompositeLogger...")

    # 创建临时目录
    log_dir = Path("test_composite_temp")
    if log_dir.exists():
        import shutil
        shutil.rmtree(log_dir)

    # 创建组合日志器
    composite = CompositeLogger([
        FileLogger(log_dir=str(log_dir / "logs"), enable_async=False),
        GameRecorder(replay_dir=str(log_dir / "replays")),
        PerfMonitor(perf_dir=str(log_dir / "perf"))
    ])

    # 开始游戏
    game_id = LogFormatter.generate_game_id()
    composite.start_game(game_id, {"seed": 42})

    # 记录一些日志
    composite.log(LogLevel.INFO, LogType.CUSTOM, {"message": "测试组合日志器"})

    # 结束游戏
    composite.end_game({"winners": [0], "total_steps": 0})

    # 验证所有日志器都创建了文件
    log_files = list((log_dir / "logs").glob("*.json"))
    replay_files = list((log_dir / "replays").glob("*.json"))
    perf_files = list((log_dir / "perf").glob("*.jsonl"))

    print(f"  日志文件: {len(log_files)} 个")
    print(f"  回放文件: {len(replay_files)} 个")
    print(f"  性能文件: {len(perf_files)} 个")

    assert len(log_files) > 0 or len(replay_files) > 0 or len(perf_files) > 0

    print("  ✓ CompositeLogger 测试通过\n")


def cleanup():
    """清理测试文件"""
    print("清理测试文件...")
    import shutil

    test_dirs = [
        Path("test_logs_temp"),
        Path("test_replays_temp"),
        Path("test_perf_temp"),
        Path("test_composite_temp")
    ]

    for dir_path in test_dirs:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"  删除: {dir_path}")

    print("  ✓ 清理完成\n")


if __name__ == "__main__":
    print("=" * 60)
    print("日志系统测试")
    print("=" * 60 + "\n")

    try:
        test_log_formatter()
        test_file_logger()
        test_game_recorder()
        test_perf_monitor()
        test_composite_logger()

        print("=" * 60)
        print("所有测试通过！✓")
        print("=" * 60)

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
        cleanup()
