"""
NFSP 训练器小规模测试

验证所有核心功能是否正常工作
"""

import os
import sys
import time

from src.drl.config import get_quick_test_config
from src.drl.trainer import NFSPTrainer
from src.drl.curriculum import CurriculumScheduler


def test_configurations():
    """测试配置系统"""
    print("=" * 80)
    print("测试 1: 配置系统")
    print("=" * 80)

    # 快速测试配置
    from src.drl.config import get_quick_test_config
    qt_config = get_quick_test_config()
    print(f"\n快速测试配置:")
    print(f"  模式: {qt_config.training.mode}")
    print(f"  总局数: {qt_config.training.actual_total_episodes:,}")
    print(f"  保存间隔: {qt_config.training.actual_save_interval:,}")
    print(f"  评估间隔: {qt_config.training.eval_interval:,}")

    assert qt_config.training.mode == 'quick_test', "快速测试模式错误"
    assert qt_config.training.actual_total_episodes == 10000, "快速测试总局数错误"
    assert qt_config.training.actual_save_interval == 100, "快速测试保存间隔错误"

    print("\n[OK] 快速测试配置正确")

    # 课程学习调度器
    print(f"\n课程学习调度器测试:")
    qt_curriculum = CurriculumScheduler(total_episodes=qt_config.training.actual_total_episodes)

    # 测试各个阶段
    test_episodes = [0, 3333, 6666, 10000]
    expected_phases = [1, 2, 3, 3]

    for ep, expected_phase in zip(test_episodes, expected_phases):
        phase, progress = qt_curriculum.get_phase(ep)
        print(f"  Episode {ep:,}: Phase {phase}, Progress {progress:.2%}")
        assert phase == expected_phase, f"阶段错误：期望 {expected_phase}，实际 {phase}"

    print("\n[OK] 课程学习调度器正确")

    print("\n" + "=" * 80)
    print("测试 2: 训练器初始化")
    print("=" * 80)

    try:
        trainer = NFSPTrainer(
            config=qt_config,
            device='cpu',  # 使用 CPU 进行测试
            log_dir='test_logs',
            checkpoint_dir='test_checkpoints'
        )
        print("\n[OK] 训练器初始化成功")

        # 验证属性
        assert hasattr(trainer, 'curriculum'), "缺少 curriculum 属性"
        assert hasattr(trainer, 'current_phase'), "缺少 current_phase 属性"
        assert hasattr(trainer, 'current_progress'), "缺少 current_progress 属性"
        assert hasattr(trainer, 'agent_pool'), "缺少 agent_pool 属性"
        assert hasattr(trainer, 'random_opponent'), "缺少 random_opponent 属性"

        print("[OK] 训练器属性完整")

    except Exception as e:
        print(f"\n[ERROR] 训练器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("测试 3: 检查点保存功能")
    print("=" * 80)

    # 测试检查点目录创建
    if not os.path.exists('test_checkpoints'):
        print("[ERROR] 检查点目录未创建")
        return False
    print("[OK] 检查点目录已创建")

    # 测试日志目录创建
    if not os.path.exists('test_logs'):
        print("[ERROR] 日志目录未创建")
        return False
    print("[OK] 日志目录已创建")

    print("\n" + "=" * 80)
    print("测试 4: 导入测试")
    print("=" * 80)

    try:
        from src.drl import (
            CurriculumScheduler,
            NFSPTrainer,
            train_nfsp,
            get_default_config,
            get_quick_test_config
        )
        print("[OK] 所有主要类成功导入")
    except ImportError as e:
        print(f"[ERROR] 导入失败: {e}")
        return False

    return True


def cleanup():
    """清理测试文件"""
    import shutil
    print("\n" + "=" * 80)
    print("清理测试文件...")
    print("=" * 80)

    # 清理检查点目录
    if os.path.exists('test_checkpoints'):
        shutil.rmtree('test_checkpoints')
        print("  删除 test_checkpoints/")

    # 清理日志目录
    if os.path.exists('test_logs'):
        shutil.rmtree('test_logs')
        print("  删除 test_logs/")

    # 清理缓存目录
    if os.path.exists('__pycache__'):
        shutil.rmtree('__pycache__')
        print("  删除 __pycache__/")

    print("[OK] 清理完成")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("NFSP 训练器小规模测试")
    print("=" * 80)

    success = test_configurations()

    if success:
        print("\n" + "=" * 80)
        print("所有测试通过!")
        print("=" * 80)

    # 清理
    try:
        cleanup()
    except:
        pass

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

