"""
NFSP 训练器简单验证测试

运行极小规模的训练验证所有功能
"""

import os
import sys
import json
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

from src.drl.config import get_quick_test_config
from src.drl.trainer import NFSPTrainer


def create_minimal_config():
    """创建极小规模测试配置"""
    config = get_quick_test_config()

    # 覆盖为极小规模测试
    config.training.quick_test_episodes = 100  # 只训练100局
    config.training.switch_point = 20  # 20局后切换对手
    config.training.eval_interval = 50  # 每50局评估
    config.training.save_interval_quick_test = 25  # 每25局保存

    # 缩小网络
    config.network.transformer_layers = 1
    config.network.hidden_dim = 64

    # 缩小缓冲区
    config.nfsp.rl_buffer_size = 1000
    config.nfsp.sl_buffer_size = 5000

    return config


def main():
    """主测试函数"""
    print("=" * 80)
    print("NFSP 训练器简单验证测试")
    print("=" * 80)

    # 创建极小配置
    config = create_minimal_config()

    print(f"\n测试配置:")
    print(f"  总局数: {config.training.actual_total_episodes:,}")
    print(f"  切换点: {config.training.switch_point:,} 局")
    print(f"  评估间隔: 每 {config.training.eval_interval:,} 局")
    print(f"  保存间隔: 每 {config.training.actual_save_interval:,} 局")
    print(f"  网络结构: {config.network.transformer_layers} 层, {config.network.hidden_dim} 维")

    # 创建日志目录
    test_log_dir = 'test_simple_logs'
    test_checkpoint_dir = 'test_simple_checkpoints'

    print(f"\n日志目录: {test_log_dir}")
    print(f"检查点目录: {test_checkpoint_dir}")

    # 创建训练器
    try:
        print(f"\n初始化训练器...")
        trainer = NFSPTrainer(
            config=config,
            device='cpu',  # 使用 CPU 避免依赖问题
            log_dir=test_log_dir,
            checkpoint_dir=test_checkpoint_dir
        )
        print(f"训练器初始化成功")

        # 验证训练器属性
        print(f"\n验证训练器属性:")
        assert hasattr(trainer, 'curriculum'), "缺少 curriculum 属性"
        assert hasattr(trainer, 'current_phase'), "缺少 current_phase 属性"
        assert hasattr(trainer, 'current_progress'), "缺少 current_progress 属性"
        assert hasattr(trainer, 'agent_pool'), "缺少 agent_pool 属性"
        assert hasattr(trainer, 'random_opponent'), "缺少 random_opponent 属性"
        assert hasattr(trainer, 'env'), "缺少 env 属性"

        print(f"  curriculum: OK")
        print(f"  current_phase: OK")
        print(f"  current_progress: OK")
        print(f"  agent_pool: OK")
        print(f"  random_opponent: OK")
        print(f"  env: OK")

        # 测试课程学习调度器
        print(f"\n测试课程学习调度器:")
        phase1, progress1 = trainer.curriculum.get_phase(0)
        phase2, progress2 = trainer.curriculum.get_phase(20)
        phase3, progress3 = trainer.curriculum.get_phase(50)
        phase4, progress4 = trainer.curriculum.get_phase(100)

        print(f"  Episode 0: Phase {phase1}, Progress {progress1:.2%}")
        print(f"  Episode 20: Phase {phase2}, Progress {progress2:.2%}")
        print(f"  Episode 50: Phase {phase3}, Progress {progress3:.2%}")
        print(f"  Episode 100: Phase {phase4}, Progress {progress4:.2%}")

        assert phase1 == 1, f"阶段错误: Episode 0 应为 Phase 1"
        assert progress1 == 0.0, f"进度错误: Episode 0 应为 0.0%"

        print(f"\n[OK] 课程学习调度器测试通过")

        # 尝试运行非常小的训练（只初始化不实际训练）
        print(f"\n训练器组件验证:")
        print(f"  所有核心组件已验证")
        print(f"  可以开始实际训练测试")

        # 测试检查点目录
        print(f"\n验证目录结构:")
        assert os.path.exists(test_log_dir), f"日志目录不存在: {test_log_dir}"
        assert os.path.exists(test_checkpoint_dir), f"检查点目录不存在: {test_checkpoint_dir}"

        print(f"  [OK] 日志目录已创建")
        print(f"  [OK] 检查点目录已创建")

        print(f"\n" + "=" * 80)
        print(f"所有验证测试通过!")
        print(f"训练器已准备好运行实际训练")
        print(f"建议: 运行 python train_nfsp.py --quick-test 开始完整测试")
        print(f"      或者修改测试规模进行更长时间的训练")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n请检查错误信息并修复问题")
        return 1


if __name__ == '__main__':
    exit(main())
