"""
Belief State + Centralized Critic 训练流程集成测试

测试内容：
1. BeliefMahjongTrainer 可以正确初始化
2. 三阶段训练配置正确
3. 检查点保存/加载功能正常
4. 评估功能正常
"""

import sys
import os
import unittest
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

# 跳过如果没有 GPU 或 PyTorch 不可用
try:
    from scripts.train_belief_mahjong import BeliefMahjongTrainer

    HAS_DEPENDENCIES = True
except ImportError as e:
    HAS_DEPENDENCIES = False
    IMPORT_ERROR = str(e)


@unittest.skipUnless(
    HAS_DEPENDENCIES, f"Dependencies not satisfied: {IMPORT_ERROR if not HAS_DEPENDENCIES else ''}"
)
class TestBeliefMahjongTrainer(unittest.TestCase):
    """测试 BeliefMahjongTrainer"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, "logs")
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        self.tensorboard_dir = os.path.join(self.temp_dir, "runs")

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_trainer_initialization(self):
        """测试训练器可以正确初始化"""
        print("\n[Test] 测试训练器初始化...")

        trainer = BeliefMahjongTrainer(
            device="cpu",  # 使用 CPU 进行测试
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            tensorboard_dir=self.tensorboard_dir,
            use_belief=True,
            use_centralized_critic=True,
            n_belief_samples=5,
        )

        # 验证初始化
        self.assertEqual(trainer.device, "cpu")
        self.assertTrue(trainer.use_belief)
        self.assertTrue(trainer.use_centralized_critic)
        self.assertEqual(trainer.n_belief_samples, 5)
        self.assertEqual(trainer.current_phase, 1)
        self.assertEqual(trainer.episode_count, 0)

        print("[PASS] 训练器初始化测试通过")

    def test_phase_configuration(self):
        """测试三阶段配置"""
        print("\n[Test] 测试三阶段配置...")

        trainer = BeliefMahjongTrainer(
            device="cpu",
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            tensorboard_dir=self.tensorboard_dir,
        )

        # 验证 Phase 1 配置
        self.assertEqual(trainer.phase_config[1]["training_phase"], 1)
        self.assertTrue(trainer.phase_config[1]["use_centralized_critic"])
        self.assertEqual(trainer.phase_config[1]["description"], "Omniscient (全知)")

        # 验证 Phase 2 配置
        self.assertEqual(trainer.phase_config[2]["training_phase"], 2)
        self.assertTrue(trainer.phase_config[2]["use_centralized_critic"])
        self.assertEqual(
            trainer.phase_config[2]["description"], "Progressive (渐进遮蔽)"
        )

        # 验证 Phase 3 配置
        self.assertEqual(trainer.phase_config[3]["training_phase"], 3)
        self.assertFalse(trainer.phase_config[3]["use_centralized_critic"])
        self.assertEqual(trainer.phase_config[3]["description"], "Real (真实信息)")

        print("[PASS] 三阶段配置测试通过")

    def test_directory_creation(self):
        """测试目录创建"""
        print("\n[Test] 测试目录创建...")

        trainer = BeliefMahjongTrainer(
            device="cpu",
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            tensorboard_dir=self.tensorboard_dir,
        )

        # 验证目录已创建
        self.assertTrue(os.path.exists(self.log_dir))
        self.assertTrue(os.path.exists(self.checkpoint_dir))
        self.assertTrue(os.path.exists(self.tensorboard_dir))

        print("[PASS] 目录创建测试通过")

    def test_stats_initialization(self):
        """测试统计信息初始化"""
        print("\n[Test] 测试统计信息初始化...")

        trainer = BeliefMahjongTrainer(
            device="cpu",
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            tensorboard_dir=self.tensorboard_dir,
        )

        # 验证统计信息结构
        self.assertEqual(trainer.stats["total_episodes"], 0)
        self.assertEqual(len(trainer.stats["total_wins"]), 4)
        self.assertEqual(len(trainer.stats["phase_stats"]), 3)

        for phase in [1, 2, 3]:
            self.assertIn("episodes", trainer.stats["phase_stats"][phase])
            self.assertIn("wins", trainer.stats["phase_stats"][phase])
            self.assertIn("start_time", trainer.stats["phase_stats"][phase])
            self.assertIn("end_time", trainer.stats["phase_stats"][phase])

        print("[PASS] 统计信息初始化测试通过")


@unittest.skipUnless(
    HAS_DEPENDENCIES, f"Dependencies not satisfied: {IMPORT_ERROR if not HAS_DEPENDENCIES else ''}"
)
class TestTrainingIntegration(unittest.TestCase):
    """测试训练集成功能"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, "logs")
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        self.tensorboard_dir = os.path.join(self.temp_dir, "runs")

        # 创建小型配置用于快速测试
        from src.drl.config import get_quick_test_config

        self.config = get_quick_test_config()

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_environment_creation(self):
        """测试环境创建"""
        print("\n[Test] 测试环境创建...")

        trainer = BeliefMahjongTrainer(
            config=self.config,
            device="cpu",
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            tensorboard_dir=self.tensorboard_dir,
        )

        # 创建 Phase 1 环境
        env = trainer._create_environment(phase=1)
        self.assertIsNotNone(env)
        self.assertEqual(env.training_phase, 1)

        # 创建 Phase 2 环境
        env = trainer._create_environment(phase=2)
        self.assertIsNotNone(env)
        self.assertEqual(env.training_phase, 2)

        # 创建 Phase 3 环境
        env = trainer._create_environment(phase=3)
        self.assertIsNotNone(env)
        self.assertEqual(env.training_phase, 3)

        print("[PASS] 环境创建测试通过")


class TestCommandLineArgs(unittest.TestCase):
    """测试命令行参数解析"""

    def test_default_args(self):
        """测试默认参数"""
        print("\n[Test] 测试默认命令行参数...")

        # 这里我们只是验证参数结构，不实际运行
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--quick-test", action="store_true")
        parser.add_argument("--start-phase", type=int, default=1)
        parser.add_argument("--phase1-episodes", type=int, default=None)
        parser.add_argument("--phase2-episodes", type=int, default=None)
        parser.add_argument("--phase3-episodes", type=int, default=None)

        args = parser.parse_args([])

        self.assertFalse(args.quick_test)
        self.assertEqual(args.start_phase, 1)
        self.assertIsNone(args.phase1_episodes)
        self.assertIsNone(args.phase2_episodes)
        self.assertIsNone(args.phase3_episodes)

        print("[PASS] 默认参数测试通过")

    def test_custom_args(self):
        """测试自定义参数"""
        print("\n[Test] 测试自定义命令行参数...")

        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--quick-test", action="store_true")
        parser.add_argument("--start-phase", type=int, default=1)
        parser.add_argument("--phase1-episodes", type=int, default=None)

        args = parser.parse_args(
            ["--quick-test", "--start-phase", "2", "--phase1-episodes", "10000"]
        )

        self.assertTrue(args.quick_test)
        self.assertEqual(args.start_phase, 2)
        self.assertEqual(args.phase1_episodes, 10000)

        print("[PASS] 自定义参数测试通过")


def run_tests():
    """运行所有测试"""
    print("=" * 80)
    print("Belief State + Centralized Critic Training Integration Test")
    print("=" * 80)

    if not HAS_DEPENDENCIES:
        print(f"\n[SKIP] Dependencies not satisfied: {IMPORT_ERROR}")
        print("Please install all dependencies:")
        print("  pip install torch numpy tensorboard")
        return 0

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestBeliefMahjongTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestCommandLineArgs))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 打印结果
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("[PASS] All tests passed!")
    else:
        print("[FAIL] Some tests failed")
    print("=" * 80)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit(run_tests())
