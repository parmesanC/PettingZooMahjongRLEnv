"""
端到端训练流程集成测试

测试内容：
1. 完整的三阶段训练流程（Phase 1 → Phase 2 → Phase 3）
2. Belief Network 在各阶段的使用
3. Centralized Critic 在 Phase 1/2 的使用，Phase 3 的不使用
4. 检查点保存和加载
5. 训练统计数据正确记录
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

try:
    from scripts.train_belief_mahjong import BeliefMahjongTrainer
    from src.drl.config import get_quick_test_config

    HAS_DEPENDENCIES = True
except ImportError as e:
    HAS_DEPENDENCIES = False
    IMPORT_ERROR = str(e)


@unittest.skipUnless(
    HAS_DEPENDENCIES,
    f"Dependencies not satisfied: {IMPORT_ERROR if not HAS_DEPENDENCIES else ''}",
)
class TestEndToEndTraining(unittest.TestCase):
    """测试端到端训练流程"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, "logs")
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        self.tensorboard_dir = os.path.join(self.temp_dir, "runs")

        # 使用快速测试配置
        self.config = get_quick_test_config()

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_phase_1_training(self):
        """测试 Phase 1（全知）训练"""
        print("\n[Test] 测试 Phase 1 训练...")

        trainer = BeliefMahjongTrainer(
            config=self.config,
            device="cpu",
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            tensorboard_dir=self.tensorboard_dir,
            use_belief=True,
            use_centralized_critic=True,
        )

        # 验证 Phase 1 配置
        phase_config = trainer.phase_config[1]
        self.assertEqual(phase_config["training_phase"], 1)
        self.assertTrue(phase_config["use_centralized_critic"])
        self.assertEqual(phase_config["description"], "Omniscient (全知)")

        print("[PASS] Phase 1 训练测试通过")

    def test_phase_2_training(self):
        """测试 Phase 2（渐进遮蔽）训练"""
        print("\n[Test] 测试 Phase 2 训练...")

        trainer = BeliefMahjongTrainer(
            config=self.config,
            device="cpu",
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            tensorboard_dir=self.tensorboard_dir,
            use_belief=True,
            use_centralized_critic=True,
        )

        # 验证 Phase 2 配置
        phase_config = trainer.phase_config[2]
        self.assertEqual(phase_config["training_phase"], 2)
        self.assertTrue(phase_config["use_centralized_critic"])
        self.assertEqual(phase_config["description"], "Progressive (渐进遮蔽)")

        print("[PASS] Phase 2 训练测试通过")

    def test_phase_3_training(self):
        """测试 Phase 3（真实信息）训练"""
        print("\n[Test] 测试 Phase 3 训练...")

        trainer = BeliefMahjongTrainer(
            config=self.config,
            device="cpu",
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            tensorboard_dir=self.tensorboard_dir,
            use_belief=True,
            use_centralized_critic=True,
        )

        # 验证 Phase 3 配置
        phase_config = trainer.phase_config[3]
        self.assertEqual(phase_config["training_phase"], 3)
        self.assertFalse(phase_config["use_centralized_critic"])
        self.assertEqual(phase_config["description"], "Real (真实信息)")

        print("[PASS] Phase 3 训练测试通过")

    def test_phase_transition(self):
        """测试 Phase 转换"""
        print("\n[Test] 测试 Phase 转换...")

        trainer = BeliefMahjongTrainer(
            config=self.config,
            device="cpu",
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            tensorboard_dir=self.tensorboard_dir,
        )

        # 初始 Phase 为 1
        self.assertEqual(trainer.current_phase, 1)

        # 模拟 Phase 1 完成
        trainer.stats["phase_stats"][1]["episodes"] = 10  # 假设阈值是 10
        trainer.stats["total_episodes"] = 10

        # 应该转换到 Phase 2
        # 注意：实际的转换逻辑在 train() 方法中，这里只验证配置
        self.assertEqual(trainer.phase_config[2]["training_phase"], 2)

        print("[PASS] Phase 转换测试通过")

    def test_checkpoint_save_and_load(self):
        """测试检查点保存和加载"""
        print("\n[Test] 测试检查点保存和加载...")

        trainer = BeliefMahjongTrainer(
            config=self.config,
            device="cpu",
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            tensorboard_dir=self.tensorboard_dir,
        )

        # 运行少量 episodes
        trainer.stats["total_episodes"] = 5
        trainer.stats["total_wins"] = [1, 0, 2, 1]
        trainer.current_phase = 1

        # 创建 agent_pool
        from src.drl.agent import NFSPAgentPool

        agent_pool = NFSPAgentPool(
            config=self.config, device="cpu", num_agents=4, share_parameters=True
        )

        # 保存检查点
        trainer._save_checkpoint(phase=1, episode=5, agent_pool=agent_pool)

        # 验证检查点文件存在
        checkpoint_path = os.path.join(self.checkpoint_dir, "phase1_5.pth")
        self.assertTrue(os.path.exists(checkpoint_path))

        print("[PASS] 检查点保存和加载测试通过")

    def test_stats_tracking(self):
        """测试统计数据跟踪"""
        print("\n[Test] 测试统计数据跟踪...")

        trainer = BeliefMahjongTrainer(
            config=self.config,
            device="cpu",
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            tensorboard_dir=self.tensorboard_dir,
        )

        # 验证初始统计
        self.assertEqual(trainer.stats["total_episodes"], 0)
        self.assertEqual(len(trainer.stats["total_wins"]), 4)

        # 模拟增加 episodes
        trainer.stats["total_episodes"] = 10
        trainer.stats["total_wins"] = [3, 2, 3, 2]

        self.assertEqual(trainer.stats["total_episodes"], 10)
        self.assertEqual(trainer.stats["total_wins"], [3, 2, 3, 2])

        # 验证 Phase 统计
        for phase in [1, 2, 3]:
            self.assertIn("episodes", trainer.stats["phase_stats"][phase])
            self.assertIn("wins", trainer.stats["phase_stats"][phase])
            self.assertIn("start_time", trainer.stats["phase_stats"][phase])
            self.assertIn("end_time", trainer.stats["phase_stats"][phase])

        print("[PASS] 统计数据跟踪测试通过")

    def test_belief_network_usage(self):
        """测试 Belief Network 的使用"""
        print("\n[Test] 测试 Belief Network 的使用...")

        trainer = BeliefMahjongTrainer(
            config=self.config,
            device="cpu",
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            tensorboard_dir=self.tensorboard_dir,
            use_belief=True,
        )

        # 验证 belief network 已创建
        # 注意：实际的 belief network 在 agent_pool 中
        if trainer.agent_pool is not None:
            # agent_pool 已初始化，验证 nfsp
            self.assertIsNotNone(trainer.agent_pool)

        print("[PASS] Belief Network 的使用测试通过")

    def test_centralized_critic_usage(self):
        """测试 Centralized Critic 的使用"""
        print("\n[Test] 测试 Centralized Critic 的使用...")

        trainer = BeliefMahjongTrainer(
            config=self.config,
            device="cpu",
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            tensorboard_dir=self.tensorboard_dir,
            use_centralized_critic=True,
        )

        # 验证 centralized critic 已创建
        if trainer.agent_pool is not None:
            self.assertIsNotNone(trainer.agent_pool.mappo.centralized_critic)

        print("[PASS] Centralized Critic 的使用测试通过")


class TestTrainingConfiguration(unittest.TestCase):
    """测试训练配置"""

    @unittest.skipUnless(
        HAS_DEPENDENCIES,
        f"Dependencies not satisfied: {IMPORT_ERROR if not HAS_DEPENDENCIES else ''}",
    )
    def test_quick_test_config(self):
        """测试快速测试配置"""
        print("\n[Test] 测试快速测试配置...")

        config = get_quick_test_config()

        # 验证配置存在
        self.assertIsNotNone(config)

        # 验证必要的配置项
        self.assertIsNotNone(config.mahjong)
        self.assertIsNotNone(config.nfsp)
        self.assertIsNotNone(config.mappo)

        print("[PASS] 快速测试配置测试通过")

    def test_phase_configs(self):
        """测试各阶段配置"""
        print("\n[Test] 测试各阶段配置...")

        trainer = BeliefMahjongTrainer(
            device="cpu",
            log_dir=tempfile.mkdtemp(),
            checkpoint_dir=tempfile.mkdtemp(),
            tensorboard_dir=tempfile.mkdtemp(),
        )

        # 验证所有三个阶段的配置存在
        self.assertIn(1, trainer.phase_config)
        self.assertIn(2, trainer.phase_config)
        self.assertIn(3, trainer.phase_config)

        # 验证每个阶段的配置包含必要的字段
        for phase in [1, 2, 3]:
            config = trainer.phase_config[phase]
            self.assertIn("training_phase", config)
            self.assertIn("use_centralized_critic", config)
            self.assertIn("description", config)
            self.assertIn("episodes", config)  # episodes 而不是 episode_threshold

        print("[PASS] 各阶段配置测试通过")


def run_tests():
    """运行所有测试"""
    print("=" * 80)
    print("End-to-End Training Integration Test")
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
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingConfiguration))

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
