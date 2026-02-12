"""单元测试：向量化训练器"""
import pytest
import numpy as np
from src.drl.config import get_default_config
from src.drl.trainer import NFSPTrainer


def test_trainer_init_with_vectorized_env():
    """测试使用向量化环境初始化训练器"""
    config = get_default_config()
    trainer = NFSPTrainer(
        config=config,
        device="cpu",
        use_vectorized_env=True,
        num_envs=4,
    )

    # 验证向量化环境已创建
    assert trainer.use_vectorized_env is True
    assert trainer.num_envs == 4
    assert trainer.vec_env is not None
    assert trainer.env is None  # 向量化模式下单环境应为 None
    assert trainer.vec_env.num_envs == 4

    # 清理
    trainer.close()


def test_trainer_init_without_vectorized_env():
    """测试不使用向量化环境初始化训练器"""
    config = get_default_config()
    trainer = NFSPTrainer(
        config=config,
        device="cpu",
        use_vectorized_env=False,
    )

    # 验证单环境已创建
    assert trainer.use_vectorized_env is False
    assert trainer.env is not None
    assert trainer.vec_env is None  # 非向量化模式下 vec_env 应为 None

    # 清理
    trainer.close()


def test_trainer_vectorized_mode_has_correct_attributes():
    """测试向量化模式训练器具有正确的属性和方法"""
    config = get_default_config()
    trainer = NFSPTrainer(
        config=config,
        device="cpu",
        use_vectorized_env=True,
        num_envs=2,
    )

    # 验证辅助方法存在
    assert hasattr(trainer, "_init_env_state")
    assert hasattr(trainer, "_group_by_current_agent")
    assert hasattr(trainer, "_execute_single_step")
    assert hasattr(trainer, "_step_agent_batch")
    assert hasattr(trainer, "_finalize_episode")
    assert hasattr(trainer, "_run_episode_vectorized")
    assert hasattr(trainer, "close")

    # 验证这些方法是可调用的
    assert callable(trainer._init_env_state)
    assert callable(trainer._group_by_current_agent)
    assert callable(trainer._execute_single_step)
    assert callable(trainer._step_agent_batch)
    assert callable(trainer._finalize_episode)
    assert callable(trainer._run_episode_vectorized)
    assert callable(trainer.close)

    # 清理
    trainer.close()


def test_trainer_close_safely():
    """测试 close 方法安全关闭环境"""
    config = get_default_config()

    # 测试向量化模式
    trainer_vec = NFSPTrainer(
        config=config,
        device="cpu",
        use_vectorized_env=True,
        num_envs=2,
    )
    trainer_vec.close()  # 应该不抛出异常

    # 测试非向量化模式
    trainer_single = NFSPTrainer(
        config=config,
        device="cpu",
        use_vectorized_env=False,
    )
    trainer_single.close()  # 应该不抛出异常
