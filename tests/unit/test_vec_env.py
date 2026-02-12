"""单元测试：向量化环境"""
import pytest
import numpy as np
from src.drl.vec_env import EnvFactory, VecEnv, make_vec_env


def test_env_factory_creates_environment():
    """测试 EnvFactory 能创建 WuhanMahjongEnv"""
    factory = EnvFactory(
        env_type="WuhanMahjongEnv",
        render_mode=None,
        training_phase=1,
        enable_logging=False,
    )

    env = factory.create()

    assert env is not None
    assert hasattr(env, "reset")
    assert hasattr(env, "step")


def test_vec_env_creates_multiple_environments():
    """测试 VecEnv 创建多个环境"""
    factory = EnvFactory(
        env_type="WuhanMahjongEnv",
        render_mode=None,
        training_phase=1,
        enable_logging=False,
    )

    vec_env = VecEnv([factory.create() for _ in range(4)])

    assert vec_env.num_envs == 4
    assert len(vec_env.envs) == 4


def test_vec_env_reset_all_environments():
    """测试 reset() 返回所有环境的观测"""
    factory = EnvFactory(
        env_type="WuhanMahjongEnv",
        render_mode=None,
        training_phase=1,
        enable_logging=False,
    )

    vec_env = VecEnv([factory.create() for _ in range(2)])

    observations = vec_env.reset()

    assert len(observations) == 2
    assert isinstance(observations[0], dict)
    assert "action_mask" in observations[0]


def test_vec_env_step_single_environment():
    """测试单环境步进"""
    factory = EnvFactory(
        env_type="WuhanMahjongEnv",
        render_mode=None,
        training_phase=1,
        enable_logging=False,
    )

    vec_env = VecEnv([factory.create()])
    vec_env.reset()

    # 获取有效动作
    obs, reward, terminated, truncated, info = vec_env.envs[0].last()
    action_mask = obs["action_mask"]
    valid_actions = np.where(action_mask > 0)[0]

    if len(valid_actions) > 0:
        action_idx = valid_actions[0]
        action_type = 0  # DISCARD
        action_param = action_idx
    else:
        action_type, action_param = 10, -1  # PASS

    obs, reward, terminated, truncated, info = vec_env.step(0, (action_type, action_param))

    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)


def test_make_vec_env_convenience_function():
    """测试 make_vec_env 便捷函数"""
    factory = EnvFactory(
        env_type="WuhanMahjongEnv",
        render_mode=None,
        training_phase=1,
        enable_logging=False,
    )

    vec_env = make_vec_env(factory, num_envs=2, use_subprocess=False)

    assert vec_env.num_envs == 2
    observations = vec_env.reset()
    assert len(observations) == 2


def test_vec_env_close():
    """测试 close() 方法"""
    factory = EnvFactory(
        env_type="WuhanMahjongEnv",
        render_mode=None,
        training_phase=1,
        enable_logging=False,
    )

    vec_env = VecEnv([factory.create() for _ in range(2)])
    vec_env.close()

    assert vec_env.envs is not None
