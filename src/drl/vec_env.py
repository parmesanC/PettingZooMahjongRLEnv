"""
向量化环境包装器

管理多个并行的 WuhanMahjongEnv 实例用于向量化训练。
每个环境独立运行，不共享状态。
"""
from typing import List, Dict, Optional


class EnvFactory:
    """环境工厂类

    用于创建标准化的 WuhanMahjongEnv 实例。
    """

    def __init__(
        self,
        env_type: str = "WuhanMahjongEnv",
        render_mode: Optional[str] = None,
        training_phase: int = 3,
        enable_logging: bool = False,
    ):
        """
        初始化环境工厂

        Args:
            env_type: 环境类型（目前只支持 WuhanMahjongEnv）
            render_mode: 渲染模式
            training_phase: 训练阶段（影响观测可见度）
            enable_logging: 是否启用日志
        """
        self.env_type = env_type
        self.render_mode = render_mode
        self.training_phase = training_phase
        self.enable_logging = enable_logging

    def create(self):
        """创建新环境实例"""
        from example_mahjong_env import WuhanMahjongEnv

        return WuhanMahjongEnv(
            render_mode=self.render_mode,
            training_phase=self.training_phase,
            enable_logging=self.enable_logging,
        )


class VecEnv:
    """
    向量化环境包装器

    管理多个并行 WuhanMahjongEnv 实例。
    每个环境独立运行，agent_iter 和状态完全隔离。
    """

    def __init__(self, envs: List):
        """
        初始化向量化环境

        Args:
            envs: WuhanMahjongEnv 实例列表
        """
        self.envs = envs
        self.num_envs = len(envs)

    def reset(self) -> List[Dict]:
        """
        重置所有环境

        Returns:
            每个环境的观测字典列表
        """
        observations = []
        for env in self.envs:
            obs, _ = env.reset()
            observations.append(obs)
        return observations

    def step(self, env_idx: int, action: tuple) -> tuple:
        """
        在指定环境中执行一步

        Args:
            env_idx: 环境索引 (0 到 num_envs-1)
            action: (action_type, action_param) 元组

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        env = self.envs[env_idx]
        return env.step(action)

    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            env.close()


def make_vec_env(
    factory: EnvFactory,
    num_envs: int = 4,
    use_subprocess: bool = False,
) -> VecEnv:
    """
    创建向量化环境的便捷函数

    Args:
        factory: 环境工厂
        num_envs: 并行环境数量
        use_subprocess: 是否使用子进程（当前未实现，必须为 False）

    Returns:
        VecEnv 实例
    """
    if use_subprocess:
        raise NotImplementedError("子进程模式尚未实现")

    envs = [factory.create() for _ in range(num_envs)]
    return VecEnv(envs)
