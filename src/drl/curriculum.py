"""
课程学习调度器

实现三阶段课程学习：
- 阶段1（全知视角）
- 阶段2（渐进式掩码）
- 阶段3（真实环境）
"""

import os


class CurriculumScheduler:
    """
    课程学习调度器
    
    管理训练阶段切换，支持两种训练模式：
    - 快速测试模式（10万局）
    - 完整训练模式（2000万局）
    """
    
    def __init__(self, total_episodes=20_000_000):
        """
        初始化课程学习调度器
        
        Args:
            total_episodes: 总训练局数
        """
        self.total_episodes = total_episodes
        
        # 阶段划分（各占33.3%）
        self.phase1_end = int(total_episodes * 0.333)
        self.phase2_end = int(total_episodes * 0.666)
        self.phase3_end = total_episodes
        
        # 阶段名称映射
        self.phase_names = {
            1: "Omniscient",      # 全知视角
            2: "Progressive",     # 渐进式掩码
            3: "Real"            # 真实环境
        }
    
    def get_phase(self, episode: int) -> tuple[int, float]:
        """
        获取当前训练阶段和进度

        Args:
            episode: 当前局数（从0开始）

        Returns:
            (phase, progress) 元组
            - phase: 1/2/3
            - progress: 掩码进度（阶段1=0.0, 阶段2=0.0→1.0, 阶段3=1.0）
        """
        if episode < self.phase1_end:
            # 阶段1：全知视角（无掩码）
            return 1, 0.0
        elif episode < self.phase2_end:
            # 阶段2：渐进式掩码
            phase1_episodes = self.phase1_end
            phase2_duration = self.phase2_end - self.phase1_end
            episode_in_phase = episode - phase1_episodes
            if phase2_duration > 0:
                progress = episode_in_phase / phase2_duration
            else:
                progress = 1.0
            return 2, progress
        else:
            # 阶段3：真实环境（完全掩码）
            return 3, 1.0
    
    def get_phase_name(self, phase: int) -> str:
        """
        获取阶段名称
        
        Args:
            phase: 阶段编号（1/2/3）
        
        Returns:
            阶段名称
        """
        return self.phase_names.get(phase, f"Phase {phase}")
    
    def get_phase_info(self, episode: int) -> dict:
        """
        获取当前阶段的完整信息
        
        Args:
            episode: 当前局数
        
        Returns:
            包含阶段信息的字典
        """
        phase, progress = self.get_phase(episode)
        
        return {
            'episode': episode,
            'phase': phase,
            'phase_name': self.get_phase_name(phase),
            'progress': progress,
            'training_phase': phase,  # 用于环境的 training_phase 参数
            'training_progress': progress  # 用于环境的 training_progress 参数
        }


def create_scheduler(mode: str = 'full_training') -> CurriculumScheduler:
    """
    便捷函数：创建课程学习调度器
    
    Args:
        mode: 训练模式
            - 'quick_test': 快速测试模式（10万局）
            - 'full_training': 完整训练模式（2000万局）
    
    Returns:
        CurriculumScheduler 实例
    """
    if mode == 'quick_test':
        return CurriculumScheduler(total_episodes=100_000)
    elif mode == 'full_training':
        return CurriculumScheduler(total_episodes=20_000_000)
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'quick_test' or 'full_training'")
