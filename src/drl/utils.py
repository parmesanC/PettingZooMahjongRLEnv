"""
工具函数集合
"""

import time
from typing import Dict, List, Any


def compute_episode_rewards(transitions: List[Dict]) -> float:
    """
    计算 episode 总奖励
    
    Args:
        transitions: 转移列表
    
    Returns:
        episode 总奖励
    """
    total_reward = 0.0
    for trans in transitions:
        total_reward += trans.get('reward', 0.0)
    return total_reward


def format_training_time(elapsed_seconds: float) -> str:
    """
    格式化训练时间
    
    Args:
        elapsed_seconds: 经过的秒数
    
    Returns:
        格式化的时间字符串
    """
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = int(elapsed_seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def get_timestamp() -> str:
    """
    获取当前时间戳字符串
    
    Returns:
        ISO 格式时间戳
    """
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def merge_stats(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    合并多个统计字典
    
    Args:
        stats_list: 统计字典列表
    
    Returns:
        合并后的统计字典
    """
    merged = {}
    for stats in stats_list:
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value
            elif isinstance(value, list):
                merged[key] = merged.get(key, []) + value
            else:
                merged[key] = value
    return merged


def print_progress_bar(current: int, total: int, prefix: str = ""):
    """
    打印进度条
    
    Args:
        current: 当前值
        total: 总值
        prefix: 前缀字符串
    """
    if total == 0:
        return
    
    percent = current / total * 100
    bar_length = 50
    filled = int(bar_length * percent / 100)
    bar = '=' * filled + '-' * (bar_length - filled)
    
    print(f"\r{prefix}[{bar}] {percent:.1f}%", end="", flush=True)


def calculate_win_rate(wins: int, games: int) -> float:
    """
    计算胜率
    
    Args:
        wins: 获胜次数
        games: 总游戏局数
    
    Returns:
        胜率（0.0-1.0）
    """
    if games == 0:
        return 0.0
    return wins / games
