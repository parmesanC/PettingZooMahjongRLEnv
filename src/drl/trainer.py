"""
NFSP 训练器

完整的训练循环，包含：
1. 自对弈（Self-Play）
2. 对手策略切换（前期随机 → 后期历史版本）
3. 评估（与随机策略对战）
4. 模型保存和日志记录
"""

import os
import time
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import deque
import random

from example_mahjong_env import WuhanMahjongEnv
from .agent import NFSPAgentPool, NFSPAgentWrapper, RandomOpponent
from .config import Config, get_default_config, get_quick_test_config


class NFSPTrainer:
    """
    NFSP 训练器
    
    管理完整的训练流程：
    - 500万局自对弈
    - 前期（100万局）使用随机对手
    - 后期（400万局）使用历史版本的平均策略网络
    - 每1000局评估一次（与随机策略对战）
    - 每10000局保存一次模型
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        device: str = 'cuda',
        log_dir: str = 'logs',
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        初始化训练器
        
        Args:
            config: 配置对象
            device: 计算设备
            log_dir: 日志目录
            checkpoint_dir: 检查点目录
        """
        self.config = config or get_default_config()
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 创建环境
        self.env = WuhanMahjongEnv(
            render_mode=None,
            training_phase=self.config.mahjong.training_phase,
            enable_logging=False
        )
        
        # 创建 NFSP 智能体池（参数共享）
        self.agent_pool = NFSPAgentPool(
            config=self.config,
            device=device,
            num_agents=4,
            share_parameters=True
        )
        
        # 随机对手（用于前期训练和评估）
        self.random_opponent = RandomOpponent()
        
        # 历史策略池（用于后期自对弈）
        self.policy_pool = []
        self.policy_pool_size = 10  # 保留最近10个策略
        
        # 训练统计
        self.episode_count = 0
        self.start_time = time.time()
        self.eval_results = []
        
        # 日志
        self.log_file = os.path.join(log_dir, 'training_log.jsonl')
        
        # 设置随机种子
        if self.config.training.seed is not None:
            self._set_seed(self.config.training.seed)
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def train(self):
        """
        主训练循环
        
        训练 500万局：
        - 0-100万局：vs 随机对手
        - 100-500万局：vs 历史版本的平均策略网络
        """
        print("=" * 80)
        print("NFSP 训练开始")
        print(f"总训练局数: {self.config.training.total_episodes:,}")
        print(f"切换点: {self.config.training.switch_point:,} 局")
        print(f"设备: {self.device}")
        print("=" * 80)
        
        try:
            while self.episode_count < self.config.training.total_episodes:
                # 运行一局自对弈
                episode_stats = self._run_episode()
                
                self.episode_count += 1
                
                # 训练
                train_stats = self.agent_pool.train_all()
                
                # 定期评估
                if self.episode_count % self.config.training.eval_interval == 0:
                    eval_stats = self._evaluate()
                    self._log_stats(episode_stats, train_stats, eval_stats)
                    self._print_progress(train_stats, eval_stats)
                
                # 定期保存
                if self.episode_count % self.config.training.save_interval == 0:
                    self._save_checkpoint()
                    self._add_to_policy_pool()
                
        except KeyboardInterrupt:
            print("\n训练被中断")
        finally:
            # 保存最终模型
            self._save_checkpoint(is_final=True)
            self._save_training_summary()
            print("\n训练完成！")
    
    def _run_episode(self) -> Dict:
        """
        运行一局自对弈
        
        Returns:
            回合统计
        """
        # 重置环境
        obs, _ = self.env.reset()
        
        # 确定对手类型
        if self.episode_count < self.config.training.switch_point:
            # 前期：使用随机对手
            use_random_opponents = True
        else:
            # 后期：使用历史策略
            use_random_opponents = False
        
        # 回合数据
        episode_rewards = [0.0] * 4
        episode_steps = 0
        winner = None
        
        # 运行一局
        while self.env.agents:
            current_agent_id = self.env.agent_selection
            if current_agent_id is None:
                break
            
            agent_idx = int(current_agent_id.split('_')[1])
            
            # 获取观测
            obs_dict = self.env.observe(current_agent_id)
            action_mask = obs_dict['action_mask']
            
            # 选择动作
            if use_random_opponents:
                # 使用随机对手
                action_type, action_param = self.random_opponent.choose_action(
                    obs_dict, action_mask
                )
            else:
                # 使用 NFSP 智能体
                agent = self.agent_pool.get_agent(agent_idx)
                action_type, action_param = agent.choose_action(obs_dict, action_mask)
            
            # 执行动作
            action = (action_type, action_param)
            next_obs_dict, reward, terminated, truncated, info = self.env.step(action)
            
            # 记录奖励
            episode_rewards[agent_idx] += reward
            
            # 存储转移（仅 NFSP 智能体）
            if not use_random_opponents:
                agent = self.agent_pool.get_agent(agent_idx)
                if hasattr(agent, 'store_transition'):
                    agent.store_transition(
                        reward=reward,
                        done=terminated or truncated,
                        next_observation=next_obs_dict if not (terminated or truncated) else None,
                        next_action_mask=next_obs_dict.get('action_mask') if not (terminated or truncated) else None
                    )
            
            # 检查获胜者
            if terminated and info.get('winners'):
                winner = info['winners'][0] if info['winners'] else None
            
            episode_steps += 1
        
        # 结束回合
        for i in range(4):
            agent = self.agent_pool.get_agent(i)
            if hasattr(agent, 'end_episode'):
                agent.end_episode(won=(winner == i))
        
        return {
            'episode': self.episode_count,
            'steps': episode_steps,
            'rewards': episode_rewards,
            'winner': winner
        }
    
    def _evaluate(self, num_games: int = 100) -> Dict:
        """
        评估智能体（与随机策略对战）
        
        Args:
            num_games: 对战局数
        
        Returns:
            评估统计
        """
        wins = 0
        total_reward = 0.0
        
        for _ in range(num_games):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            winner = None
            
            while self.env.agents:
                current_agent_id = self.env.agent_selection
                if current_agent_id is None:
                    break
                
                agent_idx = int(current_agent_id.split('_')[1])
                obs_dict = self.env.observe(current_agent_id)
                action_mask = obs_dict['action_mask']
                
                # 玩家0使用 NFSP，其他使用随机
                if agent_idx == 0:
                    agent = self.agent_pool.get_agent(0)
                    action_type, action_param = agent.choose_action(obs_dict, action_mask)
                else:
                    action_type, action_param = self.random_opponent.choose_action(
                        obs_dict, action_mask
                    )
                
                action = (action_type, action_param)
                obs_dict, reward, terminated, truncated, info = self.env.step(action)
                
                if agent_idx == 0:
                    episode_reward += reward
                
                if terminated and info.get('winners'):
                    winner = info['winners'][0] if info['winners'] else None
            
            if winner == 0:
                wins += 1
            total_reward += episode_reward
        
        win_rate = wins / num_games
        avg_reward = total_reward / num_games
        
        eval_stats = {
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'wins': wins,
            'games': num_games
        }
        
        self.eval_results.append(eval_stats)
        return eval_stats
    
    def _log_stats(
        self,
        episode_stats: Dict,
        train_stats: Dict,
        eval_stats: Dict
    ):
        """记录统计信息"""
        log_entry = {
            'episode': self.episode_count,
            'timestamp': time.time() - self.start_time,
            'episode_stats': episode_stats,
            'train_stats': train_stats,
            'eval_stats': eval_stats
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _print_progress(self, train_stats: Dict, eval_stats: Dict):
        """打印训练进度"""
        elapsed = time.time() - self.start_time
        eps_per_sec = self.episode_count / elapsed if elapsed > 0 else 0
        remaining_eps = self.config.training.total_episodes - self.episode_count
        eta_seconds = remaining_eps / eps_per_sec if eps_per_sec > 0 else 0
        
        print(f"\nEpisode {self.episode_count:,}/{self.config.training.total_episodes:,}")
        print(f"  Win Rate vs Random: {eval_stats['win_rate']:.2%}")
        print(f"  Avg Reward: {eval_stats['avg_reward']:.3f}")
        print(f"  Speed: {eps_per_sec:.1f} eps/sec")
        print(f"  ETA: {eta_seconds/3600:.1f} hours")
        
        if train_stats:
            print(f"  RL Loss: {train_stats.get('loss', 0):.4f}")
            print(f"  SL Loss: {train_stats.get('sl_loss', 0):.4f}")
    
    def _save_checkpoint(self, is_final: bool = False):
        """保存检查点"""
        if is_final:
            path = os.path.join(self.checkpoint_dir, 'final_model.pth')
        else:
            path = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_{self.episode_count}.pth'
            )
        
        self.agent_pool.save(path)
        print(f"  Saved checkpoint: {path}")
    
    def _add_to_policy_pool(self):
        """将当前策略添加到历史池"""
        # 保存当前平均策略网络
        policy_path = os.path.join(
            self.checkpoint_dir,
            f'policy_{self.episode_count}.pth'
        )
        self.agent_pool.save(policy_path)
        
        self.policy_pool.append(policy_path)
        
        # 限制池大小
        if len(self.policy_pool) > self.policy_pool_size:
            old_policy = self.policy_pool.pop(0)
            if os.path.exists(old_policy):
                os.remove(old_policy)
    
    def _save_training_summary(self):
        """保存训练总结"""
        summary = {
            'total_episodes': self.episode_count,
            'total_time': time.time() - self.start_time,
            'final_eval': self.eval_results[-1] if self.eval_results else None,
            'best_win_rate': max([r['win_rate'] for r in self.eval_results]) if self.eval_results else 0,
            'config': {
                'total_episodes': self.config.training.total_episodes,
                'switch_point': self.config.training.switch_point,
                'eta': self.config.nfsp.eta,
                'hidden_dim': self.config.network.hidden_dim,
                'transformer_layers': self.config.network.transformer_layers
            }
        }
        
        summary_path = os.path.join(self.log_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary saved: {summary_path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        self.agent_pool.load(path)
        print(f"Loaded checkpoint: {path}")


def train_nfsp(
    config: Optional[Config] = None,
    device: str = 'cuda',
    quick_test: bool = False
):
    """
    训练 NFSP 智能体的便捷函数
    
    Args:
        config: 配置对象（None 则使用默认配置）
        device: 计算设备
        quick_test: 是否使用快速测试配置
    """
    if quick_test:
        config = get_quick_test_config()
        print("Using quick test config")
    elif config is None:
        config = get_default_config()
    
    # 检查 CUDA
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # 创建训练器
    trainer = NFSPTrainer(
        config=config,
        device=device
    )
    
    # 开始训练
    trainer.train()
    
    return trainer


if __name__ == '__main__':
    # 快速测试
    train_nfsp(quick_test=True)
