"""
NFSP + MAPPO + Transformer 麻将智能体训练脚本

使用方法：
    # 标准训练（500万局，约1周）
    python train_nfsp.py
    
    # 快速测试（1万局）
    python train_nfsp.py --quick-test
    
    # 使用 CPU
    python train_nfsp.py --device cpu
    
    # 自定义配置
    python train_nfsp.py --episodes 1000000 --eta 0.15

作者：汪呜呜
"""

import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.drl.trainer import train_nfsp
from src.drl.config import get_default_config, get_quick_test_config


def main():
    parser = argparse.ArgumentParser(
        description='训练 NFSP + MAPPO + Transformer 麻将智能体',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train_nfsp.py                           # 标准训练
  python train_nfsp.py --quick-test              # 快速测试
  python train_nfsp.py --device cpu              # 使用 CPU
  python train_nfsp.py --episodes 1000000        # 自定义局数
  python train_nfsp.py --eta 0.15                # 自定义 anticipatory 参数
        """
    )
    
    # 基本参数
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='快速测试模式（1万局，小网络）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='计算设备（默认: cuda）'
    )
    
    # 训练参数
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='总训练局数（默认: 500万）'
    )
    
    parser.add_argument(
        '--switch-point',
        type=int,
        default=None,
        help='切换对手的局数（默认: 100万）'
    )
    
    parser.add_argument(
        '--eta',
        type=float,
        default=None,
        help='Anticipatory 参数（默认: 0.2）'
    )
    
    # 网络参数
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=None,
        help='隐藏层维度（默认: 256）'
    )
    
    parser.add_argument(
        '--transformer-layers',
        type=int,
        default=None,
        help='Transformer 层数（默认: 4）'
    )
    
    # 其他参数
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认: 42）'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='日志目录（默认: logs）'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='检查点目录（默认: checkpoints）'
    )
    
    args = parser.parse_args()
    
    # 获取配置
    if args.quick_test:
        config = get_quick_test_config()
        print("=" * 80)
        print("🚀 快速测试模式")
        print("=" * 80)
    else:
        config = get_default_config()
        print("=" * 80)
        print("🎮 NFSP + MAPPO + Transformer 麻将智能体训练")
        print("=" * 80)
    
    # 覆盖配置参数
    if args.episodes is not None:
        config.training.total_episodes = args.episodes
    
    if args.switch_point is not None:
        config.training.switch_point = args.switch_point
    
    if args.eta is not None:
        config.nfsp.eta = args.eta
    
    if args.hidden_dim is not None:
        config.network.hidden_dim = args.hidden_dim
    
    if args.transformer_layers is not None:
        config.network.transformer_layers = args.transformer_layers
    
    if args.seed is not None:
        config.training.seed = args.seed
    
    # 打印配置
    print("\n📋 训练配置:")
    print(f"  总训练局数: {config.training.total_episodes:,}")
    print(f"  切换点: {config.training.switch_point:,} 局")
    print(f"  Anticipatory 参数 (η): {config.nfsp.eta}")
    print(f"  隐藏层维度: {config.network.hidden_dim}")
    print(f"  Transformer 层数: {config.network.transformer_layers}")
    print(f"  设备: {args.device}")
    print(f"  随机种子: {config.training.seed}")
    print(f"  日志目录: {args.log_dir}")
    print(f"  检查点目录: {args.checkpoint_dir}")
    print("=" * 80)
    
    # 开始训练
    try:
        trainer = train_nfsp(
            config=config,
            device=args.device
        )
        
        print("\n✅ 训练完成！")
        print(f"日志保存于: {args.log_dir}")
        print(f"模型保存于: {args.checkpoint_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
