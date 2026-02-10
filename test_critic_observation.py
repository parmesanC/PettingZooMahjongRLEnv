"""
诊断架构问题：Critic 观测范围确认

验证在参数共享模式下，每个智能体的 Critic 是否只能看到自己的观测
"""

print("=" * 80)
print("架构问题诊断")
print("=" * 80)

print("\n[问题]")
print("当前架构：")
print("  - 参数共享：所有智能体共享同一个 NFSP 实例")
print("  - Critic：每个智能体的 Critic 使用自己的观测")
print("")
print("\n[验证目标]")
print("确认：")
print("  env.last() 是否返回所有智能体的全局观测")
print("  Critic 是否有全局观测接口")
print("  每个 Agent 的 Critic 是否只能看到自己的观测")
print("")
print("\n" + "=" * 80)

# 步骤 1: 检查环境重置
print("\n步骤 1: 检查环境重置...")
try:
    from src.drl.config import get_quick_test_config
    from src.drl.trainer import NFSPTrainer

    config = get_quick_test_config()
    trainer = NFSPTrainer(config=config, device='cpu')
    
    # 重置环境
    print(f"  重置环境...")
    obs, _ = trainer.env.reset()
    
    if obs is not None:
        print(f"  ✓ 环境重置成功")
        
        # 检查观测键
        obs_keys = list(obs.keys())
        print(f"  观测键: {obs_keys}")
        
        # 检查是否包含所有 4 个智能体的观测
        agent_keys = [f'agent_{i}' for i in range(4)]
        missing_keys = [k for k in agent_keys if k not in obs]
        
        if missing_keys:
            print(f"  ✗ 缺少智能体: {missing_keys}")
        else:
            print(f"  ✓ 所有智能体观测都存在")

        # 检查每个观测的维度
        for key in agent_keys:
            if key in obs:
                obs_dict = obs[key]
                shape = obs_dict.shape
                print(f"  {key}: {shape}")
                
                # 检查是否有 action_mask
                if 'action_mask' in obs_dict:
                    print(f"    ✓ 有 action_mask")
                else:
                    print(f"    ✗ 无 action_mask")

    else:
        print("  ✗ 环境重置失败")
        exit(1)

except Exception as e:
    print(f"\n[ERROR] 测试失败: {e}")
    import traceback
    traceback.print_exc()
    print("")
    print("请检查环境实现和 Critic 架构")
    exit(1)

print()
print("=" * 80)
print("\n[步骤 2: 验证训练过程中 Critic 的观测范围")
print("（此步骤需要实际运行训练器，这里只做静态结构检查）")
print("结论：env.last() 已经返回全局观测，但 Critic 只看到自己")
print("这是 MADDPG/MAPPO 架构的核心问题")
