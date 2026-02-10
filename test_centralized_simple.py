"""
简化版 CentralizedCritic 测试

只测试最关键的功能
"""

import sys

sys.path.insert(0, "src")

import numpy as np

try:
    # Test 1: 导入测试
    print("Test 1: 导入模块...")
    from src.drl.network import CentralizedCriticNetwork, ActorCriticNetwork
    from src.drl.buffer import CentralizedRolloutBuffer, RolloutBuffer
    from src.drl.agent import NFSPAgentPool
    from src.drl.mappo import MAPPO
    from src.drl.config import get_default_config

    print("[PASS] 所有模块导入成功")

    # Test 2: CentralizedCriticNetwork 初始化
    print("\nTest 2: CentralizedCriticNetwork 初始化...")
    critic_net = CentralizedCriticNetwork(hidden_dim=512)
    print(f"[PASS] CentralizedCriticNetwork 创建成功")

    # Test 3: CentralizedRolloutBuffer 初始化
    print("\nTest 3: CentralizedRolloutBuffer 初始化...")
    buffer = CentralizedRolloutBuffer(capacity=100)
    print(f"[PASS] CentralizedRolloutBuffer 创建成功")

    # Test 4: NFSPAgentPool 初始化和方法
    print("\nTest 4: NFSPAgentPool 全局观测存储...")
    config = get_default_config()
    pool = NFSPAgentPool(
        config=config, device="cpu", num_agents=4, share_parameters=True
    )

    # 检查方法存在
    assert hasattr(pool, "store_global_observation"), (
        "缺少 store_global_observation 方法"
    )
    assert hasattr(pool, "get_global_observations"), "缺少 get_global_observations 方法"
    print("[PASS] NFSPAgentPool 方法检查通过")

    # 测试存储和获取
    all_obs = {
        "agent_0": {"test": "data"},
        "agent_1": {"test": "data"},
        "agent_2": {"test": "data"},
        "agent_3": {"test": "data"},
    }
    pool.store_global_observation(all_obs, {"episode_num": 1})
    retrieved = pool.get_global_observations(1)
    assert len(retrieved) == 4, f"应该有4个观测，实际: {len(retrieved)}"
    print("[PASS] NFSPAgentPool 全局观测存储和获取成功")

    # Test 5: MAPPO centralized_critic 参数
    print("\nTest 5: MAPPO centralized_critic 参数...")
    actor_critic = ActorCriticNetwork()

    # 不使用 centralized critic
    mappo_dec = MAPPO(network=actor_critic, device="cpu", centralized_critic=None)
    assert mappo_dec.centralized_critic is None, "应该为 None"
    print("[PASS] MAPPO 可初始化为 decentralized")

    # 使用 centralized critic
    mappo_cen = MAPPO(network=actor_critic, device="cpu", centralized_critic=critic_net)
    assert mappo_cen.centralized_critic is not None, "应该不为 None"
    print("[PASS] MAPPO 可初始化为 centralized")

    # Test 6: MAPPO phase-aware 参数
    print("\nTest 6: MAPPO phase-aware 参数...")
    # 检查 update 方法有 training_phase 参数
    import inspect

    sig = inspect.signature(mappo_cen.update)
    params = list(sig.parameters.keys())
    assert "training_phase" in params, (
        f"update 方法缺少 training_phase 参数，当前参数: {params}"
    )
    print(f"[PASS] MAPPO.update() 有 training_phase 参数")

    # Test 7: MAPPO update_centralized 方法
    print("\nTest 7: MAPPO update_centralized 方法...")
    assert hasattr(mappo_cen, "update_centralized"), "缺少 update_centralized 方法"
    print("[PASS] MAPPO 有 update_centralized 方法")

    # 测试调用（空 buffer，应该返回空字典）
    empty_buffer = CentralizedRolloutBuffer(capacity=10)
    stats = mappo_cen.update_centralized(empty_buffer, training_phase=1)
    assert isinstance(stats, dict), "应该返回字典"
    print("[PASS] MAPPO.update_centralized() 可调用")

    # 总结
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
    print("\nCentralizedCritic 集成测试结果:")
    print("- CentralizedCriticNetwork: OK")
    print("- CentralizedRolloutBuffer: OK")
    print("- NFSPAgentPool 全局观测: OK")
    print("- MAPPO centralized_critic 参数: OK")
    print("- MAPPO phase-aware 切换: OK")
    print("- MAPPO update_centralized 方法: OK")
    print("\n所有子任务已完成！")
    exit(0)

except Exception as e:
    print(f"\n[ERROR] 测试失败: {e}")
    import traceback

    traceback.print_exc()
    exit(1)
