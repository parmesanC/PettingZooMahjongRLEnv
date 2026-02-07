"""
测试可见度掩码的三阶段课程学习
"""
import numpy as np
from example_mahjong_env import WuhanMahjongEnv


def test_phase_1_omniscient():
    """阶段1：全知视角，所有信息应可见"""
    env = WuhanMahjongEnv(training_phase=1)
    obs, _ = env.reset(seed=42)

    current_agent_id = env.agents_name_mapping[env.agent_selection]

    # 验证对手手牌可见（不全为5）
    global_hand = obs['global_hand']
    for i in range(4):
        if i != current_agent_id:
            player_hand = global_hand[i * 34:(i + 1) * 34]
            # 应该有非5的值（实际牌数）
            assert not np.all(player_hand == 5), \
                f"Phase 1: Agent {current_agent_id} should see opponent {i}'s hand"

    # 验证牌墙可见（不全为34）
    assert not np.all(obs['wall'] == 34), "Phase 1: Wall should be visible"

    print("[OK] Phase 1 (omniscient) test passed")


def test_phase_2_progressive():
    """阶段2：渐进式掩码"""
    # progress=0.0 应接近无掩码
    env1 = WuhanMahjongEnv(training_phase=2, phase2_progress=0.0)
    obs1, _ = env1.reset(seed=42)
    obs1 = env1.observe(env1.agent_selection)
    # 大部分情况下对手手牌应该可见
    opponent_hand = obs1['global_hand'][0:34]  # 假设当前是player_0
    # 由于随机性，多次测试应该有可见的情况
    visible_count = 0
    for _ in range(10):
        obs, _ = env1.reset(seed=42)
        obs = env1.observe(env1.agent_selection)
        if not np.all(obs['global_hand'][34:68] == 5):
            visible_count += 1
    assert visible_count > 0, "Phase 2 progress=0.0: Should sometimes show opponent hands"
    print(f"Phase 2 progress=0.0: Opponent hands visible in {visible_count}/10 resets")

    # progress=1.0 应接近完全掩码
    env2 = WuhanMahjongEnv(training_phase=2, phase2_progress=1.0)
    masked_count = 0
    for _ in range(10):
        obs, _ = env2.reset(seed=42)
        obs = env2.observe(env2.agent_selection)
        if np.all(obs['global_hand'][34:68] == 5) and np.all(obs['wall'] == 34):
            masked_count += 1
    assert masked_count > 0, "Phase 2 progress=1.0: Should sometimes mask opponent hands"
    print(f"Phase 2 progress=1.0: Opponent hands masked in {masked_count}/10 resets")

    print("[OK] Phase 2 (progressive) test passed")


def test_phase_3_real_state():
    """阶段3：真实状态，对手信息应被掩码"""
    env = WuhanMahjongEnv(training_phase=3)
    obs, _ = env.reset(seed=42)

    current_agent_id = env.agents_name_mapping[env.agent_selection]

    # 验证对手手牌被掩码（全为5）
    global_hand = obs['global_hand']
    for i in range(4):
        if i != current_agent_id:
            player_hand = global_hand[i * 34:(i + 1) * 34]
            assert np.all(player_hand == 5), \
                f"Phase 3: Agent {current_agent_id} should NOT see opponent {i}'s hand"

    # 验证牌墙被掩码（全为34）
    assert np.all(obs['wall'] == 34), "Phase 3: Wall should be masked"

    print("[OK] Phase 3 (real state) test passed")


def test_concealed_kong_masking():
    """测试暗杠牌的掩码"""
    env = WuhanMahjongEnv(training_phase=3)
    obs, _ = env.reset(seed=42)

    action_types = obs['melds']['action_types']
    tiles = obs['melds']['tiles']
    KONG_CONCEALED = 5

    # 检查是否有暗杠
    has_concealed_kong = False
    for player_id in range(4):
        for meld_idx in range(4):
            idx = player_id * 4 + meld_idx
            if action_types[idx] == KONG_CONCEALED:
                has_concealed_kong = True
                # 检查暗杠的牌是否被掩码为34
                base_tile_idx = (player_id * 4 * 4 + meld_idx * 4) * 34
                for tile_pos in range(4):
                    tile_start = base_tile_idx + tile_pos * 34
                    tile_section = tiles[tile_start:tile_start + 34]
                    # 应该全为34或全为0（取决于是否是该玩家的观测）
                    if np.all(tile_section == 34) or np.all(tile_section == 0):
                        pass  # 正确：被掩码或该位置没有牌
                    else:
                        raise AssertionError(
                            f"Concealed kong tiles should be masked to 34 or 0, "
                            f"got {tile_section}"
                        )

    if has_concealed_kong:
        print("[OK] Concealed kong masking test passed")
    else:
        print("[WARNING] No concealed kong found in test game, skipping masked tile check")


def test_phase_2_correlated_masking():
    """测试阶段2的关联掩码（对手手牌、牌墙、暗杠同时掩码或同时不掩码）"""
    env = WuhanMahjongEnv(training_phase=2, phase2_progress=0.5)

    # 多次测试，验证掩码是关联的
    consistent_count = 0
    total_tests = 20

    for _ in range(total_tests):
        obs, _ = env.reset(seed=42)
        obs = env.observe(env.agent_selection)

        # 检查掩码状态是否一致
        opponent_masked = np.all(obs['global_hand'][34:68] == 5)
        wall_masked = np.all(obs['wall'] == 34)

        # 对手手牌和牌墙应该同时掩码或同时不掩码
        if opponent_masked == wall_masked:
            consistent_count += 1

    # 至少90%的情况下应该是一致的（允许一些随机性）
    consistency_ratio = consistent_count / total_tests
    assert consistency_ratio >= 0.9, \
        f"Phase 2 correlated masking: Expected >=90% consistency, got {consistency_ratio*100}%"

    print(f"[OK] Phase 2 correlated masking test passed ({consistent_count}/{total_tests} consistent)")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Visibility Mask Curriculum Learning")
    print("=" * 60)

    test_phase_1_omniscient()
    test_phase_2_progressive()
    test_phase_3_real_state()
    test_concealed_kong_masking()
    test_phase_2_correlated_masking()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
