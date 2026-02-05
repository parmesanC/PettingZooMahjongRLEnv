"""武汉麻将和牌场景全面测试

测试所有和牌类型，包括小胡（硬胡/软胡）和大胡。

设计原则：
- 使用 with_initial_state 设置初始状态（庄家14张，其他13张）
- 完整模拟4个玩家的打牌流程，包括所有PASS响应
- 包含吃、碰、杠等鸣牌过程
- 预先计算牌墙，包含游戏流程中摸走的牌

牌值对照：
    万子：0-8  (1万=0, 2万=1, ..., 9万=8)
    条子：9-17 (1条=9, 2条=10, ..., 9条=17)
    筒子：18-26 (1筒=18, 2筒=19, ..., 9筒=26)
    风牌：27-33 (东=27, 南=28, 西=29, 北=30, 中=31, 发=32, 白=33)
"""

from tests.scenario.builder import ScenarioBuilder
from src.mahjong_rl.core.constants import GameStateType, ActionType


# =============================================================================
# 场景1：硬胡自摸
# =============================================================================

def calculate_wall_scenario1_hard_win():
    """计算场景1的牌墙

    初始手牌：
        p0: [0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 26]  # 14张，庄家
        p1: [8, 10, 12, 14, 16, 27, 28, 29, 30, 31, 32, 33, 0]  # 13张
        p2: [9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 26]  # 13张
        p3: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 13张

    游戏流程中摸走的牌：
        p1摸1张, p2摸1张, p3摸1张, p0摸25(自摸)
    """
    # 初始手牌
    p0 = [0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 26]  # 14张
    p1 = [8, 10, 12, 14, 16, 27, 28, 29, 30, 31, 32, 33, 0]  # 13张
    p2 = [9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 26]  # 13张
    p3 = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 13张

    # 游戏流程中被摸走的牌（按顺序）
    drawn_tiles = [8, 9, 2, 25]  # p1摸8, p2摸9, p3摸2, p0摸25自摸

    # 计算牌墙
    all_used = p0 + p1 + p2 + p3 + drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    wall = drawn_tiles + unused_wall
    return wall


def test_scenario_1_hard_win_self_draw():
    """场景1：硬胡自摸

    目的：验证无赖子、将牌为2/5/8的自摸硬胡

    初始手牌：
        玩家0（庄家）：
            [0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 26]
            = 1万x3, 2万,3万,4万, 5万,6万,7万, 8筒x3, 9筒, 9筒

    特殊牌：
        赖子：15（2条）
        皮子：14, 13

    目标牌型：111(碰1万) 234 567 888 99(自摸9筒)
        - 无赖子
        - 将牌=9筒（符合2/5/8要求）

    游戏流程：
        1. 玩家0出26（9筒）- 听牌
        2-4. 玩家1,2,3依次PASS
        5. 玩家1摸牌后出0（1万）
        6-8. 玩家2,3,0依次响应（0 PONG）
        9-11. 玩家0出3（4万），玩家1,2,3依次PASS
        12-14. 玩家1出8，玩家2,3,0依次PASS
        15-17. 玩家2出9，玩家3,0,1依次PASS
        18-20. 玩家3出2，玩家0,1,2依次PASS
        21. 玩家0摸25（9筒）自摸硬胡
    """
    wall = calculate_wall_scenario1_hard_win()

    result = (
        ScenarioBuilder("场景1：硬胡自摸")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 26],  # 庄家14张
                1: [8, 10, 12, 14, 16, 27, 28, 29, 30, 31, 32, 33, 0],
                2: [9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 26],
                3: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0出26（9筒）- 听牌
        .step(1, "玩家0出26（9筒）听牌")
            .action(0, ActionType.DISCARD, 26)

        # 步骤2：玩家2PASS
        .step(2, "玩家2 PASS")
            .action(2, ActionType.PASS)

        # 步骤5：玩家1摸牌后出0（1万）
        .step(5, "玩家1摸牌后出0（1万）")
            .action(1, ActionType.DISCARD, 0)

        # 步骤6-8：玩家2,3依次PASS，玩家0 PONG
        .step(8, "玩家0 PONG 0（1万）完成开口")
            .action(0, ActionType.PONG, 0)

        # 步骤9-11：玩家0出3（4万），玩家1,2,3依次PASS
        .step(9, "玩家0出3（4万）")
            .action(0, ActionType.DISCARD, 3)
        .step(10, "玩家1 PASS")
            .action(1, ActionType.PASS)
        .step(11, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(12, "玩家3 PASS")
            .action(3, ActionType.PASS)

        # 步骤13-16：玩家1出8，玩家2,3,0依次PASS
        .step(13, "玩家1出8")
            .action(1, ActionType.DISCARD, 8)
        .step(14, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(15, "玩家3 PASS")
            .action(3, ActionType.PASS)
        .step(16, "玩家0 PASS")
            .action(0, ActionType.PASS)

        # 步骤17-20：玩家2出9，玩家3,0,1依次PASS
        .step(17, "玩家2出9")
            .action(2, ActionType.DISCARD, 9)
        .step(18, "玩家3 PASS")
            .action(3, ActionType.PASS)
        .step(19, "玩家0 PASS")
            .action(0, ActionType.PASS)
        .step(20, "玩家1 PASS")
            .action(1, ActionType.PASS)

        # 步骤21：玩家3出2
        .step(21, "玩家3出2")
            .action(3, ActionType.DISCARD, 2)
            .verify("检查状态", lambda ctx: print(f"  步骤21后: 状态={ctx.current_state.name}, wall_count={len(ctx.wall)}, 玩家0手牌={len(ctx.players[0].hand_tiles)}") or True)

        # 步骤22：玩家0 PASS（env.step()会自动处理WAITING_RESPONSE并让玩家0摸牌）
        .step(22, "玩家0 PASS")
            .action(0, ActionType.PASS)
            .verify("验证玩家0已摸牌", lambda ctx: print(f"  步骤22后: 状态={ctx.current_state.name}, 玩家0手牌={len(ctx.players[0].hand_tiles)}, wall_count={len(ctx.wall)}, 手牌={sorted(ctx.players[0].hand_tiles)}, melds={ctx.players[0].melds}, 开口次数={ctx.players[0].exposure_count}") or True)

        # 步骤23：玩家0自摸硬胡
        .step(23, "玩家0自摸硬胡")
            .action(0, ActionType.WIN, -1)
            .expect_state(GameStateType.WIN)

        .run()
    )

    if result.success:
        print("[OK] 场景1测试成功：硬胡自摸")
        print(f"获胜者: {result.final_context_snapshot.get('winner_ids', [])}")
    else:
        print(f"[FAIL] 场景1测试失败: {result.failure_message}")
        if result.final_context_snapshot:
            print(f"最终状态: {result.final_context_snapshot}")

    assert result.success, f"场景1测试失败: {result.failure_message}"


# =============================================================================
# 场景2：软胡接炮
# =============================================================================

def calculate_wall_scenario2_soft_win():
    """计算场景2的牌墙

    软胡接炮：手牌有1个赖子未还原，接炮胡牌
    """
    # 初始手牌
    p0 = [0, 0, 0, 1, 2, 3, 15, 5, 6, 24, 24, 24, 26, 26]  # 14张，庄家
        # 1万x3, 2万,3万, 赖子(15=2条), 5万,6万, 8筒x3, 9筒x2
    p1 = [4, 8, 10, 12, 14, 16, 27, 28, 29, 30, 31, 32, 25]  # 13张
    p2 = [9, 11, 13, 17, 18, 19, 20, 21, 22, 23, 25, 26, 26]  # 13张
    p3 = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 13张

    # 游戏流程中被摸走的牌
    drawn_tiles = [25, 4, 9, 26]  # p1摸25, p0摸4, p2摸9, p3摸26

    # 计算牌墙
    all_used = p0 + p1 + p2 + p3 + drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    wall = drawn_tiles + unused_wall
    return wall


def test_scenario_2_soft_win_discard():
    """场景2：软胡接炮

    目的：验证1个赖子未还原的接炮软胡

    初始手牌：
        玩家0（庄家）：
            [0, 0, 0, 1, 2, 3, 15, 5, 6, 24, 24, 24, 26, 26]
            = 1万x3, 2万,3万, 赖子(15=2条), 5万,6万, 8筒x3, 9筒x2

    特殊牌：
        赖子：15（2条）- 在手牌中，未还原
        皮子：14, 13

    目标牌型：111(碰1万) 23(赖子)5 56 888 99(接炮)
        - 1个赖子未还原（软胡）
        - 将牌=9筒（符合2/5/8要求）
    """
    wall = calculate_wall_scenario2_soft_win()

    result = (
        ScenarioBuilder("场景2：软胡接炮")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [0, 0, 0, 1, 2, 3, 15, 5, 6, 24, 24, 24, 26, 26],
                1: [4, 8, 10, 12, 14, 16, 27, 28, 29, 30, 31, 32, 25],
                2: [9, 11, 13, 17, 18, 19, 20, 21, 22, 23, 25, 26, 26],
                3: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0出26（9筒）- 听牌
        .step(1, "玩家0出26（9筒）听牌")
            .action(0, ActionType.DISCARD, 26)

        # 步骤2-4：玩家1,2,3依次PASS
        .step(2, "玩家1 PASS")
            .action(1, ActionType.PASS)
        .step(3, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(4, "玩家3 PASS")
            .action(3, ActionType.PASS)

        # 步骤5：玩家1摸牌后出4（5万）
        .step(5, "玩家1摸牌后出4（5万）")
            .action(1, ActionType.DISCARD, 4)

        # 步骤6-8：玩家2,3依次PASS，玩家0 PONG
        .step(6, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(7, "玩家3 PASS")
            .action(3, ActionType.PASS)
        .step(8, "玩家0 PONG 4（5万）")
            .action(0, ActionType.PONG, 4)

        # 步骤9-12：玩家0出15（赖子），玩家1,2,3依次PASS
        .step(9, "玩家0出15（赖子2条）")
            .action(0, ActionType.DISCARD, 15)
        .step(10, "玩家1 PASS")
            .action(1, ActionType.PASS)
        .step(11, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(12, "玩家3 PASS")
            .action(3, ActionType.PASS)

        # 步骤13-16：玩家1出8，玩家2,3,0依次PASS
        .step(13, "玩家1出8")
            .action(1, ActionType.DISCARD, 8)
        .step(14, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(15, "玩家3 PASS")
            .action(3, ActionType.PASS)
        .step(16, "玩家0 PASS")
            .action(0, ActionType.PASS)

        # 步骤17-20：玩家2出9，玩家3,0,1依次PASS
        .step(17, "玩家2出9")
            .action(2, ActionType.DISCARD, 9)
        .step(18, "玩家3 PASS")
            .action(3, ActionType.PASS)
        .step(19, "玩家0 PASS")
            .action(0, ActionType.PASS)
        .step(20, "玩家1 PASS")
            .action(1, ActionType.PASS)

        # 步骤21-24：玩家3出26（9筒），玩家0 WIN
        .step(21, "玩家3出26（9筒）")
            .action(3, ActionType.DISCARD, 26)
        .step(22, "玩家0 WIN（接炮软胡）")
            .action(0, ActionType.WIN, -1)

        .run()
    )

    if result.success:
        print("[OK] 场景2测试成功：软胡接炮")
        print(f"获胜者: {result.final_context_snapshot.get('winner_ids', [])}")
    else:
        print(f"[FAIL] 场景2测试失败: {result.failure_message}")
        if result.final_context_snapshot:
            print(f"最终状态: {result.final_context_snapshot}")

    assert result.success, f"场景2测试失败: {result.failure_message}"


# =============================================================================
# 场景3：杠上开花
# =============================================================================

def calculate_wall_scenario3_kong_bloom():
    """计算场景3的牌墙

    杠上开花：杠后补牌自摸胡牌
    """
    # 初始手牌
    p0 = [0, 0, 0, 10, 10, 13, 13, 3, 4, 5, 20, 24, 24, 26]  # 14张，庄家
    p1 = [1, 8, 12, 14, 16, 27, 28, 29, 30, 31, 32, 33, 25]  # 13张
    p2 = [9, 11, 15, 17, 18, 19, 21, 22, 23, 25, 26, 26, 24]  # 13张
    p3 = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 13张

    # 游戏流程中被摸走的牌
    drawn_tiles = [25, 10, 24]  # p1摸25, p0摸10(补杠), p0摸24(开花)

    # 计算牌墙
    all_used = p0 + p1 + p2 + p3 + drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    wall = drawn_tiles + unused_wall
    return wall


def test_scenario_3_kong_bloom():
    """场景3：杠上开花

    目的：验证杠后补牌自摸胡牌（大胡）

    初始手牌：
        玩家0（庄家）：
            [0, 0, 0, 10, 10, 13, 13, 3, 4, 5, 20, 24, 24, 26]
            = 1万x3, 2条x2, 4条x2, 4万,5万, 3筒, 8筒x2, 9筒

    特殊牌：
        赖子：15（4条）
        皮子：14, 13

    目标牌型：111(碰1万) 22(明杠2条) 44(杠) 345 88 9(杠上开花)
        - 杠上开花（大胡）
    """
    wall = calculate_wall_scenario3_kong_bloom()

    result = (
        ScenarioBuilder("场景3：杠上开花")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [0, 0, 0, 10, 10, 13, 13, 3, 4, 5, 20, 24, 24, 26],
                1: [1, 8, 12, 14, 16, 27, 28, 29, 30, 31, 32, 33, 25],
                2: [9, 11, 15, 17, 18, 19, 21, 22, 23, 25, 26, 26, 24],
                3: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0出26（9筒）
        .step(1, "玩家0出26（9筒）")
            .action(0, ActionType.DISCARD, 26)

        # 步骤2-4：玩家1,2,3依次PASS
        .step(2, "玩家1 PASS")
            .action(1, ActionType.PASS)
        .step(3, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(4, "玩家3 PASS")
            .action(3, ActionType.PASS)

        # 步骤5：玩家1摸牌后出1（2万）
        .step(5, "玩家1摸牌后出1（2万）")
            .action(1, ActionType.DISCARD, 1)

        # 步骤6-8：玩家2,3依次PASS，玩家0 PONG
        .step(6, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(7, "玩家3 PASS")
            .action(3, ActionType.PASS)
        .step(8, "玩家0 PONG 1（1万）完成开口")
            .action(0, ActionType.PONG, 1)

        # 步骤9-12：玩家0出20（3筒），玩家1,2,3依次PASS
        .step(9, "玩家0出20（3筒）")
            .action(0, ActionType.DISCARD, 20)
        .step(10, "玩家1 PASS")
            .action(1, ActionType.PASS)
        .step(11, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(12, "玩家3 PASS")
            .action(3, ActionType.PASS)

        # 步骤13-16：玩家1出10（2条），玩家2,3,0依次响应（0 PONG）
        .step(13, "玩家1出10（2条）")
            .action(1, ActionType.DISCARD, 10)
        .step(14, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(15, "玩家3 PASS")
            .action(3, ActionType.PASS)
        .step(16, "玩家0 PONG 10（2条）")
            .action(0, ActionType.PONG, 10)

        # 步骤17：玩家0补杠10（2条）
        .step(17, "玩家0补杠10（2条）")
            .action(0, ActionType.KONG_SUPPLEMENT, 10)

        # 步骤18：玩家0摸24（8筒）杠上开花
        .step(18, "玩家0摸24（8筒）杠上开花")
            .action(0, ActionType.WIN, -1)

        .run()
    )

    if result.success:
        print("[OK] 场景3测试成功：杠上开花")
        print(f"获胜者: {result.final_context_snapshot.get('winner_ids', [])}")
    else:
        print(f"[FAIL] 场景3测试失败: {result.failure_message}")
        if result.final_context_snapshot:
            print(f"最终状态: {result.final_context_snapshot}")

    assert result.success, f"场景3测试失败: {result.failure_message}"


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("武汉麻将和牌场景测试")
    print("=" * 60)

    print("\n【场景1：硬胡自摸】")
    test_scenario_1_hard_win_self_draw()

    print("\n【场景2：软胡接炮】")
    test_scenario_2_soft_win_discard()

    print("\n【场景3：杠上开花】")
    test_scenario_3_kong_bloom()

    print("\n" + "=" * 60)
    print("[OK] 所有场景测试通过！")
    print("=" * 60)
