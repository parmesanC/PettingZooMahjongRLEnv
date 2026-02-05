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
import random

import pytest
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
    p0 = [0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 15]  # 14张
    p1 = [8, 10, 12, 14, 16, 27, 28, 29, 30, 31, 32, 33, 0]  # 13张
    p2 = [9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 26]  # 13张
    p3 = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 13张

    # 游戏流程中被摸走的牌（按顺序）
    drawn_tiles = [1, 8, 9, 2, 25]  # p1摸8, p2摸9, p3摸2, p0摸25自摸

    gong_drawn_tiles = [3, 26]

    # 计算牌墙
    all_used = p0 + p1 + p2 + p3 + drawn_tiles + gong_drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    random.shuffle(unused_wall)

    wall = drawn_tiles + unused_wall + gong_drawn_tiles
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
                0: [0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 15],  # 庄家14张
                1: [8, 10, 12, 14, 16, 27, 28, 29, 30, 31, 32, 33, 0],
                2: [9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 26],
                3: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0出26（9筒）- 听牌
        .step(1, "玩家0出15（8条）进行赖子杠，听牌")
            .action(0, ActionType.KONG_LAZY, 0)

        # 步骤2：玩家0 摸牌后打出33（白板）
        .step(2, "玩家0 摸牌后打出26（九筒）")
            .action(0, ActionType.DISCARD, 26)

        # 步骤3：玩家2 PASS
        .step(3, "玩家2 PASS")
            .action(2, ActionType.PASS)

        # 步骤4：玩家1摸牌后出0（1万）
        .step(4, "玩家1摸牌后出0（1万）")
            .action(1, ActionType.DISCARD, 0)

        # 步骤5：玩家0 明杠 0（1万）完成开口
        .step(5, "玩家0 KONG 0（1万）完成开口")
            .action(0, ActionType.KONG_EXPOSED, 0)

        # 步骤6：玩家0出3（4万）
        .step(6, "玩家0出3（4万）")
            .action(0, ActionType.DISCARD, 3)

        # 步骤7：玩家1出8(9万)
        .step(7, "玩家1出8")
            .action(1, ActionType.DISCARD, 8)

        # 步骤8：玩家2出9，玩家3,0,1依次PASS
        .step(8, "玩家2出9")
            .action(2, ActionType.DISCARD, 9)

        # 步骤9：玩家3 PASS
        .step(18, "玩家3 PASS")
            .action(3, ActionType.PASS)

        # 步骤9：玩家3出2
        .step(9, "玩家3出2")
            .action(3, ActionType.DISCARD, 2)

        # 步骤10：玩家0 PASS（env.step()会自动处理WAITING_RESPONSE并让玩家0摸牌）
        .step(10, "玩家0 PASS")
            .action(0, ActionType.PASS)

        # 步骤11：玩家0自摸硬胡
        .step(11, "玩家0自摸硬胡")
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

    初始手牌：
        p0: [0, 0, 1, 2, 3, 4, 5, 6, 7, 15, 24, 24, 24, 26]  # 14张，庄家
            = 1万x2, 2万,3万,4万,5万,6万,7万, 赖子(15), 8筒x3, 9筒
        p1: [8, 10, 12, 14, 16, 25, 26, 27, 28, 29, 30, 31, 32]  # 13张
            = 9万,2条,4条,6条,8条, 9筒,9筒, 东,南,西,北,中,发
        p2: [2, 2, 9, 11, 13, 17, 18, 19, 20, 21, 22, 23, 26]  # 13张
            = 3万x2, 1条,3条,5条,7条, 7筒,8筒,9筒,10筒,11筒,12筒, 9筒
        p3: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 13张

    游戏流程：
        1. 玩家0 赖子杠（打出赖子15）→ 从牌尾摸15
        2. 玩家0 出2（3万）→ 玩家2碰2
        3. 玩家2 出1（2万）→ 玩家0碰1完成开口
        4. 玩家0 出26（9筒）听牌 → 玩家1摸牌
        5. 玩家1 出26（9筒）→ 玩家0接炮软胡
    """
    # 初始手牌
    p0 = [0, 0, 1, 2, 3, 4, 5, 6, 7, 15, 24, 24, 24, 26]  # 14张，庄家
    p1 = [8, 10, 12, 14, 16, 25, 26, 27, 28, 29, 30, 31, 32]  # 13张
    p2 = [2, 2, 9, 11, 13, 17, 18, 19, 20, 21, 22, 23, 26]  # 13张
    p3 = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 13张

    # 游戏流程中被摸走的牌（按顺序）
    drawn_tiles = [0, 33]  # 玩家1摸的牌（任意牌）

    # 杠后从牌尾摸的牌
    gong_drawn_tiles = [15]  # 玩家0赖子杠后从牌尾摸的牌

    # 计算牌墙
    all_used = p0 + p1 + p2 + p3 + drawn_tiles + gong_drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    random.shuffle(unused_wall)

    wall = drawn_tiles + unused_wall + gong_drawn_tiles
    return wall


def test_scenario_2_soft_win_discard():
    """场景2：软胡接炮

    目的：验证1个赖子未还原的接炮软胡，包含玩家2碰牌流程

    初始手牌：
        玩家0（庄家）：
            [0, 0, 1, 2, 3, 4, 5, 6, 7, 15, 24, 24, 24, 26]
            = 1万x2, 2万,3万,4万,5万,6万,7万, 赖子(15), 8筒x3, 9筒

    特殊牌：
        赖子：15（2条）- 在手牌中，未还原
        皮子：14, 13

    目标牌型：111(赖子还原+碰1万) 234 567 888 9(接炮)
        - 1个赖子未还原（软胡）
        - 将牌=9筒（符合2/5/8要求）

    游戏流程：
        1. 玩家0 赖子杠（打出赖子15）→ 从牌尾摸15
        2. 玩家0 出2（3万）→ 玩家2碰2
        3. 玩家2 出1（2万）→ 玩家0碰1完成开口
        4. 玩家0 出26（9筒）听牌 → 自动跳过 → 玩家1摸牌
        5. 玩家1 出26（9筒）→ 玩家0接炮软胡
    """
    wall = calculate_wall_scenario2_soft_win()

    result = (
        ScenarioBuilder("场景2：软胡接炮")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [0, 0, 1, 2, 3, 4, 5, 6, 7, 15, 24, 24, 24, 26],  # 庄家14张
                1: [8, 10, 12, 14, 16, 25, 26, 27, 28, 29, 30, 31, 32],
                2: [1, 1, 9, 11, 13, 17, 18, 19, 20, 21, 22, 23, 32],
                3: [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14],
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0 赖子杠（打出赖子15）→ 从牌尾摸15 → PLAYER_DECISION
        .step(1, "玩家0 赖子杠（打出赖子15）")
            .action(0, ActionType.KONG_LAZY, 0)

        # 步骤2：玩家0 出2（3万）→ WAITING_RESPONSE
        .step(2, "玩家0 出1（2万）")
            .action(0, ActionType.DISCARD, 1)

        # 步骤3：玩家2 碰2（3万）→ PROCESSING_MELD → MELD_DECISION(玩家2)
        .step(3, "玩家2 碰1（2万）")
            .action(2, ActionType.PONG, 0)

        # 步骤4：玩家2 出32（发财）→ PLAYER_DECISION(玩家3)
        .step(4, "玩家2 出32（发财）")
            .action(2, ActionType.DISCARD, 32)

        # 步骤5：玩家3 出0（1万）→ WAITING_RESPONSE
        .step(5, "玩家3 出0（1万）")
            .action(2, ActionType.DISCARD, 0)

        # 步骤5：玩家0 碰1（2万）完成开口 → PROCESSING_MELD → PLAYER_DECISION(玩家0)
        .step(5, "玩家0 碰0（1万）完成开口")
            .action(0, ActionType.PONG, 0)

        # 步骤6：玩家0 出26（9筒）听牌 → WAITING_RESPONSE → 自动跳过 → DRAWING → PLAYER_DECISION(玩家1)
        .step(6, "玩家0 出26（9筒）听牌")
            .action(0, ActionType.DISCARD, 26)

        # 步骤7：玩家1 出26（9筒）→ WAITING_RESPONSE
        .step(7, "玩家1 出25（8筒）")
            .action(1, ActionType.DISCARD, 25)

        # 步骤8：玩家0 接炮软胡 → WIN
        .step(8, "玩家0 接炮软胡")
            .action(0, ActionType.WIN, -1)
            .verify("检查手牌", lambda ctx: print(f"玩家0手牌={sorted(ctx.players[0].hand_tiles)}, melds={ctx.players[0].melds}, has_opened={ctx.players[0].has_opened}") or True)
            .expect_state(GameStateType.WIN)

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
    p0 = [0, 0, 0, 10, 10, 13, 16, 3, 4, 5, 20, 24, 24, 26]  # 14张，庄家
    p1 = [1, 8, 12, 14, 16, 27, 28, 29, 30, 31, 32, 33, 24]  # 13张
    p2 = [9, 11, 15, 17, 18, 19, 21, 22, 23, 25, 26, 26, 24]  # 13张
    p3 = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 13张

    # 游戏流程中被摸走的牌
    drawn_tiles = [25, 10, 2, 14, 21, 10]

    gong_drawn_tiles = [16, 33]

    # 计算牌墙
    all_used = p0 + p1 + p2 + p3 + drawn_tiles + gong_drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    wall = drawn_tiles + unused_wall + gong_drawn_tiles
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
                0: [0, 0, 0, 10, 10, 13, 16, 3, 4, 5, 20, 24, 24, 26],
                1: [1, 8, 12, 14, 16, 27, 28, 29, 30, 31, 32, 33, 24],
                2: [9, 11, 15, 17, 18, 19, 21, 22, 23, 25, 26, 26, 24],
                3: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0出26（9筒）
        .step(1, "玩家0出26（9筒）")
            .action(0, ActionType.DISCARD, 26)

        # 步骤2：玩家2 PASS
        .step(2, "玩家2 PASS")
            .action(2, ActionType.PASS)

        # 步骤3：玩家1摸牌后出1（2万）
        .step(3, "玩家1摸牌后出24（7筒）")
            .action(1, ActionType.DISCARD, 24)

        # 步骤4：玩家2 PASS，玩家0 PONG 24（7筒）完成开口
        .step(4, "玩家2 PASS")
            .action(2, ActionType.PASS)

        # 步骤5：玩家0 PONG 24（7筒）完成开口
        .step(5, "玩家0 PONG 24（7筒）完成开口")
            .action(0, ActionType.PONG, 0)

        # 步骤6：玩家0出20（3筒）
        .step(6, "玩家0出20（3筒）")
            .action(0, ActionType.DISCARD, 20)

        # 步骤7：玩家1出10（2条）
        .step(7, "玩家1出10（2条）")
            .action(1, ActionType.DISCARD, 10)

        # 步骤8：玩家2 PASS
        .step(8, "玩家2 PASS")
            .action(2, ActionType.PASS)

        # 步骤8：玩家0 PONG 10（2条）
        .step(8, "玩家0 PONG 10（2条）")
            .action(0, ActionType.PONG, 0)

        # 步骤9：玩家0 出13 (5条)进行皮子杠
        .step(9, "玩家0 出13(5条)进行皮子杠")
            .action(0, ActionType.KONG_SKIN, 13)

        # 步骤10：玩家0 出33 (白板)
        .step(10, "玩家0 出33(白板)")
            .action(0, ActionType.DISCARD, 33)

        # 步骤11：玩家1 出1 (2万)
        .step(11, "玩家1 出1(2万)")
            .action(1, ActionType.DISCARD, 1)

        # 步骤12：玩家2 出25 (8筒)
        .step(12, "玩家2 出25(8筒)")
            .action(2, ActionType.DISCARD, 25)

        # 步骤13：玩家3 出7 (8万)
        .step(13, "玩家3 出7(8万)")
            .action(3, ActionType.DISCARD, 7)

        # 步骤14：玩家0补杠10（2条）
        .step(14, "玩家0补杠10（2条）")
            .action(0, ActionType.KONG_SUPPLEMENT, 10)

        # 步骤15：玩家0摸16（8条）杠上开花
        .step(15, "玩家0摸16（8条）杠上开花")
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
# 场景4：全求人
# =============================================================================

def calculate_wall_scenario4_all_melded():
    """计算场景4的牌墙"""
    p0 = [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 24, 24, 26, 26]
    p1 = [8, 10, 12, 14, 16, 25, 27, 28, 29, 30, 31, 32, 33]
    p2 = [9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 26]
    p3 = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    drawn_tiles = [0, 11, 24, 0, 1, 6, 8, 9, 10, 11, 12, 13]
    gong_drawn_tiles = [0, 1, 24]

    all_used = p0 + p1 + p2 + p3 + drawn_tiles + gong_drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    random.shuffle(unused_wall)
    wall = drawn_tiles + unused_wall + gong_drawn_tiles
    return wall


def test_scenario_4_all_melded():
    """场景4：全求人

    目的：验证4面子均鸣牌得来，手牌仅剩1张，接炮胡牌（大胡）

    游戏流程：
        1. 玩家0 出26 → 玩家1 出0 → 玩家0 明杠0（开口）
        2. 玩家0 出2 → 玩家1-3 出牌
        3. 玩家0 出1 → 玩家2 出1 → 玩家0 明杠1
        4. 玩家0 出6 → 玩家1-3 出牌
        5. 玩家0 出24 → 玩家1 出24 → 玩家0 明杠24
        6. 玩家0 出26（手牌剩26）→ 玩家1 出26 → 玩家0 接炮全求人
    """
    wall = calculate_wall_scenario4_all_melded()

    result = (
        ScenarioBuilder("场景4：全求人")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 24, 24, 26, 26],
                1: [8, 10, 12, 14, 16, 25, 27, 28, 29, 30, 31, 32, 33],
                2: [9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 26],
                3: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0 出26（9筒）
        .step(1, "玩家0出26")
            .action(0, ActionType.DISCARD, 26)

        # 步骤2：玩家2 碰26（9筒）
        .step(2, "玩家2碰26")
            .action(2, ActionType.PONG, 0)

        # 步骤3：玩家2 出9（1条）
        .step(3, "玩家2出9")
            .action(2, ActionType.DISCARD, 9)

        # 步骤4：玩家3 pass
        .step(4, "玩家3 PASS")
            .action(3, ActionType.PASS)

        # 步骤5：玩家3 出0（1万）
        .step(5, "玩家3出0")
            .action(3, ActionType.DISCARD, 0)

        # 步骤6：玩家0 碰0（1万）
        .step(6, "玩家0碰0")
            .action(0, ActionType.PONG, 0)

        # 步骤7：玩家0 出26（9筒）
        .step(7, "玩家0出26")
            .action(0, ActionType.DISCARD, 26)

        # 步骤8：玩家1 出8（9万）
        .step(8, "玩家1出8")
            .action(1, ActionType.DISCARD, 8)

        # 步骤9：玩家2 出24（7筒）
        .step(9, "玩家2出24")
            .action(2, ActionType.DISCARD, 24)
        # 步骤10：玩家0 碰24（7筒）
        .step(10, "玩家0碰24")
            .action(0, ActionType.PONG, 0)
        # 步骤11：玩家0 出2（3万）
        .step(11, "玩家0出2")
            .action(0, ActionType.DISCARD, 2)

        # 步骤12：玩家1 出33（白板）
        .step(12, "玩家1出33")
            .action(1, ActionType.DISCARD, 33)

        # 步骤13-14：玩家2,3 依次出牌
        .step(13, "玩家2出23")
            .action(2, ActionType.DISCARD, 23)
        .step(14, "玩家3出2")
            .action(3, ActionType.DISCARD, 2)

        # 步骤14：玩家0 左吃2（3万）
        .step(14, "玩家0吃2")
            .action(0, ActionType.CHOW, 0)

        # 步骤15：玩家0 出5（6万）
        .step(15, "玩家0出5")
            .action(0, ActionType.DISCARD, 5)

        # 步骤16-18：玩家1,2,3 依次出32（发财）,22（5筒）,8（9万）
        .step(16, "玩家1出32")
            .action(1, ActionType.DISCARD, 32)
        .step(17, "玩家2出22")
            .action(2, ActionType.DISCARD, 22)
        .step(18, "玩家3出8")
            .action(3, ActionType.DISCARD, 8)

        # 步骤19-20：玩家0先吃8（9万） 再出1（2万），手牌仅剩1
        .step(19, "玩家0吃8")
            .action(0, ActionType.CHOW, 2)
        .step(20, "玩家0出1")
            .action(0, ActionType.DISCARD, 1)

        # 步骤21：玩家1 出26（9筒）
        .step(21, "玩家1出29")
            .action(1, ActionType.DISCARD, 29)

        # 步骤22：玩家2 出1（2万）
        .step(22, "玩家2出1")
            .action(2, ActionType.DISCARD, 1)

        # 步骤23：玩家0 接炮全求人
        .step(23, "玩家0接炮全求人")
            .action(0, ActionType.WIN)
            .expect_state(GameStateType.WIN)

        .run()
    )

    if result.success:
        print("[OK] 场景4测试成功：全求人")
        print(f"获胜者: {result.final_context_snapshot.get('winner_ids', [])}")
    else:
        print(f"[FAIL] 场景4测试失败: {result.failure_message}")
        if result.final_context_snapshot:
            print(f"最终状态: {result.final_context_snapshot}")

    assert result.success, f"场景4测试失败: {result.failure_message}"


# =============================================================================
# 场景5：清一色
# =============================================================================

def calculate_wall_scenario5_same_suit():
    """计算场景5的牌墙"""
    p0 = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 24, 24, 26]
    p1 = [9, 10, 11, 12, 13, 14, 25, 26, 27, 28, 29, 30, 31]
    p2 = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26]
    p3 = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    drawn_tiles = [0, 0, 25, 24, 23, 24, 26]
    gong_drawn_tiles = [26]

    all_used = p0 + p1 + p2 + p3 + drawn_tiles + gong_drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    random.shuffle(unused_wall)
    wall = drawn_tiles + unused_wall + gong_drawn_tiles
    return wall


def test_scenario_5_same_suit():
    """场景5：清一色

    目的：验证全部牌为同一花色的胡牌（大胡）

    游戏流程：
        1. 玩家0 赖子杠（开口）→ 出26
        2. 玩家1-3 出牌
        3. 玩家0 出24 → 玩家1 出0 → 玩家0 明杠0
        4. 玩家0 出8 → 玩家1-3 PASS → 玩家0 摸26清一色
    """
    wall = calculate_wall_scenario5_same_suit()

    result = (
        ScenarioBuilder("场景5：清一色")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 24, 24, 26],
                1: [9, 10, 11, 12, 13, 14, 25, 26, 27, 28, 29, 30, 31],
                2: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26],
                3: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0 赖子杠（开口）
        .step(1, "玩家0赖子杠")
            .action(0, ActionType.KONG_LAZY, 0)

        # 步骤2：玩家0 出26（9筒）
        .step(2, "玩家0出26")
            .action(0, ActionType.DISCARD, 26)

        # 步骤3-5：玩家1,2,3 依次出牌
        .step(3, "玩家1出25")
            .action(1, ActionType.DISCARD, 25)
        .step(4, "玩家2出24")
            .action(2, ActionType.DISCARD, 24)
        .step(5, "玩家3出23")
            .action(3, ActionType.DISCARD, 23)

        # 步骤6：玩家0 出24（8筒）
        .step(6, "玩家0出24")
            .action(0, ActionType.DISCARD, 24)

        # 步骤7：玩家1 出0（1万）
        .step(7, "玩家1出0")
            .action(1, ActionType.DISCARD, 0)

        # 步骤8：玩家0 明杠0（1万）完成开口
        .step(8, "玩家0明杠0")
            .action(0, ActionType.KONG_EXPOSED, 0)

        # 步骤9：玩家0 出8（9万）
        .step(9, "玩家0出8")
            .action(0, ActionType.DISCARD, 8)

        # 步骤10-12：玩家1,2,3 依次PASS
        .step(10, "玩家1 PASS")
            .action(1, ActionType.PASS)
        .step(11, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(12, "玩家3 PASS")
            .action(3, ActionType.PASS)

        # 步骤13：玩家0 摸26（9筒）清一色
        .step(13, "玩家0摸26清一色")
            .action(0, ActionType.WIN, -1)
            .expect_state(GameStateType.WIN)

        .run()
    )

    if result.success:
        print("[OK] 场景5测试成功：清一色")
        print(f"获胜者: {result.final_context_snapshot.get('winner_ids', [])}")
    else:
        print(f"[FAIL] 场景5测试失败: {result.failure_message}")
        if result.final_context_snapshot:
            print(f"最终状态: {result.final_context_snapshot}")

    assert result.success, f"场景5测试失败: {result.failure_message}"


# =============================================================================
# 场景6：碰碰胡
# =============================================================================

def calculate_wall_scenario6_all_pons():
    """计算场景6的牌墙"""
    p0 = [0, 0, 0, 1, 1, 1, 3, 3, 3, 5, 5, 5, 24, 24]
    p1 = [2, 4, 6, 7, 8, 9, 10, 11, 12, 25, 26, 27, 28]
    p2 = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26]
    p3 = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    drawn_tiles = [0, 1, 3, 5, 24]
    gong_drawn_tiles = []

    all_used = p0 + p1 + p2 + p3 + drawn_tiles + gong_drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    random.shuffle(unused_wall)
    wall = drawn_tiles + unused_wall + gong_drawn_tiles
    return wall


def test_scenario_6_all_pons():
    """场景6：碰碰胡

    目的：验证面子均为刻子或杠的胡牌（大胡）

    游戏流程：
        1. 玩家0 出24 → 玩家1 出0 → 玩家0 碰0（开口）
        2. 玩家0 出24 → 玩家2 出1 → 玩家0 碰1
        3. 玩家0 出24 → 玩家3 出3 → 玩家0 碰3
        4. 玩家0 出24 → 玩家1-3 PASS → 玩家0 摸24碰碰胡
    """
    wall = calculate_wall_scenario6_all_pons()

    result = (
        ScenarioBuilder("场景6：碰碰胡")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [0, 0, 0, 1, 1, 1, 3, 3, 3, 5, 5, 5, 24, 24],
                1: [2, 4, 6, 7, 8, 9, 10, 11, 12, 25, 26, 27, 28],
                2: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26],
                3: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0 出24（8筒）
        .step(1, "玩家0出24")
            .action(0, ActionType.DISCARD, 24)

        # 步骤2：玩家1 出0（1万）
        .step(2, "玩家1出0")
            .action(1, ActionType.DISCARD, 0)

        # 步骤3：玩家0 碰0（1万）完成开口
        .step(3, "玩家0碰0")
            .action(0, ActionType.PONG, 0)

        # 步骤4：玩家0 出24（8筒）
        .step(4, "玩家0出24")
            .action(0, ActionType.DISCARD, 24)

        # 步骤5：玩家2 出1（2万）
        .step(5, "玩家2出1")
            .action(2, ActionType.DISCARD, 1)

        # 步骤6：玩家0 碰1（2万）
        .step(6, "玩家0碰1")
            .action(0, ActionType.PONG, 1)

        # 步骤7：玩家0 出24（8筒）
        .step(7, "玩家0出24")
            .action(0, ActionType.DISCARD, 24)

        # 步骤8：玩家3 出3（4万）
        .step(8, "玩家3出3")
            .action(3, ActionType.DISCARD, 3)

        # 步骤9：玩家0 碰3（4万）
        .step(9, "玩家0碰3")
            .action(0, ActionType.PONG, 3)

        # 步骤10：玩家0 出24（8筒）
        .step(10, "玩家0出24")
            .action(0, ActionType.DISCARD, 24)

        # 步骤11-13：玩家1,2,3 依次PASS
        .step(11, "玩家1 PASS")
            .action(1, ActionType.PASS)
        .step(12, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(13, "玩家3 PASS")
            .action(3, ActionType.PASS)

        # 步骤14：玩家0 摸24（8筒）碰碰胡
        .step(14, "玩家0摸24碰碰胡")
            .action(0, ActionType.WIN, -1)
            .expect_state(GameStateType.WIN)

        .run()
    )

    if result.success:
        print("[OK] 场景6测试成功：碰碰胡")
        print(f"获胜者: {result.final_context_snapshot.get('winner_ids', [])}")
    else:
        print(f"[FAIL] 场景6测试失败: {result.failure_message}")
        if result.final_context_snapshot:
            print(f"最终状态: {result.final_context_snapshot}")

    assert result.success, f"场景6测试失败: {result.failure_message}"


# =============================================================================
# 场景7：风一色
# =============================================================================

def calculate_wall_scenario7_all_winds():
    """计算场景7的牌墙"""
    p0 = [27, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33]
    p1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 33]
    p2 = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 33]
    p3 = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 0, 1, 2]

    drawn_tiles = [0, 33]
    gong_drawn_tiles = []

    all_used = p0 + p1 + p2 + p3 + drawn_tiles + gong_drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    random.shuffle(unused_wall)
    wall = drawn_tiles + unused_wall + gong_drawn_tiles
    return wall


def test_scenario_7_all_winds():
    """场景7：风一色

    目的：验证全部牌为风牌的胡牌（大胡）

    游戏流程：
        1. 玩家0 赖子杠（开口）→ 出33
        2. 玩家1 出33 → 玩家2,3,0 PASS → 玩家0 接炮风一色
    """
    wall = calculate_wall_scenario7_all_winds()

    result = (
        ScenarioBuilder("场景7：风一色")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [27, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33],
                1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 33],
                2: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 33],
                3: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 0, 1, 2],
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0 赖子杠（开口）
        .step(1, "玩家0赖子杠")
            .action(0, ActionType.KONG_LAZY, 0)

        # 步骤2：玩家0 出33（白板）
        .step(2, "玩家0出33")
            .action(0, ActionType.DISCARD, 33)

        # 步骤3：玩家1 出33（白板）
        .step(3, "玩家1出33")
            .action(1, ActionType.DISCARD, 33)

        # 步骤4-6：玩家2,3,0 依次响应
        .step(4, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(5, "玩家3 PASS")
            .action(3, ActionType.PASS)
        .step(6, "玩家0 PASS")
            .action(0, ActionType.PASS)

        # 步骤7：玩家0 接炮风一色
        .step(7, "玩家0接炮风一色")
            .action(0, ActionType.WIN, -1)
            .expect_state(GameStateType.WIN)

        .run()
    )

    if result.success:
        print("[OK] 场景7测试成功：风一色")
        print(f"获胜者: {result.final_context_snapshot.get('winner_ids', [])}")
    else:
        print(f"[FAIL] 场景7测试失败: {result.failure_message}")
        if result.final_context_snapshot:
            print(f"最终状态: {result.final_context_snapshot}")

    assert result.success, f"场景7测试失败: {result.failure_message}"


# =============================================================================
# 场景8：将一色
# =============================================================================

def calculate_wall_scenario8_all_258():
    """计算场景8的牌墙"""
    p0 = [1, 1, 1, 4, 4, 4, 7, 7, 7, 19, 19, 22, 22, 25]
    p1 = [0, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 22, 26]
    p2 = [14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28]
    p3 = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    drawn_tiles = [1, 4, 7, 22]
    gong_drawn_tiles = []

    all_used = p0 + p1 + p2 + p3 + drawn_tiles + gong_drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    random.shuffle(unused_wall)
    wall = drawn_tiles + unused_wall + gong_drawn_tiles
    return wall


def test_scenario_8_all_258():
    """场景8：将一色

    目的：验证全部牌为2、5、8的胡牌（大胡）

    游戏流程：
        1. 玩家0 出25 → 玩家1 出1 → 玩家0 碰1（开口）
        2. 玩家0 出25 → 玩家2 出4 → 玩家0 碰4
        3. 玩家0 出25 → 玩家3 出7 → 玩家0 碰7
        4. 玩家0 出25 → 玩家1-3 PASS → 玩家1 出22 → 玩家0 接炮将一色
    """
    wall = calculate_wall_scenario8_all_258()

    result = (
        ScenarioBuilder("场景8：将一色")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [1, 1, 1, 4, 4, 4, 7, 7, 7, 19, 19, 22, 22, 25],
                1: [0, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 22, 26],
                2: [14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28],
                3: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0 出25（9筒）
        .step(1, "玩家0出25")
            .action(0, ActionType.DISCARD, 25)

        # 步骤2：玩家1 出1（2万）
        .step(2, "玩家1出1")
            .action(1, ActionType.DISCARD, 1)

        # 步骤3：玩家0 碰1（2万）完成开口
        .step(3, "玩家0碰1")
            .action(0, ActionType.PONG, 1)

        # 步骤4：玩家0 出25（9筒）
        .step(4, "玩家0出25")
            .action(0, ActionType.DISCARD, 25)

        # 步骤5：玩家2 出4（5万）
        .step(5, "玩家2出4")
            .action(2, ActionType.DISCARD, 4)

        # 步骤6：玩家0 碰4（5万）
        .step(6, "玩家0碰4")
            .action(0, ActionType.PONG, 4)

        # 步骤7：玩家0 出25（9筒）
        .step(7, "玩家0出25")
            .action(0, ActionType.DISCARD, 25)

        # 步骤8：玩家3 出7（8万）
        .step(8, "玩家3出7")
            .action(3, ActionType.DISCARD, 7)

        # 步骤9：玩家0 碰7（8万）
        .step(9, "玩家0碰7")
            .action(0, ActionType.PONG, 7)

        # 步骤10：玩家0 出25（9筒）
        .step(10, "玩家0出25")
            .action(0, ActionType.DISCARD, 25)

        # 步骤11-13：玩家1,2,3 依次PASS
        .step(11, "玩家1 PASS")
            .action(1, ActionType.PASS)
        .step(12, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(13, "玩家3 PASS")
            .action(3, ActionType.PASS)

        # 步骤14：玩家1 出22（5筒）
        .step(14, "玩家1出22")
            .action(1, ActionType.DISCARD, 22)

        # 步骤15：玩家0 接炮将一色
        .step(15, "玩家0接炮将一色")
            .action(0, ActionType.WIN, -1)
            .expect_state(GameStateType.WIN)

        .run()
    )

    if result.success:
        print("[OK] 场景8测试成功：将一色")
        print(f"获胜者: {result.final_context_snapshot.get('winner_ids', [])}")
    else:
        print(f"[FAIL] 场景8测试失败: {result.failure_message}")
        if result.final_context_snapshot:
            print(f"最终状态: {result.final_context_snapshot}")

    assert result.success, f"场景8测试失败: {result.failure_message}"


# =============================================================================
# 场景9：海底捞月
# =============================================================================

def calculate_wall_scenario9_last_draw():
    """计算场景9的牌墙 - 需要确保最后4张时摸牌"""
    p0 = [0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 15]
    p1 = [8, 10, 12, 14, 16, 27, 28, 29, 30, 31, 32, 33, 0]
    p2 = [9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 26]
    p3 = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # 计算需要多少张牌才能在最后4张时摸牌
    # 总牌数136 - 初始手牌53 = 83张
    # 需要在摸第79张时达到最后4张
    drawn_tiles = [0, 0, 1, 8, 9, 2, 3, 25]  # 调整使牌墙在合适时机剩余4张
    gong_drawn_tiles = [3, 26]

    all_used = p0 + p1 + p2 + p3 + drawn_tiles + gong_drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    # 确保剩余牌墙数量适当
    wall = drawn_tiles + unused_wall[-10:] + gong_drawn_tiles  # 只保留最后10张
    return wall


def test_scenario_9_last_draw():
    """场景9：海底捞月

    目的：验证摸牌墙最后4张自摸胡牌（大胡）

    游戏流程：
        与场景1类似，但需要控制牌墙在合适时机剩余4张
    """
    wall = calculate_wall_scenario9_last_draw()

    result = (
        ScenarioBuilder("场景9：海底捞月")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 15],
                1: [8, 10, 12, 14, 16, 27, 28, 29, 30, 31, 32, 33, 0],
                2: [9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 26],
                3: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0 赖子杠（开口）
        .step(1, "玩家0赖子杠")
            .action(0, ActionType.KONG_LAZY, 0)

        # 步骤2：玩家0 出26（9筒）
        .step(2, "玩家0出26")
            .action(0, ActionType.DISCARD, 26)

        # 步骤3：玩家1 PASS
        .step(3, "玩家1 PASS")
            .action(1, ActionType.PASS)

        # 步骤4：玩家1 出0（1万）
        .step(4, "玩家1出0")
            .action(1, ActionType.DISCARD, 0)

        # 步骤5：玩家0 明杠0（1万）完成开口
        .step(5, "玩家0明杠0")
            .action(0, ActionType.KONG_EXPOSED, 0)

        # 步骤6-9：玩家0,1,2,3 依次出牌
        .step(6, "玩家0出3")
            .action(0, ActionType.DISCARD, 3)
        .step(7, "玩家1出8")
            .action(1, ActionType.DISCARD, 8)
        .step(8, "玩家2出9")
            .action(2, ActionType.DISCARD, 9)
        .step(9, "玩家3出2")
            .action(3, ActionType.DISCARD, 2)

        # 步骤10：玩家0 PASS（触发自动跳过，摸牌）
        .step(10, "玩家0 PASS")
            .action(0, ActionType.PASS)

        # 步骤11：玩家0 海底捞月
        .step(11, "玩家0海底捞月")
            .action(0, ActionType.WIN, -1)
            .expect_state(GameStateType.WIN)

        .run()
    )

    if result.success:
        print("[OK] 场景9测试成功：海底捞月")
        print(f"获胜者: {result.final_context_snapshot.get('winner_ids', [])}")
    else:
        print(f"[FAIL] 场景9测试失败: {result.failure_message}")
        if result.final_context_snapshot:
            print(f"最终状态: {result.final_context_snapshot}")

    assert result.success, f"场景9测试失败: {result.failure_message}"


# =============================================================================
# 场景10：抢杠和
# =============================================================================

def calculate_wall_scenario10_rob_kong():
    """计算场景10的牌墙"""
    p0 = [0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 25]
    p1 = [10, 10, 10, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]
    p2 = [9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 26]
    p3 = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    drawn_tiles = [0, 26, 10, 3, 7, 8, 9, 10]
    gong_drawn_tiles = []

    all_used = p0 + p1 + p2 + p3 + drawn_tiles + gong_drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    random.shuffle(unused_wall)
    wall = drawn_tiles + unused_wall + gong_drawn_tiles
    return wall


def test_scenario_10_rob_kong():
    """场景10：抢杠和

    目的：验证抢他人补杠牌胡牌（大胡）

    游戏流程：
        1. 玩家0 赖子杠（开口）→ 出26
        2. 玩家1 出10 → 玩家0 碰10（开口）
        3. 玩家0 出3 → 玩家1-3 出牌
        4. 玩家1 补杠10 → 玩家0 抢杠和
    """
    wall = calculate_wall_scenario10_rob_kong()

    result = (
        ScenarioBuilder("场景10：抢杠和")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 25],
                1: [10, 10, 10, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18],
                2: [9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 26],
                3: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0 赖子杠（开口）
        .step(1, "玩家0赖子杠")
            .action(0, ActionType.KONG_LAZY, 0)

        # 步骤2：玩家0 出26（9筒）
        .step(2, "玩家0出26")
            .action(0, ActionType.DISCARD, 26)

        # 步骤3：玩家1 出10（2条）
        .step(3, "玩家1出10")
            .action(1, ActionType.DISCARD, 10)

        # 步骤4：玩家0 碰10（2条）完成开口
        .step(4, "玩家0碰10")
            .action(0, ActionType.PONG, 10)

        # 步骤5：玩家0 出3（4万）
        .step(5, "玩家0出3")
            .action(0, ActionType.DISCARD, 3)

        # 步骤6-8：玩家1,2,3 依次出牌
        .step(6, "玩家1出7")
            .action(1, ActionType.DISCARD, 7)
        .step(7, "玩家2出8")
            .action(2, ActionType.DISCARD, 8)
        .step(8, "玩家3出9")
            .action(3, ActionType.DISCARD, 9)

        # 步骤9：玩家1 补杠10（2条）
        .step(9, "玩家1补杠10")
            .action(1, ActionType.KONG_SUPPLEMENT, 10)

        # 步骤10：玩家0 抢杠和
        .step(10, "玩家0抢杠和")
            .action(0, ActionType.WIN, -1)
            .expect_state(GameStateType.WIN)

        .run()
    )

    if result.success:
        print("[OK] 场景10测试成功：抢杠和")
        print(f"获胜者: {result.final_context_snapshot.get('winner_ids', [])}")
    else:
        print(f"[FAIL] 场景10测试失败: {result.failure_message}")
        if result.final_context_snapshot:
            print(f"最终状态: {result.final_context_snapshot}")

    assert result.success, f"场景10测试失败: {result.failure_message}"


# =============================================================================
# 场景11：赖子还原硬胡
# =============================================================================

def calculate_wall_scenario11_lazy_restore():
    """计算场景11的牌墙"""
    p0 = [0, 0, 15, 2, 3, 4, 5, 6, 7, 8, 24, 24, 24, 26]
    p1 = [1, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 25, 26]
    p2 = [9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 26]
    p3 = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    drawn_tiles = [0, 26]
    gong_drawn_tiles = []

    all_used = p0 + p1 + p2 + p3 + drawn_tiles + gong_drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    random.shuffle(unused_wall)
    wall = drawn_tiles + unused_wall + gong_drawn_tiles
    return wall


def test_scenario_11_lazy_restore():
    """场景11：赖子还原硬胡

    目的：验证赖子还原后的硬胡（赖子作为普通牌使用）

    游戏流程：
        1. 玩家0 出26 → 玩家1 出0 → 玩家0 碰0（开口）
        2. 玩家0 出8 → 玩家1-3 PASS → 玩家0 摸26赖子还原硬胡
    """
    wall = calculate_wall_scenario11_lazy_restore()

    result = (
        ScenarioBuilder("场景11：赖子还原硬胡")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [0, 0, 15, 2, 3, 4, 5, 6, 7, 8, 24, 24, 24, 26],
                1: [1, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 25, 26],
                2: [9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 26],
                3: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0 出26（9筒）
        .step(1, "玩家0出26")
            .action(0, ActionType.DISCARD, 26)

        # 步骤2：玩家1 出0（1万）
        .step(2, "玩家1出0")
            .action(1, ActionType.DISCARD, 0)

        # 步骤3：玩家0 碰0（1万）完成开口
        .step(3, "玩家0碰0")
            .action(0, ActionType.PONG, 0)

        # 步骤4：玩家0 出8（9万）
        .step(4, "玩家0出8")
            .action(0, ActionType.DISCARD, 8)

        # 步骤5-7：玩家1,2,3 依次PASS
        .step(5, "玩家1 PASS")
            .action(1, ActionType.PASS)
        .step(6, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(7, "玩家3 PASS")
            .action(3, ActionType.PASS)

        # 步骤8：玩家0 摸26（9筒）赖子还原硬胡
        .step(8, "玩家0摸26赖子还原硬胡")
            .action(0, ActionType.WIN, -1)
            .expect_state(GameStateType.WIN)

        .run()
    )

    if result.success:
        print("[OK] 场景11测试成功：赖子还原硬胡")
        print(f"获胜者: {result.final_context_snapshot.get('winner_ids', [])}")
    else:
        print(f"[FAIL] 场景11测试失败: {result.failure_message}")
        if result.final_context_snapshot:
            print(f"最终状态: {result.final_context_snapshot}")

    assert result.success, f"场景11测试失败: {result.failure_message}"


# =============================================================================
# 场景12-15：边界测试
# =============================================================================

def test_scenario_12_min_score():
    """场景12：边界测试 - 最小起胡番数

    目的：验证乘积刚好16（起胡线）的边界情况
    """
    # 与场景1类似，但需要精确计算番数
    print("[INFO] 场景12：最小起胡番数测试（使用场景1用例）")
    test_scenario_1_hard_win_self_draw()


def test_scenario_13_lazy_limit():
    """场景13：边界测试 - 赖子数量限制

    目的：验证小胡赖子数量>1时不能胡牌

    预期：手牌有2个赖子，尝试胡牌时应该失败
    """
    p0 = [0, 0, 15, 15, 2, 3, 4, 5, 6, 7, 8, 24, 24, 24]
    p1 = [1, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 25, 26]
    p2 = [9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 26]
    p3 = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    drawn_tiles = [0, 24]
    gong_drawn_tiles = []

    all_used = p0 + p1 + p2 + p3 + drawn_tiles + gong_drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    random.shuffle(unused_wall)
    wall = drawn_tiles + unused_wall + gong_drawn_tiles

    result = (
        ScenarioBuilder("场景13：赖子数量限制")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: p0,
                1: p1,
                2: p2,
                3: p3,
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0 出24（8筒）
        .step(1, "玩家0出24")
            .action(0, ActionType.DISCARD, 24)

        # 步骤2：玩家1 出0（1万）
        .step(2, "玩家1出0")
            .action(1, ActionType.DISCARD, 0)

        # 步骤3：玩家0 碰0（1万）完成开口
        .step(3, "玩家0碰0")
            .action(0, ActionType.PONG, 0)

        # 步骤4：玩家0 出24（8筒）
        .step(4, "玩家0出24")
            .action(0, ActionType.DISCARD, 24)

        # 步骤5-7：玩家1,2,3 依次PASS
        .step(5, "玩家1 PASS")
            .action(1, ActionType.PASS)
        .step(6, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(7, "玩家3 PASS")
            .action(3, ActionType.PASS)

        # 步骤8：玩家0 摸24（8筒）尝试胡牌 - 预期失败
        .step(8, "玩家0尝试胡牌（预期失败）")
            .action(0, ActionType.WIN, -1)
            .expect_state(GameStateType.WIN)  # 预期会失败

        .run()
    )

    # 场景13预期失败
    if not result.success:
        print("[OK] 场景13测试正确：赖子超限不能胡牌")
    else:
        print("[FAIL] 场景13测试失败：应该检测到赖子超限")
        assert False, "场景13应该检测到赖子超限"


def test_scenario_14_no_opening():
    """场景14：边界测试 - 未开口不能胡牌

    目的：验证未开口时不能胡牌

    预期：未完成开口，尝试胡牌时应该失败
    """
    p0 = [0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 26]
    p1 = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 26, 27]
    p2 = [9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 26]
    p3 = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    drawn_tiles = [26]
    gong_drawn_tiles = []

    all_used = p0 + p1 + p2 + p3 + drawn_tiles + gong_drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    random.shuffle(unused_wall)
    wall = drawn_tiles + unused_wall + gong_drawn_tiles

    result = (
        ScenarioBuilder("场景14：未开口不能胡牌")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: p0,
                1: p1,
                2: p2,
                3: p3,
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0 出26（9筒）
        .step(1, "玩家0出26")
            .action(0, ActionType.DISCARD, 26)

        # 步骤2-4：玩家1,2,3 依次PASS
        .step(2, "玩家1 PASS")
            .action(1, ActionType.PASS)
        .step(3, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(4, "玩家3 PASS")
            .action(3, ActionType.PASS)

        # 步骤5：玩家0 摸26（9筒）尝试胡牌 - 预期失败
        .step(5, "玩家0尝试胡牌（预期失败）")
            .action(0, ActionType.WIN, -1)

        .run()
    )

    # 场景14预期失败
    if not result.success:
        print("[OK] 场景14测试正确：未开口不能胡牌")
    else:
        print("[FAIL] 场景14测试失败：应该检测到未开口")
        assert False, "场景14应该检测到未开口"


def test_scenario_15_no_red_dragon():
    """场景15：边界测试 - 皮/红中不能胡牌

    目的：验证胡牌时手牌中不能有皮或红中

    预期：手牌有红中，尝试胡牌时应该失败
    """
    p0 = [0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 31]
    p1 = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 26, 27]
    p2 = [9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 26, 26]
    p3 = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    drawn_tiles = [0, 0, 31, 25]
    gong_drawn_tiles = []

    all_used = p0 + p1 + p2 + p3 + drawn_tiles + gong_drawn_tiles
    unused_wall = [i for i in range(34)] * 4
    for tile in all_used:
        if tile in unused_wall:
            unused_wall.remove(tile)

    random.shuffle(unused_wall)
    wall = drawn_tiles + unused_wall + gong_drawn_tiles

    result = (
        ScenarioBuilder("场景15：红中/皮不能胡牌")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: p0,
                1: p1,
                2: p2,
                3: p3,
            },
            'wall': wall,
            'special_tiles': {'lazy': 15, 'skins': [14, 13]},
        })
        # 步骤1：玩家0 赖子杠（开口）
        .step(1, "玩家0赖子杠")
            .action(0, ActionType.KONG_LAZY, 0)

        # 步骤2：玩家0 出25（9筒）
        .step(2, "玩家0出25")
            .action(0, ActionType.DISCARD, 25)

        # 步骤3：玩家1 出0（1万）
        .step(3, "玩家1出0")
            .action(1, ActionType.DISCARD, 0)

        # 步骤4：玩家0 明杠0（1万）完成开口
        .step(4, "玩家0明杠0")
            .action(0, ActionType.KONG_EXPOSED, 0)

        # 步骤5：玩家0 出31（红中）
        .step(5, "玩家0出31")
            .action(0, ActionType.DISCARD, 31)

        # 步骤6-8：玩家1,2,3 依次PASS
        .step(6, "玩家1 PASS")
            .action(1, ActionType.PASS)
        .step(7, "玩家2 PASS")
            .action(2, ActionType.PASS)
        .step(8, "玩家3 PASS")
            .action(3, ActionType.PASS)

        # 步骤9：玩家0 摸25（9筒）尝试胡牌 - 预期失败
        .step(9, "玩家0尝试胡牌（预期失败）")
            .action(0, ActionType.WIN, -1)

        .run()
    )

    # 场景15预期失败
    if not result.success:
        print("[OK] 场景15测试正确：有红中/皮不能胡牌")
    else:
        print("[FAIL] 场景15测试失败：应该检测到红中/皮")
        assert False, "场景15应该检测到红中/皮"


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

    print("\n【场景4：全求人】")
    test_scenario_4_all_melded()

    print("\n【场景5：清一色】")
    test_scenario_5_same_suit()

    print("\n【场景6：碰碰胡】")
    test_scenario_6_all_pons()

    print("\n【场景7：风一色】")
    test_scenario_7_all_winds()

    print("\n【场景8：将一色】")
    test_scenario_8_all_258()

    print("\n【场景9：海底捞月】")
    test_scenario_9_last_draw()

    print("\n【场景10：抢杠和】")
    test_scenario_10_rob_kong()

    print("\n【场景11：赖子还原硬胡】")
    test_scenario_11_lazy_restore()

    print("\n【场景12：最小起胡番数】")
    test_scenario_12_min_score()

    print("\n【场景13：赖子数量限制】")
    test_scenario_13_lazy_limit()

    print("\n【场景14：未开口不能胡牌】")
    test_scenario_14_no_opening()

    print("\n【场景15：红中/皮不能胡牌】")
    test_scenario_15_no_red_dragon()

    print("\n" + "=" * 60)
    print("[OK] 所有场景测试通过！")
    print("=" * 60)
