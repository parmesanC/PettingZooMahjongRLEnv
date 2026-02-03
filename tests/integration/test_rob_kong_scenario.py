"""测试抢杠和场景

玩家0是庄家，测试补杠时玩家0能否抢杠和。

初始手牌：
    player_0 = [1, 1, 1, 2, 3, 4, 4, 5, 5, 5, 6, 7, 8, 33]  # 庄家14张
    player_1 = [0, 0, 4, 6, 9, 11, 11, 18, 19, 20, 23, 24, 25]  # 13张
    player_2 = [8, 10, 12, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29]  # 13张
    player_3 = [3, 12, 13, 16, 21, 22, 23, 24, 28, 29, 30, 31, 32]  # 13张

游戏流程：
1. 玩家0打出33（红中）
2. 玩家1摸16，打出16
3. 玩家2摸0，打出0
4. 玩家1碰0
5. 玩家1打出4
6. 玩家0碰4
7. 玩家0打出3
8. 玩家1摸0，进行补杠
9. 进入 WAIT_ROB_KONG 状态
10. 测试玩家0能否抢杠和
"""

import pytest
from tests.scenario.builder import ScenarioBuilder
from tests.scenario.validators import hand_count_equals, state_is
from src.mahjong_rl.core.constants import GameStateType, ActionType


def calculate_wall_for_scenario():
    """根据手牌和游戏流程计算牌墙"""
    # 初始手牌
    p0 = [1, 1, 1, 2, 3, 4, 4, 5, 5, 5, 6, 7, 8, 33]  # 14张
    p1 = [0, 0, 4, 6, 9, 11, 11, 18, 19, 20, 23, 24, 25]  # 13张
    p2 = [8, 10, 12, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29]  # 13张
    p3 = [3, 12, 13, 16, 21, 22, 23, 24, 28, 29, 30, 31, 32]  # 13张

    # 游戏流程中被摸走的牌（按顺序）
    drawn_tiles = [16, 0, 0]

    # 计算牌墙：136张 - 所有手牌 - 摸走的牌
    all_used = p0 + p1 + p2 + p3 + drawn_tiles

    # 创建完整牌墙并移除已使用的牌
    unused_wall = [i for i in range(34)] * 4

    for tile in all_used:
        unused_wall.remove(tile)

    wall = drawn_tiles + unused_wall

    return wall


def test_rob_kong_scenario():
    """测试完整的抢杠和场景"""
    wall = calculate_wall_for_scenario()

    result = (
        ScenarioBuilder("抢杠和场景测试")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [1, 1, 1, 2, 3, 4, 4, 5, 5, 5, 6, 7, 8, 33],  # 庄家14张（33已在手牌中）
                1: [0, 0, 4, 6, 9, 11, 11, 18, 19, 20, 23, 24, 25],
                2: [8, 10, 12, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29],
                3: [3, 12, 13, 16, 21, 22, 23, 24, 28, 29, 30, 31, 32],
            },
            'wall': wall,
            'special_tiles': {'lazy': 21, 'skins': [19, 18]},
            # 不设置 last_drawn_tile，因为33已经在手牌中了
            # 如果设置，会导致检测动作时重复添加，手牌变成15张
        })
        # 步骤1：玩家0打出33（红中杠 KONG_RED）
        .step(1, "玩家0打出33")
            .action(0, ActionType.DISCARD, 33)  # KONG_RED 参数通常为0
            # .expect_state(GameStateType.PLAYER_DECISION)

        # # 步骤2：杠后补牌（玩家1摸16）
        # .step(2, "杠后补牌")
        #     .auto_advance()
        #     .verify("牌墙减少", lambda ctx: len(ctx.wall) == len(wall) - 1)

        # 步骤3：玩家1打出16
        .step(3, "玩家1打出16")
            .action(1, ActionType.DISCARD, 16)
            # .expect_state(GameStateType.PLAYER_DECISION)

        # # 步骤4：所有人过
        .step(4, "玩家2过")
            .action(2, ActionType.PASS)
        #     .expect_state(GameStateType.PLAYER_DECISION)

        # 步骤5：玩家2摸0后打出0
        .step(5, "玩家2打出0")
            .action(2, ActionType.DISCARD, 0)
            # .expect_state(GameStateType.WAITING_RESPONSE)

        # 步骤6：玩家1碰0
        .step(6, "玩家1碰0")
            .action(1, ActionType.PONG, 0)
            # .expect_state(GameStateType.MELD_DECISION)

        # # 步骤7：自动处理碰牌
        # .step(7, "碰牌后处理")
        #     .auto_advance()
        #     .expect_state(GameStateType.MELD_DECISION)

        # 步骤8：玩家1打出4
        .step(8, "玩家1打出4")
            .action(1, ActionType.DISCARD, 4)
            # .expect_state(GameStateType.WAITING_RESPONSE)

        # 步骤9：玩家0碰4
        .step(9, "玩家0碰4")
            .action(0, ActionType.PONG, 4)
            # .expect_state(GameStateType.PLAYER_DECISION)

        # # 步骤10：自动处理碰牌
        # .step(10, "碰牌后处理")
        #     .auto_advance()
        #     .expect_state(GameStateType.PLAYER_DECISION)

        # 步骤11：玩家0打出3
        .step(11, "玩家0打出3")
            .action(0, ActionType.DISCARD, 3)
            # .expect_state(GameStateType.PLAYER_DECISION)

        # # 步骤12：所有人过
        # .step(12, "所有人过")
        #     .auto_advance()
        #     .expect_state(GameStateType.PLAYER_DECISION)

        # 步骤14：玩家1补杠（触发抢杠检测）
        .step(14, "玩家1补杠0")
            .action(1, ActionType.KONG_SUPPLEMENT, 0)
            # .expect_state(GameStateType.WAIT_ROB_KONG)

        # 步骤15：玩家0抢杠和
        .step(15, "玩家0抢杠和")
            .action(0, ActionType.WIN, -1)
            # .expect_state(GameStateType.WIN)

        .run()
    )

    if result.success:
        print("✅ 抢杠和测试成功！")
        print(f"获胜者: {result.final_context_snapshot.get('winner_ids', [])}")
    else:
        print(f"❌ 测试失败: {result.failure_message}")
        if result.final_context_snapshot:
            print(f"最终状态: {result.final_context_snapshot}")

    assert result.success, f"测试失败: {result.failure_message}"


if __name__ == "__main__":
    test_rob_kong_scenario()
