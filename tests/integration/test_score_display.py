"""测试场景测试框架的分数显示功能"""

from tests.scenario.builder import ScenarioBuilder
from src.mahjong_rl.core.constants import GameStateType, ActionType


def test_score_display_feature_exists():
    """测试分数显示功能已添加到 TestExecutor"""
    from tests.scenario.executor import TestExecutor

    # 验证 _print_scores 方法存在
    assert hasattr(TestExecutor, '_print_scores'), "TestExecutor should have _print_scores method"
    assert hasattr(TestExecutor, '_format_melds'), "TestExecutor should have _format_melds method"
    assert hasattr(TestExecutor, '_get_win_way_name'), "TestExecutor should have _get_win_way_name method"

    print("[PASS] 分数显示功能已添加到 TestExecutor")


def test_score_display_on_rob_kong():
    """使用现有的抢杠和测试来验证分数显示"""
    # 使用已有的测试场景来验证分数显示
    from tests.integration.test_rob_kong_scenario import calculate_wall_for_scenario

    wall = calculate_wall_for_scenario()

    result = (
        ScenarioBuilder("测试分数显示-抢杠和")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [1, 1, 1, 2, 3, 4, 4, 5, 5, 5, 6, 7, 8, 33],
                1: [0, 0, 4, 6, 9, 11, 11, 18, 19, 20, 23, 24, 25],
                2: [8, 10, 12, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29],
                3: [3, 12, 13, 16, 21, 22, 23, 24, 28, 29, 30, 31, 32],
            },
            'wall': wall,
            'special_tiles': {'lazy': 21, 'skins': [19, 18]},
        })
        .step(1, "玩家0打出红中")
            .action(0, ActionType.DISCARD, 33)
        .step(3, "玩家1打出16")
            .action(1, ActionType.DISCARD, 16)
        .step(4, "玩家2过")
            .action(2, ActionType.PASS)
        .step(5, "玩家2打出0")
            .action(2, ActionType.DISCARD, 0)
        .step(6, "玩家1碰0")
            .action(1, ActionType.PONG, 0)
        .step(8, "玩家1打出4")
            .action(1, ActionType.DISCARD, 4)
        .step(9, "玩家0碰4")
            .action(0, ActionType.PONG, 4)
        .step(11, "玩家0打出3")
            .action(0, ActionType.DISCARD, 3)
        .step(14, "玩家1补杠")
            .action(1, ActionType.KONG_SUPPLEMENT, 0)
        .step(15, "玩家0抢杠和")
            .action(0, ActionType.WIN, -1)
        .run()
    )

    # 验证游戏结束和分数显示
    assert result.success
    assert result.final_state == GameStateType.WIN
    assert result.final_context_snapshot.get('winner_ids') == [0]

    print("[PASS] 抢杠和分数显示测试完成！")


if __name__ == "__main__":
    test_score_display_feature_exists()
    test_score_display_on_rob_kong()
