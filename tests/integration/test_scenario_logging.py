"""测试场景测试的日志增强功能"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.scenario.builder import ScenarioBuilder
from src.mahjong_rl.core.constants import GameStateType, ActionType

def test_basic_scenario_with_logging():
    """测试基本场景的日志输出"""

    result = (
        ScenarioBuilder("日志测试")
        .with_initial_state({
            'dealer_idx': 0,
            'current_player_idx': 0,
            'hands': {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                1: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                2: [26, 27, 28, 29, 30, 31, 32, 33, 0, 1, 2, 3, 4],
                3: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            },
            'wall': list(range(18, 100)),
            'special_tiles': {'lazy': 8, 'skins': [7, 9]},
            'last_drawn_tile': 12,
        })
        .step(1, "玩家0打牌")
            .action(0, ActionType.DISCARD, 5)
            .expect_state(GameStateType.WAITING_RESPONSE)
        .run()
    )

    assert result.success, f"测试失败: {result.failure_message}"
    print("\n=== 测试通过 ===")

if __name__ == "__main__":
    test_basic_scenario_with_logging()
