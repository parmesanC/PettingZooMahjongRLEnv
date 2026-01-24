"""测试接炮胡牌检测功能"""

import pytest
from collections import deque

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import GameStateType, ActionType
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.action_validator import ActionValidator


class TestWinByDiscard:
    """测试接炮胡牌检测"""

    def test_win_by_discard_action_available(self):
        """测试：接炮胡牌动作应被正确检测"""
        # 创建游戏上下文
        context = GameContext(num_players=4)
        context.current_state = GameStateType.WAITING_RESPONSE
        context.discard_player = 1
        context.last_discarded_tile = 8  # 1万 (ID: 8)
        context.wall = deque([i for i in range(34) for _ in range(4)])

        # 设置玩家0的手牌（包括可以胡牌的牌型）
        # 给玩家0一个简单的一对+顺子组合，加上1万可以胡牌
        player0 = context.players[0]
        player0.hand_tiles = [0, 1, 2, 9, 10, 11, 18, 19, 20, 8, 8]  # 包含一对1万
        player0.melds = []
        player0.special_gangs = [0, 0, 0]

        # 设置其他玩家手牌（确保起胡番条件满足）
        for i in range(1, 4):
            player = context.players[i]
            player.hand_tiles = [3, 4, 5, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27]
            player.melds = []
            player.special_gangs = [0, 0, 0]

        # 创建 ActionValidator
        validator = ActionValidator(context)

        # 测试：玩家0可以接炮胡牌
        actions = validator.detect_available_actions_after_discard(
            player0, context.last_discarded_tile, context.discard_player
        )

        # 验证：WIN 动作在可用动作列表中
        action_types = [a.action_type for a in actions]
        assert ActionType.WIN in action_types, "WIN action should be available for winning hand"

        # 验证：WIN 动作参数为 -1
        win_action = next(a for a in actions if a.action_type == ActionType.WIN)
        assert win_action.parameter == -1, "WIN action parameter should be -1"

    def test_no_win_when_cannot_form_win(self):
        """测试：无法胡牌时不应返回 WIN 动作"""
        # 创建游戏上下文
        context = GameContext(num_players=4)
        context.current_state = GameStateType.WAITING_RESPONSE
        context.discard_player = 1
        context.last_discarded_tile = 8  # 1万
        context.wall = deque([i for i in range(34) for _ in range(4)])

        # 设置玩家0的手牌（无法胡牌）
        player0 = context.players[0]
        player0.hand_tiles = [0, 1, 2, 9, 10, 11, 18, 19, 20, 30, 31]  # 散牌
        player0.melds = []
        player0.special_gangs = [0, 0, 0]

        # 设置其他玩家
        for i in range(1, 4):
            player = context.players[i]
            player.hand_tiles = [3, 4, 5, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27]
            player.melds = []
            player.special_gangs = [0, 0, 0]

        # 创建 ActionValidator
        validator = ActionValidator(context)

        # 测试
        actions = validator.detect_available_actions_after_discard(
            player0, context.last_discarded_tile, context.discard_player
        )

        # 验证：WIN 动作不在可用动作列表中
        action_types = [a.action_type for a in actions]
        assert ActionType.WIN not in action_types, "WIN action should NOT be available for non-winning hand"

    def test_cannot_win_own_discard(self):
        """测试：不能接自己的炮"""
        context = GameContext(num_players=4)
        context.current_state = GameStateType.WAITING_RESPONSE
        context.discard_player = 0  # 玩家0自己弃牌
        context.last_discarded_tile = 8
        context.wall = deque([i for i in range(34) for _ in range(4)])

        player0 = context.players[0]
        player0.hand_tiles = [0, 1, 2, 9, 10, 11, 18, 19, 20, 8, 8]
        player0.melds = []
        player0.special_gangs = [0, 0, 0]

        validator = ActionValidator(context)

        actions = validator.detect_available_actions_after_discard(
            player0, context.last_discarded_tile, context.discard_player
        )

        # 验证：不能接自己的炮
        assert len(actions) == 0, "Should have no actions when responding to own discard"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
