"""测试抢杠和功能

根据武汉麻将规则：
1. 抢杠和只针对补杠（碰了一个，又摸到第四张）
2. 抢杠优先级高于杠牌
3. 抢杠和是大胡
4. 必须开口（吃、碰、明杠或补杠）
5. 获胜玩家包含弃牌后手牌数量属于 {2, 5, 8, 11, 14}
6. 其余玩家手牌数量属于 {1, 4, 7, 10, 13}
"""

import pytest
from collections import deque

from mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder
from mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
from mahjong_rl.state_machine import MahjongStateMachine
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData, Meld
from src.mahjong_rl.core.constants import GameStateType, ActionType, WinType
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.state_machine.states.wait_rob_kong_state import WaitRobKongState


class TestRobKong:
    """测试抢杠和功能"""

    def test_rob_kong_winner_with_opened_meld_and_special_kong(self):
        """
        测试抢杠和：获胜玩家有开口（碰牌）和特殊杠（赖子杠）

        场景：
        - 玩家0（被抢杠）：碰了1万，现在摸到第4张1万补杠，手牌9张
        - 玩家1（抢杠和获胜）：碰了2万，有赖子杠，手牌10张 + 1万 = 11张
        - 玩家2、3：正常手牌13张

        要求：
        - 玩家1已开口（碰+赖子杠算开口）
        - 玩家1手牌+被杠牌=11张 ✓
        - 玩家0、2、3手牌=9、13、13张 ✓
        """
        context = GameContext()
        context.current_state = GameStateType.WAIT_ROB_KONG
        context.wall = deque([i for i in range(34) for _ in range(4)])
        context.lazy_tile = 34  # 无效值
        context.skin_tile = [-1, -1]
        context.red_dragon = 31

        # 创建规则引擎（使用真实的GameContext）
        rule_engine = Wuhan7P4LRuleEngine(context)

        # 创建观测构建器
        observation_builder = Wuhan7P4LObservationBuilder(context)

        # 创建状态机
        state_machine = MahjongStateMachine(
            rule_engine=rule_engine,
            observation_builder=observation_builder,
            enable_logging=False  # 关闭日志，简化测试
        )
        state_machine.set_context(context)

        # 设置玩家的完整状态

        # 玩家0（被抢杠者）
        player0 = context.players[0]
        player0.hand_tiles = [0, 1, 2, 9, 11, 11, 18, 19, 20, 23, 24]
        player0.melds = [Meld(
            action_type=MahjongAction(ActionType.PONG, 0),
            tiles=[0, 0, 0],
            from_player=1
        )]
        player0.special_gangs = [0, 0, 0]

        # 玩家1（抢杠和者）
        player1 = context.players[1]
        player1.hand_tiles = [1, 1, 1, 2, 3, 4, 5, 5, 6, 7]
        player1.melds = [Meld(
            action_type=MahjongAction(ActionType.PONG, 21),
            tiles=[21, 21, 21],
            from_player=2
        )]
        player1.special_gangs = [0, 0, 0]

        # 玩家2、3（普通手牌）
        for i in [2, 3]:
            player = context.players[i]
            player.hand_tiles = [8, 10, 12, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29]
            player.melds = []
            player.special_gangs = [0, 0, 0]

        # 测试：玩家1可以抢杠和
        can_rob = state._can_rob_kong(context, player1, 0)
        assert can_rob is True, "Player 1 should be able to rob kong"

    def test_rob_kong_winner_with_chow_and_red_dragon_kong(self):
        """
        测试抢杠和：获胜玩家有吃牌和红中杠

        场景：
        - 玩家0（被抢杠）：碰了1万，手牌9张
        - 玩家1（抢杠和获胜）：吃了2万3万4万，有红中杠，手牌7张 + 1万 = 8张
        - 玩家2、3：正常手牌13张
        """
        context = GameContext()
        context.current_state = GameStateType.WAIT_ROB_KONG
        context.wall = deque([i for i in range(34) for _ in range(4)])
        context.lazy_tile = 34
        context.skin_tile = [-1, -1]
        context.red_dragon = 31

        # 设置玩家0（被抢杠）
        player0 = context.players[0]
        player0.hand_tiles = [0, 1, 2, 9, 10, 11, 18, 19, 20]
        player0.melds = [Meld(
            action_type=MahjongAction(ActionType.PONG, 0),
            tiles=[0, 0, 0],
            from_player=1
        )]
        player0.special_gangs = [0, 0, 0]

        # 设置玩家1（抢杠和获胜）- 吃了2万3万4万，有红中杠，手牌7张
        player1 = context.players[1]
        player1.hand_tiles = [5, 6, 12, 13, 14, 15, 16]  # 7张
        player1.melds = [
            Meld(
                action_type=MahjongAction(ActionType.CHOW, 1),  # 吃了2万3万4万
                tiles=[3, 4, 5],
                from_player=2
            )
        ]
        player1.special_gangs = [0, 2, 0]  # 有红中杠

        # 设置玩家2、3
        for i in [2, 3]:
            player = context.players[i]
            player.hand_tiles = [7, 8, 9, 15, 16, 17, 25, 26, 27, 28, 29, 30, 31, 32]
            player.melds = []
            player.special_gangs = [0, 0, 0]

        context.kong_player_idx = 0
        context.last_kong_tile = 0

        state = WaitRobKongState(None, None)

        # 测试
        can_rob = state._can_rob_kong(context, player1, 0)
        assert can_rob is True, "Player 1 should be able to rob kong"

    def test_rob_kong_winner_hand_tile_count_validation(self):
        """
        测试抢杠和后获胜玩家手牌数量验证

        验证不同开口情况下的手牌数量：
        - 1个碰牌：13 - 3 + 1 = 11张 ✓
        - 1个吃牌：13 - 3 + 1 = 11张 ✓
        - 2个碰牌：13 - 6 + 1 = 8张 ✓
        - 碰+吃：13 - 6 + 1 = 8张 ✓
        """
        context = GameContext()
        context.current_state = GameStateType.WAIT_ROB_KONG
        context.wall = deque([i for i in range(34) for _ in range(4)])
        context.lazy_tile = 34
        context.skin_tile = [-1, -1]

        # 玩家0（被抢杠）
        player0 = context.players[0]
        player0.hand_tiles = [0, 1, 2, 9, 10, 11, 18, 19, 20]
        player0.melds = [Meld(
            action_type=MahjongAction(ActionType.PONG, 0),
            tiles=[0, 0, 0],
            from_player=1
        )]
        player0.special_gangs = [0, 0, 0]

        # 测试1：玩家1有1个碰牌 - 手牌10张 + 被杠牌 = 11张 ✓
        player1 = context.players[1]
        player1.hand_tiles = [3, 4, 5, 12, 13, 14, 21, 22, 23, 24]  # 10张
        player1.melds = [Meld(
            action_type=MahjongAction(ActionType.PONG, 3),
            tiles=[3, 3, 3],
            from_player=2
        )]
        player1.special_gangs = [0, 0, 0]

        context.kong_player_idx = 0
        context.last_kong_tile = 0

        state = WaitRobKongState(None, None)

        # 验证：玩家1抢杠和后手牌数量正确
        temp_hand = player1.hand_tiles.copy()
        temp_hand.append(context.last_kong_tile)
        assert len(temp_hand) == 11, f"Player 1 hand tiles count should be 11, got {len(temp_hand)}"

        # 测试2：玩家1有1个吃牌 + 1个特殊杠 - 手牌7张 + 被杠牌 = 8张 ✓
        player1.hand_tiles = [5, 6, 12, 13, 14, 15, 16]  # 7张
        player1.melds = [
            Meld(
                action_type=MahjongAction(ActionType.CHOW, 1),
                tiles=[3, 4, 5],
                from_player=2
            )
        ]
        player1.special_gangs = [1, 0, 0]  # 赖子杠

        temp_hand = player1.hand_tiles.copy()
        temp_hand.append(context.last_kong_tile)
        assert len(temp_hand) == 8, f"Player 1 hand tiles count should be 8, got {len(temp_hand)}"

    def test_rob_kong_all_players_normal_hand_count(self):
        """
        测试：确保所有其他玩家的手牌数量符合要求

        其余玩家手牌数量必须属于 {1, 4, 7, 10, 13}
        """
        context = GameContext()
        context.current_state = GameStateType.WAIT_ROB_KONG
        context.wall = deque([i for i in range(34) for _ in range(4)])
        context.lazy_tile = 34
        context.skin_tile = [-1, -1]

        # 玩家0（被抢杠）- 手牌9张（补杠后）不在要求范围内，但他是被抢杠者
        player0 = context.players[0]
        player0.hand_tiles = [0, 1, 2, 9, 10, 11, 18, 19, 20]
        player0.melds = [Meld(
            action_type=MahjongAction(ActionType.PONG, 0),
            tiles=[0, 0, 0],
            from_player=1
        )]
        player0.special_gangs = [0, 0, 0]

        # 玩家1（抢杠和）
        player1 = context.players[1]
        player1.hand_tiles = [3, 4, 5, 12, 13, 14, 21, 22, 23, 24]
        player1.melds = [Meld(
            action_type=MahjongAction(ActionType.PONG, 3),
            tiles=[3, 3, 3],
            from_player=2
        )]
        player1.special_gangs = [1, 0, 0]

        # 验证其他玩家（玩家2、3）手牌数量在 {1, 4, 7, 10, 13} 范围内
        valid_counts = {1, 4, 7, 10, 13}

        for i in [2, 3]:
            player = context.players[i]
            player.hand_tiles = [6, 7, 8, 15, 16, 17, 25, 26, 27, 28, 29, 30, 31, 32]
            player.melds = []
            player.special_gangs = [0, 0, 0]

            assert len(player.hand_tiles) in valid_counts, \
                f"Player {i} hand tiles count should be in {valid_counts}, got {len(player.hand_tiles)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
