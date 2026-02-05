"""
测试 WaitResponseState 自动 PASS 优化功能
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import Tiles
from src.mahjong_rl.state_machine.states.wait_response_state import WaitResponseState
from src.mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder


class TestAutoPassOptimization:
    """测试自动 PASS 优化功能"""

    def setup_method(self):
        """每个测试前的设置"""
        # 创建基本的游戏上下文
        self.context = GameContext()
        self.rule_engine = Wuhan7P4LRuleEngine()
        self.obs_builder = Wuhan7P4LObservationBuilder()
        self.state = WaitResponseState(self.rule_engine, self.obs_builder)

    def test_all_pass_auto(self):
        """测试所有人只能 PASS 的场景"""
        # 设置一个所有人只能 PASS 的场景
        # 创建3个玩家
        player0 = PlayerData(player_id=0)
        player1 = PlayerData(player_id=1)
        player2 = PlayerData(player_id=2)

        self.context.players = [player0, player1, player2]
        self.context.current_player_idx = 0

        # 设置打出的牌为一张简单的条子 (3条 = tile_id 11)
        tile_id = 11  # Tiles.THREE_BAM.value
        self.context.last_discarded_tile = tile_id
        self.context.last_discard_player_idx = 0
        self.context.discard_player = 0

        # 设置响应顺序为所有玩家
        self.context.response_order = [1, 2]

        # 模拟没有任何玩家有可用的动作（只能 PASS）
        def mock_get_valid_responses(player_idx, discarded_tile):
            return []  # 空列表表示只能 PASS

        # 替换 rule_engine 的方法
        self.rule_engine.detect_available_actions_after_discard = mock_get_valid_responses

        # 进入状态
        self.state.enter(self.context)

        # 验证：active_responders 为空
        assert len(self.context.active_responders) == 0, \
            "当所有人只能 PASS 时，active_responders 应该为空"

        # 验证：状态直接转换到 DRAWING
        # 由于没有 active_responders，状态应该立即完成
        # 这里我们验证状态在 enter 后已经判断应该跳过

        print("✓ 测试通过：所有人只能 PASS 时正确跳过")

    def test_partial_responders(self):
        """测试部分玩家可以响应的场景"""
        # 设置场景：3个玩家中只有1个能碰牌
        player0 = PlayerData(player_id=0)
        player1 = PlayerData(player_id=1)
        player2 = PlayerData(player_id=2)

        self.context.players = [player0, player1, player2]
        self.context.current_player_idx = 0

        # 设置打出的牌 (3条 = tile_id 11)
        tile_id = 11
        self.context.last_discarded_tile = tile_id
        self.context.last_discard_player_idx = 0
        self.context.discard_player = 0

        # 设置响应顺序
        self.context.response_order = [1, 2]

        # 模拟只有玩家1能碰牌，玩家2只能 PASS
        def mock_get_valid_responses(player_idx, discarded_tile):
            if player_idx == 1:
                from src.mahjong_rl.core.mahjong_action import MahjongAction
                from src.mahjong_rl.core.constants import ActionType
                return [MahjongAction(ActionType.PONG, tile_id)]  # 玩家1能碰
            else:
                return []  # 玩家2只能 PASS

        self.rule_engine.detect_available_actions_after_discard = mock_get_valid_responses

        # 进入状态
        self.state.enter(self.context)

        # 验证：active_responders 只有1个玩家（玩家1）
        assert len(self.context.active_responders) == 1, \
            f"应该只有1个活跃响应者，实际有 {len(self.context.active_responders)} 个"
        assert 1 in self.context.active_responders, \
            "玩家1应该在 active_responders 中"
        assert 2 not in self.context.active_responders, \
            "玩家2不应该在 active_responders 中（自动PASS）"

        # 验证：active_responder_idx 从玩家1开始
        assert self.context.active_responder_idx == 0, \
            "active_responder_idx 应该从0开始"

        print("✓ 测试通过：部分玩家响应时正确过滤自动PASS玩家")

    def test_response_order_preserved(self):
        """测试响应顺序不被改变"""
        # 设置场景：4个玩家中2个能响应
        player0 = PlayerData(player_id=0)
        player1 = PlayerData(player_id=1)
        player2 = PlayerData(player_id=2)
        player3 = PlayerData(player_id=3)

        self.context.players = [player0, player1, player2, player3]
        self.context.current_player_idx = 0

        # 设置打出的牌 (5筒 = tile_id 22)
        tile_id = 22
        self.context.last_discarded_tile = tile_id
        self.context.last_discard_player_idx = 0
        self.context.discard_player = 0

        # 设置响应顺序：玩家1 -> 玩家2 -> 玩家3
        self.context.response_order = [1, 2, 3]

        # 模拟玩家1和玩家3能响应，玩家2只能PASS
        def mock_get_valid_responses(player_idx, discarded_tile):
            from src.mahjong_rl.core.mahjong_action import MahjongAction
            from src.mahjong_rl.core.constants import ActionType
            if player_idx == 1:
                return [MahjongAction(ActionType.PONG, tile_id)]
            elif player_idx == 3:
                return [MahjongAction(ActionType.KONG_EXPOSED, tile_id)]
            else:
                return []

        self.rule_engine.detect_available_actions_after_discard = mock_get_valid_responses

        # 进入状态
        self.state.enter(self.context)

        # 验证：active_responders 保持原始 response_order 的相对顺序
        assert len(self.context.active_responders) == 2, \
            f"应该有2个活跃响应者，实际有 {len(self.context.active_responders)} 个"

        # 验证顺序：玩家1应该在玩家3之前
        idx1 = self.context.active_responders.index(1)
        idx3 = self.context.active_responders.index(3)
        assert idx1 < idx3, \
            "active_responders 应该保持原始 response_order 的相对顺序"

        # 验证具体列表
        assert self.context.active_responders == [1, 3], \
            f"active_responders 应该是 [1, 3]，实际是 {self.context.active_responders}"

        print("✓ 测试通过：响应顺序保持正确")

    def test_empty_response_order(self):
        """测试空响应顺序的场景"""
        # 设置一个没有需要响应玩家的场景
        player0 = PlayerData(player_id=0)
        player1 = PlayerData(player_id=1)

        self.context.players = [player0, player1]
        self.context.current_player_idx = 0

        # 设置打出的牌 (7万 = tile_id 6)
        tile_id = 6
        self.context.last_discarded_tile = tile_id
        self.context.last_discard_player_idx = 0
        self.context.discard_player = 0

        # 空响应顺序
        self.context.response_order = []

        # 进入状态
        self.state.enter(self.context)

        # 验证：active_responders 为空
        assert len(self.context.active_responders) == 0, \
            "空响应顺序时，active_responders 应该为空"

        print("✓ 测试通过：空响应顺序处理正确")

    def test_single_responder(self):
        """测试单个响应者的场景"""
        # 设置场景：只有1个玩家能响应
        player0 = PlayerData(player_id=0)
        player1 = PlayerData(player_id=1)
        player2 = PlayerData(player_id=2)

        self.context.players = [player0, player1, player2]
        self.context.current_player_idx = 0

        # 设置打出的牌 (9条 = tile_id 17)
        tile_id = 17
        self.context.last_discarded_tile = tile_id
        self.context.last_discard_player_idx = 0
        self.context.discard_player = 0

        # 设置响应顺序
        self.context.response_order = [1, 2]

        # 模拟只有玩家1能响应
        def mock_get_valid_responses(player_idx, discarded_tile):
            from src.mahjong_rl.core.mahjong_action import MahjongAction
            from src.mahjong_rl.core.constants import ActionType
            if player_idx == 1:
                return [MahjongAction(ActionType.WIN, tile_id)]  # 玩家1能胡牌
            else:
                return []

        self.rule_engine.detect_available_actions_after_discard = mock_get_valid_responses

        # 进入状态
        self.state.enter(self.context)

        # 验证：只有玩家1在 active_responders 中
        assert self.context.active_responders == [1], \
            f"应该只有玩家1，实际是 {self.context.active_responders}"

        # 验证：active_responder_idx 正确初始化
        assert self.context.active_responder_idx == 0, \
            "active_responder_idx 应该初始化为0"

        print("✓ 测试通过：单个响应者场景处理正确")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Auto-Pass 优化功能测试")
    print("=" * 60)

    test_suite = TestAutoPassOptimization()

    tests = [
        ("所有人只能 PASS 的场景", test_suite.test_all_pass_auto),
        ("部分玩家可以响应的场景", test_suite.test_partial_responders),
        ("响应顺序不被改变", test_suite.test_response_order_preserved),
        ("空响应顺序的场景", test_suite.test_empty_response_order),
        ("单个响应者的场景", test_suite.test_single_responder),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"测试: {test_name}")
        print('=' * 60)
        try:
            test_suite.setup_method()
            test_func()
            passed += 1
            print(f"✓ 测试通过")
        except AssertionError as e:
            failed += 1
            print(f"✗ 测试失败: {e}")
        except Exception as e:
            failed += 1
            print(f"✗ 测试错误: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print('=' * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
