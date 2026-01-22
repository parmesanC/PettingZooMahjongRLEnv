from typing import Dict, Optional

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import ActionType, ResponsePriority


class ResponseAction:
    """响应动作封装类"""

    def __init__(self, player_id: int, action_type: ActionType, priority: ResponsePriority, parameter: int = -1):
        self.player_id = player_id
        self.action_type = action_type
        self.priority = priority
        self.parameter = parameter  # 吃法的参数（0=左吃, 1=中吃, 2=右吃），对于其他动作为-1
        self.clockwise_distance: Optional[int] = None  # 顺时针距离出牌者的距离


class ResponseCollector:
    """响应收集器 - 返回 ResponseAction 对象"""

    def __init__(self):
        self.responses: Dict[int, ResponseAction] = {}

    def add_response(self, player_id: int, action_type: ActionType, priority: ResponsePriority, parameter: int = -1):
        """添加响应"""
        self.responses[player_id] = ResponseAction(player_id, action_type, priority, parameter)

    def get_best_response(self, context: GameContext) -> Optional[ResponseAction]:
        """获取最佳响应（考虑优先级和位置）

        返回: ResponseAction对象或None
        """
        if not self.responses:
            return None

        # print(f"\n=== 收集到的响应 ===")
        # for player_id, response in self.responses.items():
            # print(f"玩家{player_id}: {response.action_type.name} (优先级: {response.priority.value})")

        # 找到最高优先级的响应
        min_priority = min(response.priority.value for response in self.responses.values())

        # 筛选出最高优先级的响应
        best_responses = [
            r for r in self.responses.values() if r.priority.value == min_priority
        ]

        # print(f"最高优先级: {min_priority}, 候选响应: {len(best_responses)}个")

        if len(best_responses) == 1:
            result = best_responses[0]
            # print(f"唯一候选: 玩家{result.player_id}")
            return result

        # 多个同优先级响应，按顺时针距离排序（距离越小优先级越高）
        # print(f"同优先级{min_priority}的多个响应，按距离排序:")

        # 为每个候选响应添加距离信息
        candidates_with_distance = []
        for response in best_responses:
            player_id = response.player_id
            distance = context.response_priorities.get(player_id, float('inf'))
            candidates_with_distance.append((response, distance))
            # print(f"  玩家{player_id}: 距离={distance}")

        # 选择距离最近的（值最小）
        best_response = min(candidates_with_distance, key=lambda x: x[1])[0]
        # print(
        #     f"最终选择: 玩家{best_response.player_id} (距离: {context.response_priorities.get(best_response.player_id)})")

        return best_response

    def reset(self):
        """重置收集器"""
        self.responses.clear()

    def get_all_responses(self) -> Dict[int, ResponseAction]:
        """获取所有响应（用于调试）"""
        return self.responses.copy()


# ============ 测试用例 ============

def create_test_context(discarder_idx: int = 0, tile: int = 10) -> GameContext:
    """创建测试上下文"""
    context = GameContext()
    context.current_player_idx = discarder_idx
    context.last_discarded_tile = tile
    context.setup_response_order(discarder_idx)
    return context


def action_type_to_priority(action_type: ActionType) -> ResponsePriority:
    """动作类型转优先级"""
    priority_map = {
        ActionType.WIN: ResponsePriority.WIN,
        ActionType.KONG_EXPOSED: ResponsePriority.KONG,
        ActionType.PONG: ResponsePriority.PONG,
        ActionType.CHOW_RIGHT: ResponsePriority.CHOW,
        ActionType.CHOW_MIDDLE: ResponsePriority.CHOW,
        ActionType.CHOW_LEFT: ResponsePriority.CHOW,
        ActionType.PASS: ResponsePriority.PASS,
    }
    return priority_map.get(action_type, ResponsePriority.PASS)


def test_case_1():
    """测试用例1：和牌优先级最高"""
    print("\n=== 测试用例1：和牌优先级最高 ===")
    print("场景：玩家0出牌，玩家3（上家）和牌，其他玩家碰牌")
    print("预期：选择玩家3和牌（优先级最高）")

    context = create_test_context(discarder_idx=0, tile=10)
    collector = ResponseCollector()

    # 模拟响应（按顺时针顺序：玩家3是下家，玩家2是对家，玩家1是上家）
    # 注意：setup_response_order设置了顺时针顺序
    collector.add_response(3, ActionType.WIN, ResponsePriority.WIN)  # 下家和牌
    collector.add_response(2, ActionType.PONG, ResponsePriority.PONG)  # 对家碰
    collector.add_response(1, ActionType.PONG, ResponsePriority.PONG)  # 上家碰

    best_response = collector.get_best_response(context)

    print(f"\n结果: {best_response}")
    assert best_response is not None, "应该返回一个响应"
    assert best_response.player_id == 3, "应该选择玩家3（和牌优先级最高）"
    assert best_response.action_type == ActionType.WIN, "应该是和牌"
    print("✓ 测试通过")


def test_case_2():
    """测试用例2：同优先级按距离选择"""
    print("\n=== 测试用例2：同优先级按距离选择 ===")
    print("场景：玩家0出牌，所有玩家都碰牌")
    print("预期：选择玩家3（下家，距离最近）")

    context = create_test_context(discarder_idx=0, tile=10)
    collector = ResponseCollector()

    # 所有玩家都碰牌
    collector.add_response(3, ActionType.PONG, ResponsePriority.PONG)  # 下家碰
    collector.add_response(2, ActionType.PONG, ResponsePriority.PONG)  # 对家碰
    collector.add_response(1, ActionType.PONG, ResponsePriority.PONG)  # 上家碰

    best_response = collector.get_best_response(context)

    print(f"\n结果: {best_response}")
    assert best_response is not None, "应该返回一个响应"
    assert best_response.player_id == 3, "应该选择玩家3（下家，距离最近）"
    assert best_response.action_type == ActionType.PONG, "应该是碰牌"
    print("✓ 测试通过")


def test_case_3():
    """测试用例3：杠 vs 碰 vs 吃"""
    print("\n=== 测试用例3：杠 vs 碰 vs 吃 ===")
    print("场景：玩家0出牌，玩家3（下家）吃，玩家2（对家）碰，玩家1（上家）杠")
    print("预期：选择玩家1杠（杠优先级高于碰和吃）")

    context = create_test_context(discarder_idx=0, tile=10)
    collector = ResponseCollector()

    collector.add_response(3, ActionType.CHOW_MIDDLE, ResponsePriority.CHOW)  # 下家吃
    collector.add_response(2, ActionType.PONG, ResponsePriority.PONG)  # 对家碰
    collector.add_response(1, ActionType.KONG_EXPOSED, ResponsePriority.KONG)  # 上家杠

    best_response = collector.get_best_response(context)

    print(f"\n结果: {best_response}")
    assert best_response is not None, "应该返回一个响应"
    assert best_response.player_id == 1, "应该选择玩家1（杠优先级最高）"
    assert best_response.action_type == ActionType.KONG_EXPOSED, "应该是杠牌"
    print("✓ 测试通过")


def test_case_4():
    """测试用例4：只有过牌"""
    print("\n=== 测试用例4：只有过牌 ===")
    print("场景：玩家0出牌，所有玩家都过")
    print("预期：返回None或过牌（具体实现决定）")

    context = create_test_context(discarder_idx=0, tile=10)
    collector = ResponseCollector()

    collector.add_response(3, ActionType.PASS, ResponsePriority.PASS)  # 下家过
    collector.add_response(2, ActionType.PASS, ResponsePriority.PASS)  # 对家过
    collector.add_response(1, ActionType.PASS, ResponsePriority.PASS)  # 上家过

    best_response = collector.get_best_response(context)

    print(f"\n结果: {best_response}")
    # 注意：这里会返回一个过牌响应（优先级最低）
    # 实际使用中，调用者可以检查是否是PASS并做相应处理
    if best_response:
        assert best_response.action_type == ActionType.PASS, "应该是过牌"
    print("✓ 测试通过")


def test_case_5():
    """测试用例5：多个和牌"""
    print("\n=== 测试用例5：多个和牌 ===")
    print("场景：玩家0出牌，玩家3和玩家2都和牌")
    print("预期：选择玩家3（下家，距离最近）")

    context = create_test_context(discarder_idx=0, tile=10)
    collector = ResponseCollector()

    collector.add_response(3, ActionType.WIN, ResponsePriority.WIN)  # 下家和
    collector.add_response(2, ActionType.WIN, ResponsePriority.WIN)  # 对家和
    collector.add_response(1, ActionType.PASS, ResponsePriority.PASS)  # 上家过

    best_response = collector.get_best_response(context)

    print(f"\n结果: {best_response}")
    assert best_response is not None, "应该返回一个响应"
    assert best_response.player_id == 3, "应该选择玩家3（下家，距离最近）"
    assert best_response.action_type == ActionType.WIN, "应该是和牌"
    print("✓ 测试通过")


def test_case_6():
    """测试用例6：混合情况，下家吃 vs 对家碰"""
    print("\n=== 测试用例6：下家吃 vs 对家碰 ===")
    print("场景：玩家0出牌，玩家3（下家）吃，玩家2（对家）碰")
    print("预期：选择玩家2碰（碰优先级高于吃）")

    context = create_test_context(discarder_idx=0, tile=10)
    collector = ResponseCollector()
    # 响应顺序会重新计算：玩家3是下家，玩家2是对家，玩家1是上家
    collector.add_response(3, ActionType.CHOW_MIDDLE, ResponsePriority.CHOW)  # 下家吃
    collector.add_response(2, ActionType.PONG, ResponsePriority.PONG)  # 对家碰
    collector.add_response(1, ActionType.PASS, ResponsePriority.PASS)  # 上家过

    best_response = collector.get_best_response(context)

    print(f"\n结果: {best_response}")
    assert best_response is not None, "应该返回一个响应"
    assert best_response.player_id == 2, "应该选择玩家2（碰优先级高于吃）"
    assert best_response.action_type == ActionType.PONG, "应该是碰牌"
    print("✓ 测试通过")


def test_case_7():
    """测试用例7：不同出牌者"""
    print("\n=== 测试用例7：不同出牌者 ===")
    print("场景：玩家2出牌，玩家3（下家）和牌，玩家0（对家）碰，玩家1（上家）杠")
    print("预期：选择玩家3和牌（优先级最高）")

    context = create_test_context(discarder_idx=2, tile=15)
    collector = ResponseCollector()

    # 响应顺序会重新计算：玩家3是下家，玩家0是对家，玩家1是上家
    collector.add_response(3, ActionType.WIN, ResponsePriority.WIN)  # 下家和
    collector.add_response(0, ActionType.PONG, ResponsePriority.PONG)  # 对家碰
    collector.add_response(1, ActionType.KONG_EXPOSED, ResponsePriority.KONG)  # 上家杠

    best_response = collector.get_best_response(context)

    print(f"\n结果: {best_response}")
    assert best_response is not None, "应该返回一个响应"
    assert best_response.player_id == 3, "应该选择玩家3（和牌优先级最高）"
    assert best_response.action_type == ActionType.WIN, "应该是和牌"
    print("✓ 测试通过")


def test_case_8():
    """测试用例8：响应顺序验证"""
    print("\n=== 测试用例8：响应顺序验证 ===")
    print("场景：测试不同出牌者的响应顺序计算")

    # 测试玩家0出牌
    context = create_test_context(discarder_idx=0, tile=10)
    print(f"\n玩家0出牌时，响应顺序: {context.response_order}")
    print(f"响应优先级: {context.response_priorities}")
    # 期望：玩家3（下家）、玩家2（对家）、玩家1（上家）
    assert context.response_order == [3, 2, 1], "玩家0出牌时响应顺序错误"
    assert context.response_priorities == {3: 1, 2: 2, 1: 3}, "优先级错误"

    # 测试玩家1出牌
    context.current_player_idx = 1
    context.setup_response_order(1)
    print(f"\n玩家1出牌时，响应顺序: {context.response_order}")
    print(f"响应优先级: {context.response_priorities}")
    # 期望：玩家0（下家）、玩家3（对家）、玩家2（上家）
    assert context.response_order == [0, 3, 2], "玩家1出牌时响应顺序错误"
    assert context.response_priorities == {0: 1, 3: 2, 2: 3}, "优先级错误"

    print("✓ 测试通过")


# ============ 集成测试 ============

def test_integration():
    """集成测试：模拟完整流程"""
    print("\n=== 集成测试：完整响应流程 ===")

    # 创建规则引擎模拟
    class MockRuleEngine:
        def can_respond(self, player, tile):
            # 所有玩家都可以碰牌
            return {ActionType.PONG, ActionType.KONG_EXPOSED}

        def can_win(self, player, tile):
            # 只有玩家3可以和牌
            return player.player_id == 3 or player.player_id == 2

    # 创建上下文和状态
    context = create_test_context(discarder_idx=0, tile=10)
    rule_engine = MockRuleEngine()

    # 创建收集器
    collector = ResponseCollector()

    print("模拟场景：玩家0出牌，玩家3和牌，玩家2和牌，玩家1过")

    # 模拟响应
    # 玩家3（下家）和牌
    if rule_engine.can_win(context.players[3], context.last_discarded_tile):
        collector.add_response(3, ActionType.WIN, ResponsePriority.WIN)

    # 玩家2（对家）碰牌
    if rule_engine.can_win(context.players[2], context.last_discarded_tile):
        collector.add_response(2, ActionType.WIN, ResponsePriority.WIN)

    # 玩家1（上家）过
    collector.add_response(1, ActionType.PASS, ResponsePriority.PASS)

    # 获取最佳响应
    best_response = collector.get_best_response(context)

    print(f"\n最终选择: {best_response}")

    if best_response:
        print(f"玩家{best_response.player_id} {best_response.action_type.name}")

        # 根据响应类型决定下一步
        if best_response.action_type == ActionType.WIN:
            print("→ 进入和牌状态")
            context.winner_ids.append(best_response.player_id)
        elif best_response.action_type == ActionType.KONG_EXPOSED:
            print("→ 进入杠牌状态")
            context.current_player_idx = best_response.player_id
        elif best_response.action_type in [ActionType.PONG, ActionType.CHOW_MIDDLE]:
            print("→ 进入副露处理状态")
            context.current_player_idx = best_response.player_id
        elif best_response.action_type == ActionType.PASS:
            print("→ 所有玩家都过，进入下一家摸牌")
    else:
        print("→ 没有有效响应")

    print("✓ 集成测试完成")


# ============ 运行所有测试 ============

def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("开始麻将响应收集器测试")
    print("=" * 50)

    tests = [
        test_case_1,
        test_case_2,
        test_case_3,
        test_case_4,
        test_case_5,
        test_case_6,
        test_case_7,
        test_case_8,
        test_integration,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ 测试失败: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"测试完成: 通过 {passed}, 失败 {failed}")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("所有测试通过！🎉")
    else:
        print("有测试失败，请检查！")