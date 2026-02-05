"""场景测试框架使用示例

演示如何使用流式测试构建器测试各种游戏场景。
"""

from tests.scenario.builder import ScenarioBuilder
from tests.scenario.validators import wall_count_equals, hand_count_equals, state_is
from src.mahjong_rl.core.constants import GameStateType, ActionType


def create_standard_wall():
    """创建标准牌墙（每种牌4张，共136张）"""
    wall = []
    for tile_id in range(34):
        wall.extend([tile_id] * 4)
    return wall


def test_example_1_basic_discard():
    """示例1：测试基本的打牌流程"""
    wall = create_standard_wall()

    result = (
        ScenarioBuilder("基本打牌流程")
        .description("验证玩家能正常打牌并进入等待响应状态")
        .with_wall(wall)
        .with_special_tiles(lazy=8, skins=[7, 9])

        # 第一步：玩家0打出一张牌
        .step(1, "玩家0打1万（牌ID=0）")
        .action(0, ActionType.DISCARD, 0)
        .expect_state(GameStateType.WAITING_RESPONSE)
        .verify("牌墙未减少", wall_count_equals(136))  # 打牌不摸牌

        .run()
    )

    print(f"测试结果: {'通过' if result.success else '失败'}")
    if not result.success:
        print(f"失败原因: {result.failure_message}")
        print(f"快照: {result.final_context_snapshot}")


def test_example_2_concealed_kong():
    """示例2：测试暗杠流程"""
    wall = create_standard_wall()

    result = (
        ScenarioBuilder("暗杠流程")
        .description("验证玩家暗杠后状态转换")
        .with_wall(wall)

        .step(1, "玩家0暗杠1万")
        .action(0, ActionType.KONG_CONCEALED, 0)
        .expect_state(GameStateType.GONG)
        .verify("牌墙未减少", wall_count_equals(136))  # 暗杠不摸牌

        .step(2, "杠后自动补牌")
        .auto_advance()
        .expect_state(GameStateType.DRAWING_AFTER_GONG)
        .verify("牌墙减少1张", wall_count_equals(135))  # 补牌摸1张

        .run()
    )

    print(f"测试结果: {'通过' if result.success else '失败'}")


def test_example_3_custom_validator():
    """示例3：使用自定义验证器"""
    wall = create_standard_wall()

    result = (
        ScenarioBuilder("自定义验证器示例")
        .with_wall(wall)

        .step(1, "验证当前玩家是0")
        .auto_advance()
        .verify("当前玩家是玩家0", lambda ctx: ctx.current_player_idx == 0)
        .verify("当前是PLAYER_DECISION状态", state_is(GameStateType.PLAYER_DECISION))

        .run()
    )

    print(f"测试结果: {'通过' if result.success else '失败'}")


def test_example_4_expect_winner():
    """示例4：测试预期获胜者"""
    wall = create_standard_wall()

    # 这个测试可能会失败，因为随机对局不一定是玩家0赢
    # 但展示了如何设置预期获胜者
    result = (
        ScenarioBuilder("预期获胜者测试")
        .with_wall(wall)
        .expect_winner([0])  # 预期玩家0获胜
        .run()
    )

    if result.success:
        print("玩家0获胜！")
    else:
        print(f"测试失败: {result.failure_message}")
        print("（这是预期的，因为对局结果是随机的）")


# 运行示例
if __name__ == "__main__":
    print("=" * 60)
    print("场景测试框架使用示例")
    print("=" * 60)

    print("\n示例1: 基本打牌流程")
    test_example_1_basic_discard()

    print("\n示例2: 暗杠流程")
    test_example_2_concealed_kong()

    print("\n示例3: 自定义验证器")
    test_example_3_custom_validator()

    print("\n示例4: 预期获胜者")
    test_example_4_expect_winner()

    print("\n" + "=" * 60)
