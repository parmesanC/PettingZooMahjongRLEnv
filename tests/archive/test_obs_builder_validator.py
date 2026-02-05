"""
诊断脚本：检查 ObservationBuilder 中的 ActionValidator 初始化

该脚本用于检查 Wuhan7P4LObservationBuilder 在调用 ActionValidator 时，
context 的状态是否正确。
"""

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder
from src.mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
from src.mahjong_rl.core.constants import Tiles


def test_observation_builder_action_validator():
    """测试 ObservationBuilder 中的 ActionValidator 初始化"""
    print("=" * 80)
    print("测试 ObservationBuilder 中的 ActionValidator")
    print("=" * 80)

    # 创建游戏上下文
    context = GameContext()

    # 初始化特殊牌
    context.skin_tile[0] = 5  # 6万
    context.skin_tile[1] = 4  # 5万
    context.lazy_tile = 6  # 7万
    context.red_dragon = Tiles.RED_DRAGON.value

    # 创建4个玩家
    context.players = [PlayerData(player_id=i) for i in range(4)]

    # 为玩家2添加赖子牌
    context.players[2].hand_tiles = [0, 1, 2, 6, 9, 10, 11, 12, 18, 19, 20]
    print(f"玩家2手牌: {sorted(context.players[2].hand_tiles)}")

    # 创建 ObservationBuilder
    obs_builder = Wuhan7P4LObservationBuilder(context)

    print("\n调用 build_action_mask...")
    print("-" * 80)

    # 构建 action_mask（会调用 ActionValidator）
    action_mask = obs_builder.build_action_mask(2, context)

    print(f"\naction_mask 形状: {action_mask.shape}")
    print(f"action_mask 非零位置: {[(i, int(action_mask[i])) for i in range(len(action_mask)) if action_mask[i] > 0]}")

    # 检查赖子杠位（145位action_mask，KONG_LAZY在索引108）
    lazy_kong_pos = 108
    print(f"\nKONG_LAZY 位 ({lazy_kong_pos}): {action_mask[lazy_kong_pos]}")

    if action_mask[lazy_kong_pos] > 0:
        print(f"✓ 检测到赖子杠可用")
    else:
        print(f"✗ 未检测到赖子杠")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


def test_observation_builder_with_meld_decision():
    """测试 MELD_DECISION 状态下的 action_mask 构建"""
    print("\n" + "=" * 80)
    print("测试 MELD_DECISION 状态下的 action_mask")
    print("=" * 80)

    # 创建游戏上下文
    context = GameContext()

    # 初始化特殊牌
    context.skin_tile[0] = 5  # 6万
    context.skin_tile[1] = 4  # 5万
    context.lazy_tile = 6  # 7万
    context.red_dragon = Tiles.RED_DRAGON.value

    # 创建4个玩家
    context.players = [PlayerData(player_id=i) for i in range(4)]

    # 玩家2鸣牌后（有11张牌），包含赖子牌
    context.players[2].hand_tiles = [0, 1, 2, 6, 9, 10, 11, 12, 18, 19, 20]
    # 添加一个meld（鸣牌）
    from src.mahjong_rl.core.PlayerData import Meld
    from src.mahjong_rl.core.mahjong_action import MahjongAction
    from src.mahjong_rl.core.constants import ActionType
    context.players[2].melds.append(
        Meld(
            action_type=MahjongAction(ActionType.PONG, 3),
            tiles=[3, 3, 3],
            from_player=0
        )
    )

    print(f"玩家2手牌: {sorted(context.players[2].hand_tiles)}")
    print(f"玩家2melds: {[(m.action_type.action_type.name, m.tiles) for m in context.players[2].melds]}")

    # 设置当前状态为 MELD_DECISION
    from src.mahjong_rl.core.constants import GameStateType
    context.current_state = GameStateType.MELD_DECISION

    # 创建 ObservationBuilder
    obs_builder = Wuhan7P4LObservationBuilder(context)

    print("\n调用 build_action_mask (MELD_DECISION 状态)...")
    print("-" * 80)

    # 构建 action_mask
    action_mask = obs_builder.build_action_mask(2, context)

    print(f"\naction_mask 形状: {action_mask.shape}")

    # 检查赖子杠位（145位action_mask，KONG_LAZY在索引108）
    if context.lazy_tile is not None:
        lazy_kong_pos = 108
        print(f"赖子杠位置 ({lazy_kong_pos}): {action_mask[lazy_kong_pos]}")

    if action_mask[lazy_kong_pos] > 0:
        print(f"✓ 检测到赖子杠可用")
    else:
        print(f"✗ 未检测到赖子杠")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_observation_builder_action_validator()
    test_observation_builder_with_meld_decision()
