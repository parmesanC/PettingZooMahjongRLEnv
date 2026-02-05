"""
诊断脚本：检查 ActionValidator 初始化时机问题

该脚本专门用于检查 ActionValidator 的 special_tiles 构成问题，
特别是 self.pizi 引用问题。
"""

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.action_validator import ActionValidator
from src.mahjong_rl.core.constants import Tiles


def test_action_validator_initialization():
    """测试 ActionValidator 初始化时机"""
    print("=" * 80)
    print("测试 ActionValidator 初始化时机")
    print("=" * 80)

    # 场景1：在 skin_tile 初始化之前创建 ActionValidator
    print("\n场景1：在 skin_tile 初始化之前创建 ActionValidator")
    print("-" * 80)

    context1 = GameContext()
    context1.skin_tile = [-1, -1]  # 初始值
    context1.lazy_tile = None
    context1.red_dragon = Tiles.RED_DRAGON.value

    print(f"context1.skin_tile: {context1.skin_tile}")
    print(f"context1.lazy_tile: {context1.lazy_tile}")
    print(f"context1.red_dragon: {context1.red_dragon}")

    validator1 = ActionValidator(context1)

    print(f"validator1.pizi: {validator1.pizi}")
    print(f"validator1.special_tiles: {validator1.special_tiles}")
    print(f"id(validator1.pizi): {id(validator1.pizi)}")
    print(f"id(context1.skin_tile): {id(context1.skin_tile)}")
    print(f"validator1.pizi is context1.skin_tile: {validator1.pizi is context1.skin_tile}")

    # 现在更新 skin_tile
    print("\n更新 context1.skin_tile...")
    context1.skin_tile[0] = 5  # 6万
    context1.skin_tile[1] = 4  # 5万
    context1.lazy_tile = 6  # 7万

    print(f"context1.skin_tile: {context1.skin_tile}")
    print(f"context1.lazy_tile: {context1.lazy_tile}")
    print(f"validator1.pizi: {validator1.pizi}")
    print(f"validator1.special_tiles: {validator1.special_tiles}")

    # 场景2：在 skin_tile 初始化之后创建 ActionValidator
    print("\n" + "=" * 80)
    print("场景2：在 skin_tile 初始化之后创建 ActionValidator")
    print("-" * 80)

    context2 = GameContext()
    context2.skin_tile[0] = 5  # 6万
    context2.skin_tile[1] = 4  # 5万
    context2.lazy_tile = 6  # 7万
    context2.red_dragon = Tiles.RED_DRAGON.value

    print(f"context2.skin_tile: {context2.skin_tile}")
    print(f"context2.lazy_tile: {context2.lazy_tile}")
    print(f"context2.red_dragon: {context2.red_dragon}")

    validator2 = ActionValidator(context2)

    print(f"validator2.pizi: {validator2.pizi}")
    print(f"validator2.special_tiles: {validator2.special_tiles}")
    print(f"id(validator2.pizi): {id(validator2.pizi)}")
    print(f"id(context2.skin_tile): {id(context2.skin_tile)}")

    # 场景3：测试检测逻辑
    print("\n" + "=" * 80)
    print("场景3：测试赖子杠检测逻辑")
    print("-" * 80)

    # 创建一个有赖子牌的玩家
    player = PlayerData(player_id=0)
    player.hand_tiles = [0, 1, 2, 6, 6, 6, 6, 9, 10, 11, 12]  # 包含4张7万（赖子）

    print(f"玩家手牌: {sorted(player.hand_tiles)}")
    print(f"赖子牌 tile_id: {context2.lazy_tile} (7万)")

    # 检测可用动作
    print("\n调用 detect_available_actions_after_draw...")
    actions = validator2.detect_available_actions_after_draw(player, None)

    print(f"\n检测到的动作:")
    for action in actions:
        print(f"  {action.action_type.name}: {action.parameter}")

    # 检查是否有赖子杠
    lazy_kong_actions = [a for a in actions if a.action_type.name == 'KONG_LAZY']
    if lazy_kong_actions:
        print(f"\n✓ 检测到赖子杠: {lazy_kong_actions}")
    else:
        print(f"\n✗ 未检测到赖子杠")

    # 场景4：测试玩家手牌中没有赖子牌的情况
    print("\n" + "=" * 80)
    print("场景4：玩家手牌中没有赖子牌")
    print("-" * 80)

    player2 = PlayerData(player_id=1)
    player2.hand_tiles = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13]  # 没有赖子牌

    print(f"玩家手牌: {sorted(player2.hand_tiles)}")
    print(f"赖子牌 tile_id: {context2.lazy_tile} (7万)")

    print("\n调用 detect_available_actions_after_draw...")
    actions2 = validator2.detect_available_actions_after_draw(player2, None)

    print(f"\n检测到的动作:")
    for action in actions2:
        print(f"  {action.action_type.name}: {action.parameter}")

    # 检查是否有赖子杠
    lazy_kong_actions2 = [a for a in actions2 if a.action_type.name == 'KONG_LAZY']
    if lazy_kong_actions2:
        print(f"\n✗ 检测到赖子杠（不应该）: {lazy_kong_actions2}")
    else:
        print(f"\n✓ 正确：未检测到赖子杠（因为手牌中没有赖子牌）")

    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    test_action_validator_initialization()
