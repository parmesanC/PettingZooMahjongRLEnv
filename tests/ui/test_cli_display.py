"""测试CLI显示修复"""
import sys
sys.path.insert(0, '.')

from src.mahjong_rl.core.constants import Tiles

# 测试1: Tiles.get_tile_name()返回中文名称
print("=" * 60)
print("测试1: Tiles.get_tile_name() 返回中文名称")
print("=" * 60)

test_cases = [
    (0, "1万"),
    (5, "6万"),
    (8, "9万"),
    (9, "1条"),
    (17, "9条"),
    (18, "1筒"),
    (26, "9筒"),
    (27, "东风"),
    (28, "南风"),
    (29, "西风"),
    (30, "北风"),
    (31, "红中"),
    (32, "发财"),
    (33, "白板"),
]

all_passed = True
for tile_id, expected in test_cases:
    result = Tiles.get_tile_name(tile_id)
    if result == expected:
        print(f"✓ ID {tile_id:2d}: {result}")
    else:
        print(f"✗ ID {tile_id:2d}: 期望 '{expected}', 得到 '{result}'")
        all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("测试1通过！所有牌名正确显示为中文")
else:
    print("测试1失败！部分牌名不正确")
print("=" * 60)

# 测试2: 测试环境初始化和action_mask
print("\n" + "=" * 60)
print("测试2: 环境初始化和action_mask")
print("=" * 60)

try:
    from example_mahjong_env import WuhanMahjongEnv

    env = WuhanMahjongEnv(render_mode=None, training_phase=3, enable_logging=False)
    obs, info = env.reset(seed=42)

    print(f"✓ 环境初始化成功")
    print(f"  当前agent: {env.agent_selection}")
    print(f"  当前状态: {env.state_machine.current_state_type.name}")

    # 检查observation
    if 'action_mask' in obs:
        action_mask = obs['action_mask']
        types = action_mask['types']
        params = action_mask['params']

        print(f"\n✓ action_mask存在")
        print(f"  types: {types}")
        print(f"  params: {params[:20]}...")  # 只显示前20个

        # 检查是否有可用的动作
        available_actions = []
        for i in range(len(types)):
            if types[i] > 0:
                action_names = {
                    0: "打牌",
                    1: "吃牌",
                    2: "碰牌",
                    3: "明杠",
                    4: "补杠",
                    5: "暗杠",
                    6: "红中杠",
                    7: "皮子杠",
                    8: "赖子杠",
                    9: "胡牌",
                    10: "过牌",
                }
                available_actions.append(action_names.get(i, f"动作{i}"))

        if available_actions:
            print(f"\n✓ 可用动作: {', '.join(available_actions)}")
        else:
            print(f"\n✗ 没有可用的动作！")
    else:
        print(f"✗ observation中没有action_mask")

    env.close()
    print("\n测试2完成！")

except Exception as e:
    print(f"✗ 测试2失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("所有测试完成！")
print("=" * 60)
