"""
验证状态机修复
"""

import os
import sys

# 确保项目路径在 sys.path 中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"项目根目录: {project_root}")

try:
    print("\n1. 导入模块...")
    from src.mahjong_rl.core.GameData import GameContext
    from src.mahjong_rl.core.PlayerData import PlayerData
    from src.mahjong_rl.core.constants import GameStateType, ActionType
    from src.mahjong_rl.core.mahjong_action import MahjongAction
    from src.mahjong_rl.state_machine.machine import MahjongStateMachine
    from src.mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
    from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder
    from src.mahjong_rl.rules.round_info import RoundInfo
    print("✓ 导入成功")

    print("\n2. 创建游戏上下文...")
    context = GameContext()
    context.players = [PlayerData(player_id=i) for i in range(4)]
    context.round_info = RoundInfo()
    context.round_info.dealer_position = 0
    print("✓ 游戏上下文创建成功")

    print("\n3. 初始化规则引擎和观测构建器...")
    rule_engine = Wuhan7P4LRuleEngine(context)
    observation_builder = Wuhan7P4LObservationBuilder(context)
    print("✓ 初始化成功")

    print("\n4. 创建状态机...")
    state_machine = MahjongStateMachine(rule_engine, observation_builder, enable_logging=True)
    state_machine.set_context(context)
    print(f"✓ 状态机创建成功")
    print(f"  当前状态类型: {state_machine.current_state_type}")

    print("\n5. 转换到初始状态...")
    state_machine.transition_to(GameStateType.INITIAL, context)
    print(f"✓ 转换成功")
    print(f"  当前状态类型: {state_machine.current_state_type.name}")

    print("\n6. 执行第一次 step...")
    next_state = state_machine.step(context, 'auto')
    print(f"✓ Step 完成")
    print(f"  下一个状态: {next_state.name if next_state else 'None'}")
    print(f"  当前状态类型: {state_machine.current_state_type.name}")

    print("\n7. 检查状态历史...")
    history = state_machine.get_history()
    print(f"✓ 状态历史记录数: {len(history)}")
    for i, snap in enumerate(history):
        state_name = snap['state_type'].name if snap['state_type'] else 'None'
        print(f"  快照 {i}: {state_name}")

    print("\n8. 测试回滚...")
    if len(history) > 0:
        try:
            print(f"  回滚前状态: {state_machine.current_state_type.name}")
            rolled_back_context = state_machine.rollback(1)
            current_state_name = state_machine.current_state_type.name if state_machine.current_state_type else 'None'
            print(f"✓ 回滚成功")
            print(f"  回滚后状态: {current_state_name}")
        except Exception as e:
            print(f"✗ 回滚失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ 没有历史记录，无法回滚")

    print("\n9. 测试自动推进...")
    for i in range(5):
        if state_machine.is_terminal():
            print(f"✓ 到达终端状态")
            break

        current_state_name = state_machine.current_state_type.name if state_machine.current_state_type else 'None'
        print(f"\n  步骤 {i + 1}:")
        print(f"    当前状态: {current_state_name}")

        try:
            # 根据状态类型决定动作
            if state_machine.current_state_type == GameStateType.PLAYER_DECISION:
                # 手动状态：打一张牌
                current_player = context.players[context.current_player_idx]
                if len(current_player.hand_tiles) > 0:
                    discard_tile = current_player.hand_tiles[0]
                    action = MahjongAction(ActionType.DISCARD, discard_tile)
                    print(f"    动作: 打出牌 {discard_tile}")
                    next_state = state_machine.step(context, action)
                    if next_state:
                        print(f"    下一个状态: {next_state.name}")
                else:
                    print(f"    无牌可打，跳过")
                    break
            else:
                # 自动状态
                next_state = state_machine.step(context, 'auto')
                if next_state:
                    print(f"    下一个状态: {next_state.name}")
        except Exception as e:
            print(f"    ✗ Step 失败: {e}")
            import traceback
            traceback.print_exc()
            break

    print("\n" + "="*60)
    print("✓ 所有测试完成！")
    print("="*60)

except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
