#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# 确保项目路径在 sys.path 中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 打开日志文件
with open('verify_log.txt', 'w', encoding='utf-8') as log_file:
    def log(msg):
        print(msg, file=log_file)
        log_file.flush()
    
    log(f"项目根目录: {project_root}")
    
    try:
        log("\n1. 导入模块...")
        from src.mahjong_rl.core.GameData import GameContext
        from src.mahjong_rl.core.PlayerData import PlayerData
        from src.mahjong_rl.core.constants import GameStateType, ActionType
        from src.mahjong_rl.core.mahjong_action import MahjongAction
        from src.mahjong_rl.state_machine.machine import MahjongStateMachine
        from src.mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
        from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder
        from src.mahjong_rl.rules.round_info import RoundInfo
        log("✓ 导入成功")

        log("\n2. 创建游戏上下文...")
        context = GameContext()
        context.players = [PlayerData(player_id=i) for i in range(4)]
        context.round_info = RoundInfo()
        context.round_info.dealer_position = 0
        log("✓ 游戏上下文创建成功")

        log("\n3. 初始化规则引擎和观测构建器...")
        rule_engine = Wuhan7P4LRuleEngine(context)
        observation_builder = Wuhan7P4LObservationBuilder(context)
        log("✓ 初始化成功")

        log("\n4. 创建状态机...")
        state_machine = MahjongStateMachine(rule_engine, observation_builder, enable_logging=True)
        state_machine.set_context(context)
        log(f"✓ 状态机创建成功")
        log(f"  当前状态类型: {state_machine.current_state_type}")

        log("\n5. 转换到初始状态...")
        state_machine.transition_to(GameStateType.INITIAL, context)
        log(f"✓ 转换成功")
        log(f"  当前状态类型: {state_machine.current_state_type.name}")

        log("\n6. 执行第一次 step...")
        next_state = state_machine.step(context, 'auto')
        log(f"✓ Step 完成")
        log(f"  下一个状态: {next_state.name if next_state else 'None'}")
        log(f"  当前状态类型: {state_machine.current_state_type.name}")

        log("\n7. 检查状态历史...")
        history = state_machine.get_history()
        log(f"✓ 状态历史记录数: {len(history)}")
        for i, snap in enumerate(history):
            state_name = snap['state_type'].name if snap['state_type'] else 'None'
            log(f"  快照 {i}: {state_name}")

        log("\n8. 测试回滚...")
        if len(history) > 0:
            try:
                log(f"  回滚前状态: {state_machine.current_state_type.name}")
                rolled_back_context = state_machine.rollback(1)
                current_state_name = state_machine.current_state_type.name if state_machine.current_state_type else 'None'
                log(f"✓ 回滚成功")
                log(f"  回滚后状态: {current_state_name}")
            except Exception as e:
                log(f"✗ 回滚失败: {e}")
                import traceback
                traceback.print_exc(file=log_file)
        else:
            log("✗ 没有历史记录，无法回滚")

        log("\n9. 测试自动推进...")
        for i in range(5):
            if state_machine.is_terminal():
                log(f"✓ 到达终端状态")
                break

            current_state_name = state_machine.current_state_type.name if state_machine.current_state_type else 'None'
            log(f"\n  步骤 {i + 1}:")
            log(f"    当前状态: {current_state_name}")

            try:
                # 根据状态类型决定动作
                if state_machine.current_state_type == GameStateType.PLAYER_DECISION:
                    # 手动状态：打一张牌
                    current_player = context.players[context.current_player_idx]
                    if len(current_player.hand_tiles) > 0:
                        discard_tile = current_player.hand_tiles[0]
                        action = MahjongAction(ActionType.DISCARD, discard_tile)
                        log(f"    动作: 打出牌 {discard_tile}")
                        next_state = state_machine.step(context, action)
                        if next_state:
                            log(f"    下一个状态: {next_state.name}")
                    else:
                        log(f"    无牌可打，跳过")
                        break
                else:
                    # 自动状态
                    next_state = state_machine.step(context, 'auto')
                    if next_state:
                        log(f"    下一个状态: {next_state.name}")
            except Exception as e:
                log(f"    ✗ Step 失败: {e}")
                import traceback
                traceback.print_exc(file=log_file)
                break

        log("\n" + "="*60)
        log("✓ 所有测试完成！")
        log("="*60)

    except Exception as e:
        log(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc(file=log_file)
        sys.exit(1)
