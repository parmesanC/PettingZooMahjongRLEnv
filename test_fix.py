#!/usr/bin/env python
"""
临时测试脚本 - 用于验证状态机修复
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import GameStateType
from src.mahjong_rl.state_machine.machine import MahjongStateMachine
from src.mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder
from src.mahjong_rl.rules.round_info import RoundInfo

def test_rollback():
    """测试回滚功能"""
    results = []

    # 创建游戏上下文
    context = GameContext()
    context.players = [PlayerData(player_id=i) for i in range(4)]
    context.round_info = RoundInfo()
    context.round_info.dealer_position = 0

    # 初始化规则引擎和观测构建器
    rule_engine = Wuhan7P4LRuleEngine(context)
    observation_builder = Wuhan7P4LObservationBuilder(context)

    # 创建状态机
    state_machine = MahjongStateMachine(rule_engine, observation_builder, enable_logging=True)
    state_machine.set_context(context)

    results.append("状态机创建成功")

    # 转换到初始状态
    state_machine.transition_to(GameStateType.INITIAL, context)
    results.append(f"进入初始状态: {state_machine.current_state_type.name}")

    # 执行第一次step
    next_state = state_machine.step(context, 'auto')
    results.append(f"第一次step完成: {next_state.name if next_state else 'None'}")

    # 检查快照历史
    history = state_machine.get_history()
    results.append(f"状态历史记录数: {len(history)}")

    for i, snap in enumerate(history):
        state_name = snap['state_type'].name if snap['state_type'] else 'None'
        results.append(f"  快照{i}: {state_name}")

    # 测试回滚
    if len(history) > 0:
        try:
            rolled_back_context = state_machine.rollback(1)
            current_state_name = state_machine.current_state_type.name if state_machine.current_state_type else 'None'
            results.append(f"回滚成功: 当前状态 = {current_state_name}")
        except Exception as e:
            results.append(f"回滚失败: {e}")
    else:
        results.append("没有历史记录，无法回滚")

    # 尝试自动推进
    results.append("\n自动推进测试:")
    for i in range(3):
        if state_machine.is_terminal():
            results.append(f"到达终端状态")
            break

        current_state_name = state_machine.current_state_type.name if state_machine.current_state_type else 'None'
        results.append(f"步骤 {i + 1}: 当前状态 = {current_state_name}")

        try:
            next_state = state_machine.step(context, 'auto')
            if next_state:
                results.append(f"  下一个状态: {next_state.name}")
        except Exception as e:
            results.append(f"  Step失败: {e}")
            break

    # 写入结果文件
    with open('test_results.txt', 'w', encoding='utf-8') as f:
        for result in results:
            f.write(result + '\n')

    return True

if __name__ == "__main__":
    try:
        test_rollback()
        sys.exit(0)
    except Exception as e:
        with open('test_results.txt', 'w', encoding='utf-8') as f:
            f.write(f"测试失败: {e}\n")
            import traceback
            traceback.print_exc(file=f)
        sys.exit(1)
