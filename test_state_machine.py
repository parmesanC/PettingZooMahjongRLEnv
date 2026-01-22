"""
状态机测试脚本

测试MahjongStateMachine的基本功能，包括状态转换、自动推进等。
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import GameStateType, Tiles
from src.mahjong_rl.core.mahjong_action import MahjongAction, ActionType
from src.mahjong_rl.state_machine.machine import MahjongStateMachine
from src.mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
from src.mahjong_rl.observation.wuhan_7p4l_observation_builder import Wuhan7P4LObservationBuilder
from src.mahjong_rl.rules.round_info import RoundInfo


def test_state_machine_initialization():
    """测试状态机初始化"""
    print("=" * 60)
    print("测试1: 状态机初始化")
    print("=" * 60)
    
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
    
    print(f"✓ 状态机创建成功")
    print(f"✓ 注册状态数: {len(state_machine.states)}")
    print(f"✓ 日志启用: {state_machine.logger is not None}")
    
    return state_machine, context


def test_initial_state(state_machine, context):
    """测试初始状态"""
    print("\n" + "=" * 60)
    print("测试2: 初始状态")
    print("=" * 60)
    
    # 转换到初始状态
    state_machine.transition_to(GameStateType.INITIAL, context)
    print(f"✓ 进入初始状态")
    print(f"✓ 当前状态: {state_machine.current_state_type.name}")
    
    # 执行step
    next_state = state_machine.step(context, 'auto')
    print(f"✓ 初始状态step完成")
    print(f"✓ 下一个状态: {next_state.name}")
    
    # 检查发牌
    print(f"\n发牌情况:")
    for i, player in enumerate(context.players):
        print(f"  玩家{i}: {len(player.hand_tiles)}张牌")
    
    # 检查特殊牌
    print(f"\n特殊牌:")
    print(f"  赖子: {context.lazy_tile}")
    print(f"  皮子: {context.skin_tile}")
    print(f"  红中: {context.red_dragon}")
    print(f"  牌墙剩余: {len(context.wall)}张")


def test_state_history(state_machine, context):
    """测试状态历史和回滚"""
    print("\n" + "=" * 60)
    print("测试3: 状态历史和回滚")
    print("=" * 60)
    
    # 检查历史记录
    history = state_machine.get_history()
    print(f"✓ 状态历史记录数: {len(history)}")
    
    if len(history) > 0:
        print(f"✓ 最新快照: {history[-1]['state_type'].name}")
    
    # 测试回滚
    try:
        if len(history) > 0:
            rolled_back_context = state_machine.rollback(1)
            print(f"✓ 回滚成功")
            print(f"✓ 回滚到状态: {state_machine.current_state_type.name}")
    except Exception as e:
        print(f"✗ 回滚失败: {e}")


def test_auto_progression(state_machine, context):
    """测试自动推进"""
    print("\n" + "=" * 60)
    print("测试4: 自动推进")
    print("=" * 60)
    
    # 尝试自动推进几个状态
    max_steps = 10
    step_count = 0
    
    for i in range(max_steps):
        if state_machine.is_terminal():
            print(f"✓ 到达终端状态")
            break
        
        print(f"\n步骤 {i + 1}:")
        print(f"  当前状态: {state_machine.current_state_type.name}")
        print(f"  当前玩家: {state_machine.get_current_player_id()}")
        
        # 根据状态类型决定动作
        if state_machine.current_state_type == GameStateType.PLAYER_DECISION:
            # 简单测试：随机打一张牌
            current_player = context.players[context.current_player_idx]
            if len(current_player.hand_tiles) > 0:
                discard_tile = current_player.hand_tiles[0]
                action = MahjongAction(ActionType.DISCARD, discard_tile)
                print(f"  动作: 打出牌{discard_tile}")
                next_state = state_machine.step(context, action)
                if next_state:
                    print(f"  下一个状态: {next_state.name}")
        else:
            # 自动状态
            try:
                next_state = state_machine.step(context, 'auto')
                if next_state:
                    print(f"  下一个状态: {next_state.name}")
                step_count += 1
            except Exception as e:
                print(f"  ✗ Step失败: {e}")
                break
    
    print(f"\n✓ 自动推进完成，共执行{step_count}步")


def test_logger(state_machine):
    """测试日志功能"""
    print("\n" + "=" * 60)
    print("测试5: 日志功能")
    print("=" * 60)
    
    logger = state_machine.get_logger()
    if logger:
        history = logger.get_history()
        print(f"✓ 日志记录数: {len(history)}")
        
        # 显示最近几条日志
        print("\n最近日志:")
        for i, log_entry in enumerate(history[-5:]):
            log_type = log_entry.get('type', 'unknown')
            if log_type == 'transition':
                print(f"  [{i+1}] 状态转换: {log_entry['from_state']} -> {log_entry['to_state']}")
            elif log_type == 'action':
                print(f"  [{i+1}] 玩家{log_entry['player_id']}动作: {log_entry['action_type']}")
            elif log_type == 'error':
                print(f"  [{i+1}] 错误: {log_entry['message']}")
            else:
                print(f"  [{i+1}] 日志: {log_entry.get('message', 'N/A')}")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Mahjong状态机测试")
    print("=" * 60)
    
    try:
        # 测试1: 状态机初始化
        state_machine, context = test_state_machine_initialization()
        
        # 测试2: 初始状态
        test_initial_state(state_machine, context)
        
        # 测试3: 状态历史和回滚
        test_state_history(state_machine, context)
        
        # 测试4: 自动推进
        test_auto_progression(state_machine, context)
        
        # 测试5: 日志功能
        test_logger(state_machine)
        
        print("\n" + "=" * 60)
        print("所有测试完成！✓")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
