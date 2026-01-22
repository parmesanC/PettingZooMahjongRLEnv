"""
命令行手动控制器
使用元组形式输入，麻将牌直接用汉字
"""

import os
import sys
import re
from typing import Tuple, Dict
from .base import ManualController
from ..visualization.cli_renderer import SimpleCLIRenderer


class CLIManualController(ManualController):
    """
    命令行手动控制器（简化版）
    
    输入格式：(action_type, parameter)`
    例如：(0, 5) 表示打出5号牌（2万）
    """
    
    def __init__(self, env, max_episodes=1, strategies=None):
        super().__init__(env, max_episodes, strategies)
        self.renderer = SimpleCLIRenderer()
    
    def render_env(self):
        """渲染环境"""
        self.renderer.render(self.env.context, self.env.agent_selection)
    
    def get_human_action(self, observation, info) -> Tuple[int, int]:
        """获取人类动作（元组形式）"""
        action_mask = observation['action_mask']
        
        self.renderer.render_action_help(action_mask)
        
        while True:
            try:
                user_input = input("\n请输入动作 (格式: (action_type, parameter)): ").strip()
                action = self._parse_tuple_input(user_input, action_mask)
                if action is not None:
                    return action
            except KeyboardInterrupt:
                print("\n游戏退出")
                sys.exit(0)
            except ValueError as e:
                print(f"无效输入: {e}")
                print("请按格式输入，例如: (0, 5) 表示打出5号牌")
    
    def render_final_state(self, info):
        """渲染最终状态"""
        print("\n" + "=" * 60)
        winner = info.get('winners', [])
        if winner:
            print(f"🏆 获胜者: 玩家{winner[0]}")
            win_way = info.get('win_way', 'unknown')
            win_way_map = {0: "自摸", 1: "抢杠", 2: "杠开", 3: "点炮"}
            print(f"胜利方式: {win_way_map.get(win_way, win_way)}")
        else:
            print("荒牌流局")
        print("=" * 60)
    
    def _parse_tuple_input(self, user_input: str, action_mask) -> Tuple[int, int]:
        """
        解析元组形式输入
        
        支持格式：
        - (0, 5)
        - 0, 5
        - （action_type, parameter）
        """
        cleaned = user_input.replace(' ', '').replace('（', '(').replace('）', ')')
        
        match = re.match(r'\(?(\d+),\s*(\d+)\)?', cleaned)
        if not match:
            match = re.match(r'(\d+),\s*(\d+)', cleaned)
        
        if not match:
            raise ValueError("格式错误，请使用 (action_type, parameter) 格式")
        
        action_type = int(match.group(1))
        parameter = int(match.group(2))
        
        if not (0 <= action_type <= 10):
            raise ValueError(f"动作类型必须在0-10之间，当前为{action_type}")
        
        if action_type == 0:
            if not (0 <= parameter <= 33):
                raise ValueError(f"牌ID必须在0-33之间，当前为{parameter}")
        elif action_type == 1:
            if not (0 <= parameter <= 2):
                raise ValueError(f"吃牌类型必须在0-2之间，当前为{parameter}")
        elif action_type in [2, 3, 9, 10]:  # 碰牌、明杠、胡牌、过牌 不需要参数
            if parameter != -1:
                action_name = self._get_action_name(action_type)
                raise ValueError(f"{action_name}不需要参数")
        elif action_type in [4, 5, 6, 7, 8]:  # 补杠、暗杠、红中杠、皮子杠、赖子杠 需要牌ID参数
            if not (0 <= parameter <= 33):
                raise ValueError(f"牌ID必须在0-33之间，当前为{parameter}")
        
        valid = self._validate_action(action_type, parameter, action_mask)
        if not valid:
            raise ValueError(f"动作 ({action_type}, {parameter}) 当前不可用")
        
        return (action_type, parameter)
    
    def _validate_action(self, action_type: int, parameter: int, action_mask) -> bool:
        """验证动作是否有效（基于action_mask）"""
        types = action_mask['types']
        params = action_mask['params']

        if not types[action_type]:
            return False

        # 需要参数的动作类型：打牌、补杠、暗杠、红中杠、皮子杠、赖子杠
        if action_type in [0, 4, 5, 6, 7, 8]:
            if parameter >= 0 and parameter < len(params):
                return params[parameter]

        return True
    
    def _get_action_name(self, action_type: int) -> str:
        """获取动作名称"""
        names = {
            0: "打牌", 1: "吃牌", 2: "碰牌",
            3: "明杠", 4: "补杠", 5: "暗杠",
            6: "红中杠", 7: "皮子杠", 8: "赖子杠",
            9: "胡牌", 10: "过牌"
        }
        return names.get(action_type, "未知")
