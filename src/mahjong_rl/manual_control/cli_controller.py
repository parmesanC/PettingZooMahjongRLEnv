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
                # 显示详细的错误/警告信息
                print(str(e))
                # 添加额外的帮助提示
                if "格式错误" in str(e):
                    print("提示: 请按格式输入，例如: (0, 5) 表示打出5号牌")
    
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
        - (10, -1) 过牌
        """
        cleaned = user_input.replace(' ', '').replace('（', '(').replace('）', ')')

        # 匹配支持负数
        match = re.match(r'\((-?\d+),\s*(-?\d+)\)', cleaned)

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
        elif action_type in [2, 3]:  # 碰牌、明杠 parameter 应该是 0（被忽略）
            if parameter != 0:
                action_name = self._get_action_name(action_type)
                raise ValueError(f"{action_name}的parameter应该是0")
        elif action_type in [9, 10]:  # 胡牌、过牌 不需要参数
            if parameter != -1:
                action_name = self._get_action_name(action_type)
                raise ValueError(f"{action_name}不需要参数")
        elif action_type in [6, 8]:  # 红中杠、赖子杠 只有1位，不需要牌ID参数
            if parameter != 0:
                action_name = self._get_action_name(action_type)
                raise ValueError(f"{action_name}的parameter应该是0")
        elif action_type in [4, 5, 7]:  # 补杠、暗杠、皮子杠 需要牌ID参数
            if not (0 <= parameter <= 33):
                raise ValueError(f"牌ID必须在0-33之间，当前为{parameter}")
        
        valid = self._validate_action(action_type, parameter, action_mask)
        if not valid:
            # 生成详细的警告信息
            action_name = self._get_action_name(action_type)
            warning_msg = f"⚠️  {action_name} 当前不可用"

            # 根据动作类型提供具体的帮助信息
            if action_type == 1:  # CHOW
                chow_names = ['左吃', '中吃', '右吃']
                available_chows = []
                for i in range(3):
                    if action_mask[34 + i]:
                        available_chows.append(f"{chow_names[i]}(1,{i})")
                if available_chows:
                    warning_msg += f"，可用: {', '.join(available_chows)}"
                else:
                    warning_msg += "，无法吃牌"

            elif action_type == 0:  # DISCARD
                # 列出可打出的牌
                playable_tiles = [i for i in range(34) if action_mask[i]]
                if playable_tiles:
                    from src.mahjong_rl.core.constants import Tiles
                    tile_names = [Tiles.get_tile_name(t) for t in playable_tiles]
                    if len(playable_tiles) <= 8:
                        warning_msg += f"，可用: {', '.join(tile_names)}"
                    else:
                        # 显示前几张和数量
                        warning_msg += f"，可用: {', '.join(tile_names[:5])} 等{len(playable_tiles)}张"

            elif action_type == 2:  # PONG
                warning_msg += "，无法碰这张牌"

            elif action_type in [3, 4, 5, 6, 7, 8]:  # 各种杠
                kong_names = {3: "明杠", 4: "补杠", 5: "暗杠", 6: "红中杠", 7: "皮子杠", 8: "赖子杠"}
                warning_msg += f"，无法执行{kong_names.get(action_type, '')}"

            # 添加 PASS 提示
            if action_mask[144]:
                warning_msg += "\n   提示: 可以选择过牌 (10, -1)"

            raise ValueError(warning_msg)
        
        return (action_type, parameter)
    
    def _validate_action(self, action_type: int, parameter: int, action_mask) -> bool:
        """验证动作是否有效（基于新的145位action_mask）"""
        action_ranges = {
            0: (0, 33),      # DISCARD
            1: (34, 36),     # CHOW
            2: (37, 37),     # PONG
            3: (38, 38),     # KONG_EXPOSED（1位）
            4: (39, 72),     # KONG_SUPPLEMENT
            5: (73, 106),    # KONG_CONCEALED
            6: (107, 107),   # KONG_RED（1位）
            7: (108, 141),   # KONG_SKIN（34位）
            8: (142, 142),   # KONG_LAZY（1位）
            9: (143, 143),   # WIN
            10: (144, 144),  # PASS
        }

        if action_type not in action_ranges:
            return False

        start, end = action_ranges[action_type]

        # 检查动作类型是否可用
        if not any(action_mask[start:end+1]):
            return False

        # 对于需要参数的动作类型，检查具体参数是否有效
        if action_type in [0, 4, 5, 7]:  # 打牌、补杠、暗杠、皮子杠
            if parameter < 0 or parameter > 33:
                return False

            # 计算对应的索引
            if action_type == 0:  # DISCARD
                index = parameter
            elif action_type == 4:  # KONG_SUPPLEMENT
                index = 39 + parameter
            elif action_type == 5:  # KONG_CONCEALED
                index = 73 + parameter
            elif action_type == 7:  # KONG_SKIN
                index = 108 + parameter
            else:
                return False

            return action_mask[index]

        # 对于不需要参数的动作类型（KONG_RED、KONG_LAZY只有1位）
        if action_type in [6, 8]:  # 红中杠、赖子杠
            start, end = action_ranges[action_type]
            return action_mask[start] == 1

        return True
    
    def _get_action_name(self, action_type: int) -> str:
        """获取动作名称"""
        names = {
            0: "打牌", 1: "吃牌", 2: "碰牌",
            3: "明杠", 4: "补杠", 5: "暗杠",
            6: "红中杠", 7: "赖子杠", 8: "皮子杠",
            9: "胡牌", 10: "过牌"
        }
        return names.get(action_type, "未知")
