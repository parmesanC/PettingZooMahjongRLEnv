"""
随机AI策略
用于测试和baseline
"""

import numpy as np
from typing import Tuple, Dict
from ..base import PlayerStrategy
from src.mahjong_rl.core.constants import ActionType


class RandomStrategy(PlayerStrategy):
    """
    随机策略

    从有效动作中随机选择。

    设计原则：
    - SRP: 单一职责 - 只负责随机选择动作
    - OCP: 可替换为其他AI策略
    """

    def choose_action(self, observation: Dict, action_mask: np.ndarray) -> Tuple[int, int]:
        """
        根据扁平化 action_mask 随机选择动作

        Args:
            observation: 观测字典（action_mask 现在是 244 位的一维数组）
            action_mask: 扁平化的 244 位动作掩码

        Returns:
            (action_type, parameter) 元组
        """
        # 定义索引范围（总长度：145位）
        RANGES = {
            'DISCARD': (0, 34),
            'CHOW': (34, 37),
            'PONG': (37, 38),
            'KONG_EXPOSED': (38, 39),       # 1位
            'KONG_SUPPLEMENT': (39, 73),
            'KONG_CONCEALED': (73, 107),
            'KONG_RED': (107, 108),       # 1位（全场只有一张红中）
            'KONG_SKIN': (108, 142),       # 34位（两张皮子独立）
            'KONG_LAZY': (142, 143),       # 1位（全场只有一张赖子）
            'WIN': (143, 144),             # 1位
            'PASS': (144, 145),            # 1位
        }

        # 收集所有可用的动作类型
        available_actions = []

        # 检查每个动作类型
        for action_type, (start, end) in RANGES.items():
            segment = action_mask[start:end]

            if np.any(segment > 0):
                # 该动作类型可用
                action_type_value = ActionType[action_type].value

                if action_type in ['DISCARD', 'KONG_SUPPLEMENT', 'KONG_CONCEALED']:
                    # 需要参数：从可用的牌ID中随机选择
                    valid_params = np.where(segment > 0)[0]
                    if len(valid_params) > 0:
                        param = int(np.random.choice(valid_params))
                        available_actions.append((action_type_value, param))

                elif action_type == 'CHOW':
                    # 吃法：0=左, 1=中, 2=右
                    valid_chows = np.where(segment > 0)[0]
                    if len(valid_chows) > 0:
                        param = int(np.random.choice(valid_chows))
                        available_actions.append((action_type_value, param))

                elif action_type == 'KONG_SKIN':
                    # 皮子杠：34位（两张皮子独立，需要参数选择具体哪张）
                    valid_tiles = np.where(segment > 0)[0]
                    if len(valid_tiles) > 0:
                        param = int(np.random.choice(valid_tiles))
                        available_actions.append((action_type_value, param))

                elif action_type in ['KONG_RED', 'KONG_LAZY', 'PONG', 'KONG_EXPOSED', 'WIN', 'PASS']:
                    # 无参数动作
                    # PONG 和 KONG_EXPOSED 的 parameter 被忽略（实际使用的是 discard_tile）
                    # KONG_RED 和 KONG_LAZY 只有 1 位，不需要参数
                    available_actions.append((action_type_value, 0))

        # 如果没有可用动作，返回默认
        if len(available_actions) == 0:
            return (ActionType.PASS.value, 0)

        # 随机选择一个动作
        return tuple(available_actions[np.random.choice(len(available_actions))])

    def reset(self):
        """重置策略"""
        pass
