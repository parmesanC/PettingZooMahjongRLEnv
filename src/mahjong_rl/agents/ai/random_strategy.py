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
            observation: 观测字典（action_mask 现在是 278 位的一维数组）
            action_mask: 扁平化的 278 位动作掩码

        Returns:
            (action_type, parameter) 元组
        """
        # 定义索引范围
        RANGES = {
            'DISCARD': (0, 34),
            'CHOW': (34, 37),
            'PONG': (37, 38),
            'KONG_EXPOSED': (38, 72),
            'KONG_SUPPLEMENT': (72, 106),
            'KONG_CONCEALED': (106, 140),
            'KONG_RED': (140, 174),
            'KONG_LAZY': (174, 208),
            'KONG_SKIN': (208, 276),
            'WIN': (276, 277),
            'PASS': (277, 278),
        }

        # 收集所有可用的动作类型
        available_actions = []

        # 检查每个动作类型
        for action_type, (start, end) in RANGES.items():
            segment = action_mask[start:end]

            if np.any(segment > 0):
                # 该动作类型可用
                action_type_value = ActionType[action_type].value

                if action_type in ['DISCARD', 'KONG_EXPOSED', 'KONG_SUPPLEMENT',
                                  'KONG_CONCEALED']:
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

                elif action_type in ['KONG_RED', 'KONG_LAZY']:
                    # 特殊杠：从掩码中找出具体牌ID
                    valid_tiles = np.where(segment > 0)[0]
                    if len(valid_tiles) > 0:
                        param = int(np.random.choice(valid_tiles))
                        available_actions.append((action_type_value, param))

                elif action_type == 'KONG_SKIN':
                    # 皮子杠：两个皮子，各34位
                    for i in range(2):
                        skin_segment = segment[i * 34:(i + 1) * 34]
                        valid_tiles = np.where(skin_segment > 0)[0]
                        if len(valid_tiles) > 0:
                            param = int(np.random.choice(valid_tiles))
                            available_actions.append((action_type_value, param))
                            break  # 只选一个皮子杠

                elif action_type in ['PONG', 'WIN', 'PASS']:
                    # 无参数动作
                    available_actions.append((action_type_value, 0))

        # 如果没有可用动作，返回默认
        if len(available_actions) == 0:
            return (ActionType.PASS.value, 0)

        # 随机选择一个动作
        return tuple(available_actions[np.random.choice(len(available_actions))])

    def reset(self):
        """重置策略"""
        pass
