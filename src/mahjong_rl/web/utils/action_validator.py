"""动作验证工具 - 从 CLI 控制器提取的可复用验证逻辑"""

import numpy as np
from typing import Tuple, Optional


class ActionValidator:
    """
    动作验证器

    基于145位 action_mask 验证动作的合法性。
    复用自 CLI 控制器的验证逻辑。
    """

    # action_mask 索引范围定义
    ACTION_RANGES = {
        0: (0, 33),      # DISCARD
        1: (34, 36),     # CHOW
        2: (37, 37),     # PONG
        3: (38, 38),     # KONG_EXPOSED
        4: (39, 72),     # KONG_SUPPLEMENT
        5: (73, 106),    # KONG_CONCEALED
        6: (107, 107),   # KONG_RED
        7: (108, 141),   # KONG_SKIN
        8: (142, 142),   # KONG_LAZY
        9: (143, 143),   # WIN
        10: (144, 144),  # PASS
    }

    ACTION_NAMES = {
        0: "打牌", 1: "吃牌", 2: "碰牌",
        3: "明杠", 4: "补杠", 5: "暗杠",
        6: "红中杠", 7: "皮子杠", 8: "赖子杠",
        9: "胡牌", 10: "过牌"
    }

    def validate_action(self, action_type: int, parameter: int, action_mask: np.ndarray) -> bool:
        """
        验证动作是否有效

        Args:
            action_type: 动作类型 (0-10)
            parameter: 动作参数（牌ID或吃牌类型）
            action_mask: 145位动作掩码

        Returns:
            True if action is valid, False otherwise
        """
        # 检查动作类型是否在有效范围内
        if action_type not in self.ACTION_RANGES:
            return False

        start, end = self.ACTION_RANGES[action_type]

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

            return action_mask[index] == 1

        # 对于不需要参数的动作类型
        if action_type in [6, 8]:  # 红中杠、赖子杠
            return action_mask[start] == 1

        return True

    def get_action_name(self, action_type: int) -> str:
        """
        获取动作名称

        Args:
            action_type: 动作类型

        Returns:
            动作的中文名称
        """
        return self.ACTION_NAMES.get(action_type, "未知")

    def validate_action_with_error_message(
        self,
        action_type: int,
        parameter: int,
        action_mask: np.ndarray
    ) -> Tuple[bool, Optional[str]]:
        """
        验证动作并返回错误消息

        Args:
            action_type: 动作类型
            parameter: 动作参数
            action_mask: 动作掩码

        Returns:
            (is_valid, error_message) 元组
        """
        if self.validate_action(action_type, parameter, action_mask):
            return True, None

        # 生成错误消息
        action_name = self.get_action_name(action_type)
        return False, f"⚠️ {action_name} 当前不可用"
