"""动作验证工具 - 从 CLI 控制器提取的可复用验证逻辑"""

import numpy as np
from typing import Tuple, Optional


class ActionValidator:
    """
    动作验证器

    基于145位 action_mask 验证动作的合法性。
    复用自 CLI 控制器的验证逻辑。
    """

    # action_mask 索引范围定义（使用 Python 惯例：end 为独占）
    ACTION_RANGES = {
        0: (0, 34),      # DISCARD - indices 0-33
        1: (34, 37),     # CHOW - indices 34-36
        2: (37, 38),     # PONG - index 37
        3: (38, 39),     # KONG_EXPOSED - index 38
        4: (39, 73),     # KONG_SUPPLEMENT - indices 39-72
        5: (73, 107),    # KONG_CONCEALED - indices 73-106
        6: (107, 108),   # KONG_RED - index 107
        7: (108, 142),   # KONG_SKIN - indices 108-141
        8: (142, 143),   # KONG_LAZY - index 142
        9: (143, 144),   # WIN - index 143
        10: (144, 145),  # PASS - index 144
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
        # 验证 action_mask 形状
        if len(action_mask) != 145:
            return False

        # 检查动作类型是否在有效范围内
        if action_type not in self.ACTION_RANGES:
            return False

        start, end = self.ACTION_RANGES[action_type]

        # 检查动作类型是否可用（使用独占 end，不需要 +1）
        if not any(action_mask[start:end]):
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

        # CHOW 动作需要参数验证（0=左吃, 1=中吃, 2=右吃）
        if action_type == 1:
            if parameter not in [0, 1, 2]:
                return False
            index = 34 + parameter
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
