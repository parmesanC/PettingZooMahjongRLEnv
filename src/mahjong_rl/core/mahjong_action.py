from dataclasses import dataclass

from src.mahjong_rl.core.constants import ActionType


@dataclass
class MahjongAction:
    """统一的动作表示"""
    action_type: ActionType  # 动作类型
    parameter: int = -1  # 参数ID（牌ID或吃牌类型）

    @property
    def is_valid(self) -> bool:
        """检查动作是否有效"""
        if self.action_type in [ActionType.WIN, ActionType.PASS]:
            return self.parameter == -1

        if self.action_type == ActionType.DISCARD:
            return 0 <= self.parameter < 34

        if self.action_type == ActionType.CHOW:
            return 0 <= self.parameter < 3  # 3种吃牌方式

        if self.action_type in [ActionType.KONG_SUPPLEMENT, ActionType.KONG_CONCEALED]:
            return 0 <= self.parameter < 34

        # PONG, KONG_EXPOSED 自动确定牌，parameter可忽略
        return True

    def __hash__(self):
        """自定义哈希函数，用于集合操作"""
        return hash((self.action_type.value, self.parameter))


# 动作空间维度定义
ACTION_TYPE_DIM = len(ActionType)
TILE_PARAM_DIM = 34  # 牌参数
CHOW_PARAM_DIM = 3  # 吃牌参数
MAX_PARAM_DIM = max(TILE_PARAM_DIM, CHOW_PARAM_DIM)  # 34