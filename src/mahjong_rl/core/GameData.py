import random
import warnings
from collections import deque
from dataclasses import field, dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import ACTION_SPACE_SIZE, ActionType, GameStateType, Tiles, WinWay
from src.mahjong_rl.rules.round_info import RoundInfo


@dataclass
class ActionRecord:
    """动作历史记录"""
    action_type: MahjongAction
    tile: Optional[int]  # 涉及的牌（如打出的牌）
    player_id: int


@dataclass
class GameContext:
    """强化学习专用游戏上下文"""
    # 核心状态
    current_state: GameStateType = None

    # 游戏数据
    wall: deque[int] = field(default_factory=deque)  # 牌墙（标准牌值0-33）
    discard_pile: List[int] = field(default_factory=list)  # 弃牌堆
    players: List[PlayerData] = field(default_factory=lambda: [PlayerData(player_id=i) for i in range(4)])
    current_player_idx: int = 0  # 当前操作玩家索引
    dealer_idx: int = None  # 庄家索引
    last_discarded_tile: Optional[int] = None  # 最后打出的牌（标准牌值）
    last_kong_tile: Optional[int] = None  # 最后杠的牌
    last_drawn_tile: Optional[int] = None  # 最后摸到的牌（用于PLAYER_DECISION状态检测可用动作）
    pending_discard_tile: Optional[int] = None  # 待打出的牌（由PLAYER_DECISION状态设置，DISCARDING状态执行）
    pending_responses: Optional[Dict[int, ActionType]] = field(default_factory=dict)  # 等待响应的玩家动作类型（吃/碰/杠）

    # 特殊牌（动态生成）
    lazy_tile: int = None  # 赖子牌
    skin_tile: List[int] = field(default_factory=lambda: [-1, -1])  # 皮子牌
    red_dragon: int = Tiles.RED_DRAGON.value  # 红中牌
    special_tiles: Tuple[int, int, int, int] = field(default_factory=lambda: (-1, -1, -1, 31))  # 第一位为赖子，第二、第三位为皮子、第四位为红中

    # 状态数据
    winner_ids: List[int] = field(default_factory=list)
    is_kong_draw: bool = False  # 是否杠后摸牌
    is_win: bool = False
    is_flush: bool = False
    win_way: Optional[int] = None  # 胡牌方式: 自摸、点炮、杠开、抢杠
    discard_player: Optional[int] = None  # 点炮者

    # 轮次管理
    round_info: RoundInfo = field(default_factory=RoundInfo)
    player_seat_winds: List[int] = field(default_factory=list)  # 玩家座位风（0:东, 1:南, 2:西, 3:北）

    # 强化学习接口
    observation: Dict[str, Any] = field(default_factory=dict)  # 符合您定义的观测空间
    action_mask: np.ndarray = field(default_factory=lambda: np.ones(ACTION_SPACE_SIZE, dtype=bool))
    reward: float = 0.0

    # 相应顺序管理
    response_order: List[int] = field(default_factory=list)  # 响应顺序（玩家索引）
    current_responder_idx: int = 0  # 当前响应玩家索引
    response_priorities: Dict[int, int] = field(default_factory=dict)  # 玩家响应优先级（值越小优先级越高）
    selected_responder: Optional[int] = None  # 最终被选中的响应者
    last_kong_action: Optional[MahjongAction] = None  # 最后一次杠牌动作（供GongState使用）

    # 响应状态优化：真实响应者列表（排除只能 PASS 的玩家）
    active_responders: List[int] = field(default_factory=list)  # 真实需要响应的玩家列表
    active_responder_idx: int = 0  # 当前在 active_responders 中的索引

    # 历史记录
    action_history: List[ActionRecord] = field(default_factory=list)

    # 轮庄专用字段
    next_dealer_idx: Optional[int] = None  # 下一局庄家
    retain_dealer_count: int = 0  # 连庄次数

    @classmethod
    def create_new_round(
            cls,
            previous_context: Optional['GameContext'] = None,
            seed: Optional[int] = None
    ) -> 'GameContext':
        """
        [已废弃] 创建新一局游戏，考虑轮庄规则

        .. deprecated::
            此方法已废弃。初始化逻辑现在由 InitialState 完成。
            请使用以下方式替代：

            1. 创建空 GameContext: context = GameContext()
            2. 传入 round_info: context.round_info = env.round_info
            3. 让 InitialState 完成初始化

        新的初始化流程：
        - WuhanMahjongEnv 内部管理 RoundInfo
        - InitialState.step() 负责洗牌、发牌、定庄
        - 游戏结束时更新 round_info 用于下一局
        """
        warnings.warn(
            "GameContext.create_new_round() 已废弃，请让 InitialState 完成初始化。"
            "参考 WuhanMahjongEnv.reset() 的新实现。",
            DeprecationWarning,
            stacklevel=2
        )
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        context = cls()

        # 如果提供了上一局的信息，应用轮庄规则
        if previous_context:
            # 确定下一局庄家
            context.dealer_idx = cls._determine_next_dealer(previous_context)

            # 继承轮次信息（推进一局）
            context.round_info = previous_context.round_info
            context.round_info.advance_round(
                is_dealer_retained=(context.dealer_idx == previous_context.dealer_idx)
            )

            # 更新连庄次数
            if context.dealer_idx == previous_context.dealer_idx:
                context.retain_dealer_count = previous_context.retain_dealer_count + 1
            else:
                context.retain_dealer_count = 0

            # 继承座位风
            context.player_seat_winds = previous_context.player_seat_winds.copy()
        else:
            # 新游戏，随机确定庄家
            context.dealer_idx = random.randint(0, 3)
            context.retain_dealer_count = 0

            # 确定座位风：庄家为东风，逆时针依次为南、西、北
            context.player_seat_winds = []
            for i in range(4):
                seat_wind = (4 - context.dealer_idx + i) % 4  # 0:东, 1:南, 2:西, 3:北
                context.player_seat_winds.append(seat_wind)

        # 设置当前玩家为庄家
        context.current_player_idx = context.dealer_idx

        # 初始化牌墙
        context._initialize_wall()

        # 生成特殊牌
        context._generate_special_tiles()

        return context



    def __post_init__(self):
        """初始化"""
        if self.dealer_idx is None:
            self.dealer_idx = 0
            self.players[0].is_dealer = True

    def setup_response_order(self, discard_player: int) -> None:
        """
        设置响应顺序

        响应顺序：下家、对家、上家（跳过打牌玩家）

        Args:
            discard_player: 打牌玩家的索引
        """
        self.response_order = [
            (discard_player + 1) % 4,
            (discard_player + 2) % 4,
            (discard_player + 3) % 4
        ]
        self.current_responder_idx = 0

    def get_current_responder(self) -> Optional[int]:
        """
        获取当前响应者ID

        Returns:
            当前响应者索引，如果没有则返回 None
        """
        if self.current_responder_idx < len(self.response_order):
            return self.response_order[self.current_responder_idx]
        return None

    def move_to_next_responder(self) -> None:
        """移动到下一个响应者"""
        self.current_responder_idx += 1

    def is_all_responded(self) -> bool:
        """
        检查是否所有玩家都已响应

        Returns:
            True 如果所有玩家都已响应
        """
        return self.current_responder_idx >= len(self.response_order)


