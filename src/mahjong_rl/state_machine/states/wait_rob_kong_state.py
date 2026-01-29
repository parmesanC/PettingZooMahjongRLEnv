"""
等待抢杠和状态 - 手动状态

该状态处理补杠时的抢杠和逻辑。
当玩家选择补杠时，其他玩家可以抢杠和。

功能:
1. 检查哪些玩家可以抢杠和
2. 逐个收集其他玩家的响应（WIN或PASS）
3. 如果有玩家抢杠和成功，进入WIN状态
4. 如果都PASS，回到GONG状态执行补杠
"""

from typing import Dict, List, Optional, Union

from src.mahjong_rl.core.GameData import GameContext, ActionRecord
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import GameStateType, ActionType
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.observation.builder import IObservationBuilder
from src.mahjong_rl.rules.base import IRuleEngine
from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.win_detector import WuhanMahjongWinChecker, WinCheckResult
from src.mahjong_rl.state_machine.base import GameState


class WaitRobKongState(GameState):
    """
    等待抢杠和状态

    当玩家选择补杠时，检查其他玩家是否可以抢杠和。
    逐个收集其他玩家的响应（WIN或PASS）。
    这类似于WaitResponseState，但只有两个选项：WIN或PASS。

    Attributes:
        rule_engine: 规则引擎实例
        observation_builder: 观测构建器实例
    """

    def __init__(self, rule_engine: IRuleEngine, observation_builder: IObservationBuilder):
        """
        初始化等待抢杠和状态

        Args:
            rule_engine: 规则引擎实例
            observation_builder: 观测构建器实例
        """
        super().__init__(rule_engine, observation_builder)

    def enter(self, context: GameContext) -> None:
        """
        进入等待抢杠和状态

        初始化响应收集器，检查哪些玩家可以抢杠和，
        设置响应顺序。

        Args:
            context: 游戏上下文
        """
        context.current_state = GameStateType.WAIT_ROB_KONG

        # 获取补杠的玩家和牌
        kong_player_idx = context.current_player_idx
        kong_tile = context.rob_kong_tile  # 从context获取被抢杠的牌

        # 检查哪些玩家可以抢杠和
        self.rob_kong_players = []
        for i, player in enumerate(context.players):
            if i == kong_player_idx:
                continue

            # 检查是否可以胡这张牌（将牌加入手牌后检查）
            if self._can_rob_kong(context, player, kong_tile):
                self.rob_kong_players.append(i)

        # 设置响应顺序（按照逆时针顺序，从下家开始）
        self.response_order = []
        for i in range(1, 4):
            player_idx = (kong_player_idx + i) % 4
            if player_idx in self.rob_kong_players:
                self.response_order.append(player_idx)

        # 初始化响应字典
        context.rob_kong_responses = {}

        # 设置当前响应者索引
        context.current_responder_idx = 0

        # 如果没有玩家能抢杠和，直接跳过
        if not self.response_order:
            context.should_skip_wait_rob_kong = True
        else:
            context.should_skip_wait_rob_kong = False
            # 为第一个响应者生成观测
            first_responder = self.response_order[0]
            context.current_player_idx = first_responder
            self.build_observation(context)

    def step(self, context: GameContext, action: Union[MahjongAction, str]) -> GameStateType:
        """
        收集一个玩家的抢杠和响应

        处理当前玩家的响应，移动到下一个响应者。
        当所有可抢杠和的玩家都响应后，决定下一个状态。

        Args:
            context: 游戏上下文
            action: 玩家响应动作（MahjongAction对象）或'auto'

        Returns:
            WIN 如果有玩家抢杠和成功
            GONG 如果都PASS（执行补杠）

        Raises:
            ValueError: 如果动作类型不是 MahjongAction 或 'auto'
            ValueError: 如果动作类型不是 WIN 或 PASS
            ValueError: 如果玩家不在 rob_kong_players 列表中却尝试抢杠
        """
        # 如果跳过状态，直接回到GONG执行补杠
        if hasattr(context, 'should_skip_wait_rob_kong') and context.should_skip_wait_rob_kong:
            return GameStateType.GONG

        # 获取当前响应者
        if context.current_responder_idx >= len(self.response_order):
            # 所有玩家都已响应，检查结果
            return self._check_rob_kong_result(context)

        current_responder = self.response_order[context.current_responder_idx]

        # 处理当前玩家的响应
        if action == 'auto':
            # 如果是自动模式，默认PASS
            response_action = MahjongAction(ActionType.PASS, -1)
        else:
            # ===== 新增：验证动作类型 =====
            if not isinstance(action, MahjongAction):
                raise ValueError(
                    f"WaitRobKongState expects MahjongAction or 'auto', got {type(action).__name__}"
                )

            response_action = action

            # 只允许 WIN 或 PASS 动作
            if response_action.action_type not in [ActionType.WIN, ActionType.PASS]:
                raise ValueError(
                    f"Only WIN or PASS actions allowed in WAIT_ROB_KONG state, "
                    f"got {response_action.action_type.name}"
                )

            # ===== 新增：验证抢杠条件（防御性检查）=====
            if response_action.action_type == ActionType.WIN:
                if current_responder not in self.rob_kong_players:
                    raise ValueError(
                        f"Player {current_responder} cannot rob kong: "
                        f"not in rob_kong_players list. "
                        f"Current hand: {context.players[current_responder].hand_tiles}"
                    )

        # 记录响应
        context.rob_kong_responses[current_responder] = response_action

        # 移动到下一个响应者
        context.current_responder_idx += 1

        # 如果是WIN响应，直接结束（抢杠成功）
        if response_action.action_type == ActionType.WIN:
            context.winner_ids = [current_responder]
            context.is_win = True
            context.win_way = 1  # WinWay.ROB_KONG
            return GameStateType.WIN

        # 检查是否所有玩家都已响应
        if context.current_responder_idx >= len(self.response_order):
            return self._check_rob_kong_result(context)

        # 为下一个响应者生成观测
        next_responder = self.response_order[context.current_responder_idx]
        context.current_player_idx = next_responder
        self.build_observation(context)

        # 继续收集响应
        return GameStateType.WAIT_ROB_KONG

    def exit(self, context: GameContext) -> None:
        """
        离开等待抢杠和状态

        Args:
            context: 游戏上下文
        """
        # 清理临时变量
        if hasattr(context, 'rob_kong_responses'):
            delattr(context, 'rob_kong_responses')
        if hasattr(context, 'should_skip_wait_rob_kong'):
            delattr(context, 'should_skip_wait_rob_kong')

    def _can_rob_kong(self, context: GameContext, player: PlayerData, tile: int) -> bool:
        """
        检查玩家是否可以抢杠和

        将被杠的牌临时加入玩家手牌，检查是否可以胡牌。

        Args:
            context: 游戏上下文
            player: 玩家数据
            tile: 被杠的牌

        Returns:
            True如果可以抢杠和
        """
        win_checker = WuhanMahjongWinChecker(context)

        # 临时添加牌到手牌
        temp_hand = player.hand_tiles.copy()
        temp_hand.append(tile)

        # 创建临时玩家对象
        temp_player = PlayerData(
            player_id=player.player_id,
            hand_tiles=temp_hand,
            melds=player.melds.copy(),
            special_gangs=player.special_gangs.copy()
        )

        # 检查是否可以胡牌
        win_result = win_checker.check_win(temp_player)
        return win_result.can_win

    def _check_rob_kong_result(self, context: GameContext) -> GameStateType:
        """
        检查抢杠和结果

        检查所有响应，决定下一个状态。

        Args:
            context: 游戏上下文

        Returns:
            WIN 如果有玩家抢杠和
            GONG 如果都PASS
        """
        # 检查是否有玩家WIN
        for player_idx, response in context.rob_kong_responses.items():
            if response.action_type == ActionType.WIN:
                context.winner_ids = [player_idx]
                context.is_win = True
                context.win_way = 1  # WinWay.ROB_KONG
                return GameStateType.WIN

        # 都PASS，直接执行补杠，然后进入杠后补牌状态
        player_id = context.kong_player_idx
        player = context.players[player_id]
        kong_tile = context.rob_kong_tile

        # 执行补杠逻辑
        # 找到已有的碰牌
        pong_meld = None
        for meld in player.melds:
            if meld.action_type.action_type == ActionType.PONG and meld.tiles[0] == kong_tile:
                pong_meld = meld
                break

        if pong_meld:
            # 移除碰牌
            player.melds.remove(pong_meld)
            # 从手牌移除一张牌
            player.hand_tiles.remove(kong_tile)

            # 添加补杠
            from src.mahjong_rl.core.PlayerData import Meld
            meld = Meld(
                action_type=MahjongAction(ActionType.KONG_SUPPLEMENT, kong_tile),
                tiles=[kong_tile, kong_tile, kong_tile, kong_tile],
                from_player=pong_meld.from_player
            )
            player.melds.append(meld)

            # 记录动作历史
            context.action_history.append(
                ActionRecord(
                    action_type=MahjongAction(ActionType.KONG_SUPPLEMENT, kong_tile),
                    tile=kong_tile,
                    player_id=player.player_id
                )
            )

        # 清理临时变量
        if hasattr(context, 'rob_kong_tile'):
            delattr(context, 'rob_kong_tile')
        if hasattr(context, 'kong_player_idx'):
            delattr(context, 'kong_player_idx')
        if hasattr(context, 'saved_kong_action'):
            delattr(context, 'saved_kong_action')

        # 设置杠后摸牌标记
        context.current_player_idx = player_id
        context.is_kong_draw = True
        return GameStateType.DRAWING_AFTER_GONG
