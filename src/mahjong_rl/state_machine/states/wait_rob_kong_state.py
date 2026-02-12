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
from src.mahjong_rl.core.constants import GameStateType, ActionType, WinWay
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

        # 性能优化：缓存的 WinChecker
        self._cached_win_checker = None

    def set_cached_components(self, validator=None, win_checker=None) -> None:
        """
        设置缓存的组件（由状态机调用）

        Args:
            validator: 缓存的 ActionValidator 实例（此状态不使用）
            win_checker: 缓存的 WuhanMahjongWinChecker 实例
        """
        if win_checker is not None:
            self._cached_win_checker = win_checker

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
        rob_kong_players = []
        for i, player in enumerate(context.players):
            if i == kong_player_idx:
                continue

            # 检查是否可以胡这张牌（将牌加入手牌后检查）
            if self._can_rob_kong(context, player, kong_tile):
                rob_kong_players.append(i)

        # 设置响应顺序（按照逆时针顺序，从下家开始）
        context.response_order = []
        for i in range(1, 4):
            player_idx = (kong_player_idx + i) % 4
            if player_idx in rob_kong_players:
                context.response_order.append(player_idx)

        # 设置 active_responders（用于自动跳过优化）
        context.active_responders = context.response_order.copy()
        context.active_responder_idx = 0

        # 初始化响应字典
        context.pending_responses = {}

        # 如果没有玩家能抢杠和，不生成观测
        if not context.active_responders:
            context.current_player_idx = kong_player_idx
        else:
            # 为第一个响应者生成观测
            context.current_player_idx = context.response_order[0]
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
            DRAWING_AFTER_GONG 如果都PASS（执行补杠）

        Raises:
            ValueError: 如果动作类型不是 MahjongAction 或 'auto'
            ValueError: 如果动作类型不是 WIN 或 PASS
            ValueError: 如果玩家不在 active_responders 列表中却尝试抢杠
        """
        # 【新增】优先处理自动跳过场景
        # 当 should_auto_skip() 返回 True 时，状态机会调用 step(context, 'auto')
        if action == 'auto':
            if not context.active_responders:
                # 没有玩家能抢杠，直接执行补杠逻辑
                return self._check_rob_kong_result(context)
            # 有响应者时，不应该用 'auto' 调用
            # （正常流程由状态机在 enter 后检查 should_auto_skip）
            raise ValueError(
                f"Unexpected 'auto' action with active responders. "
                f"State machine should skip this state via should_auto_skip() "
                f"when active_responders is empty."
            )

        # 获取当前响应者
        if context.active_responder_idx >= len(context.active_responders):
            # 所有玩家都已响应，检查结果
            return self._check_rob_kong_result(context)

        current_responder = context.active_responders[context.active_responder_idx]

        # 处理当前玩家的响应
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

        # 验证抢杠条件（防御性检查）
        if response_action.action_type == ActionType.WIN:
            if current_responder not in context.active_responders:
                raise ValueError(
                    f"Player {current_responder} cannot rob kong: "
                    f"not in active_responders list. "
                    f"Current hand: {context.players[current_responder].hand_tiles}"
                )

        # 记录响应
        context.pending_responses[current_responder] = response_action

        # 移动到下一个响应者
        context.active_responder_idx += 1

        # 如果是WIN响应，直接结束（抢杠成功）
        if response_action.action_type == ActionType.WIN:
            # 1. 把被抢的杠牌加入获胜者手牌
            winner = context.players[current_responder]
            winner.hand_tiles.append(context.rob_kong_tile)

            # 2. 从被抢玩家手牌中移除那张牌（保留副露中的碰）
            kong_player = context.players[context.kong_player_idx]
            if context.rob_kong_tile in kong_player.hand_tiles:
                kong_player.hand_tiles.remove(context.rob_kong_tile)

            # 3. 记录动作历史
            context.action_history.append(
                ActionRecord(
                    action_type=MahjongAction(ActionType.WIN, context.rob_kong_tile),
                    tile=context.rob_kong_tile,
                    player_id=current_responder
                )
            )

            # 4. 设置获胜信息
            context.winner_ids = [current_responder]
            context.is_win = True
            context.win_way = WinWay.ROB_KONG.value  # WinWay.ROB_KONG
            return GameStateType.WIN

        # 检查是否所有玩家都已响应
        if context.active_responder_idx >= len(context.active_responders):
            return self._check_rob_kong_result(context)

        # 为下一个响应者生成观测
        next_responder = context.active_responders[context.active_responder_idx]
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
        if hasattr(context, 'rob_kong_tile'):
            delattr(context, 'rob_kong_tile')
        if hasattr(context, 'pending_responses'):
            context.pending_responses.clear()

        # 注意：不清理 response_order 和 active_responders，
        # 因为它们是 GameContext 的标准属性

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
        # 使用缓存的 WinChecker 或降级创建新实例
        if self._cached_win_checker is not None:
            win_checker = self._cached_win_checker
        else:
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
            DRAWING_AFTER_GONG 如果都PASS（执行补杠后）
        """
        # 检查是否有玩家WIN
        for player_idx, response in context.pending_responses.items():
            if response.action_type == ActionType.WIN:
                # 执行抢杠和逻辑
                # 1. 把被抢的杠牌加入获胜者手牌
                winner = context.players[player_idx]
                winner.hand_tiles.append(context.rob_kong_tile)

                # 2. 从被抢玩家手牌中移除那张牌（保留副露中的碰）
                kong_player = context.players[context.kong_player_idx]
                if context.rob_kong_tile in kong_player.hand_tiles:
                    kong_player.hand_tiles.remove(context.rob_kong_tile)

                # 3. 记录动作历史
                context.action_history.append(
                    ActionRecord(
                        action_type=MahjongAction(ActionType.WIN, context.rob_kong_tile),
                        tile=context.rob_kong_tile,
                        player_id=player_idx
                    )
                )

                # 4. 设置获胜信息
                context.winner_ids = [player_idx]
                context.is_win = True
                context.win_way = WinWay.ROB_KONG.value
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

    def should_auto_skip(self, context: GameContext) -> bool:
        """
        检查是否应该自动跳过此状态

        如果没有玩家可以抢杠和（active_responders 为空），则自动跳过。
        这允许状态机在 transition_to() 中自动推进到下一个状态。

        设计意图：
        - 避免在 enter() 中执行状态转换逻辑
        - 由状态机统一处理自动跳过
        - 保持 enter() 的单一职责（初始化）
        - 与 WaitResponseState 保持一致的设计模式

        Args:
            context: 游戏上下文

        Returns:
            True 如果没有玩家能抢杠和（应该自动跳过）
            False 如果有玩家需要决策
        """
        return len(context.active_responders) == 0
