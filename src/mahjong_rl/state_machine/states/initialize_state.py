from collections import deque
import random
from typing import Optional

from src.mahjong_rl.rules.wuhan_7p4l_rule_engine import Wuhan7P4LRuleEngine
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.PlayerData import PlayerData
from src.mahjong_rl.core.constants import GameStateType, Tiles
from src.mahjong_rl.observation.SimpleObservationBuilder import SimpleObservationBuilder
from src.mahjong_rl.state_machine.base import GameState


class InitialState(GameState):
    """初始状态"""

    def enter(self, context: GameContext) -> None:
        """进入状态：初始化游戏基础数据"""
        # 重置上下文状态
        context.is_win = False
        context.is_flush = False
        context.winner_ids = []
        context.reward = 0.0

        context.current_state = GameStateType.INITIAL
        # 自动状态不需要生成观测和动作掩码
        context.observation = None
        context.action_mask = None

        # 弃牌相关
        context.discard_pile = []
        context.last_discarded_tile = None
        context.last_drawn_tile = None
        context.pending_discard_tile = None

        # 响应相关
        context.pending_responses = {}
        context.response_order = []
        context.current_responder_idx = 0
        context.selected_responder = None
        context.response_priorities = {}

        # 杠牌相关
        context.last_kong_action = None

        # 轮庄相关
        context.next_dealer_idx = None
        context.retain_dealer_count = 0


    def step(self, context: GameContext, action: Optional[tuple] = None) -> GameStateType:
        """初始化状态：洗牌、发牌、定庄、生成特殊牌（赖子/皮子）"""
        # 1. 确定庄家位置
        if context.round_info.total_rounds_played == 0:
            # 新游戏：随机确定庄家
            context.dealer_idx = random.randint(0, 3)
        else:
            # 轮局：从 round_info 获取庄家位置（已在游戏结束时更新）
            context.dealer_idx = context.round_info.dealer_position

        # 2. 洗牌
        base_tiles = Tiles.get_all_tile_ids()
        context.wall = deque(base_tiles * 4)
        random.shuffle(context.wall)

        # 3. 发牌（每人13张）
        for player in context.players:
            player.hand_tiles = [context.wall.popleft() for _ in range(13)]

        # 4. 设置庄家标志（重置所有玩家）
        for i in range(4):
            context.players[i].is_dealer = (i == context.dealer_idx)
        context.current_player_idx = context.dealer_idx

        # 5. 庄家额外摸一张牌（14张）
        dealer_draw_tile = context.wall.popleft()
        context.players[context.current_player_idx].hand_tiles.append(dealer_draw_tile)
        # 存储摸到的牌供PLAYER_DECISION状态使用
        context.last_drawn_tile = dealer_draw_tile

        # 生成特殊牌（赖子和皮子）
        context.lazy_tile, context.skin_tile[0], context.skin_tile[1] = Tiles.generate_special_tiles(
            context.wall.popleft())

        # 6. 更新 special_tiles 元组
        context.special_tiles = (
            context.lazy_tile,
            context.skin_tile[0],
            context.skin_tile[1],
            context.red_dragon
        )

        for i in range(4):
            context.players[i].special_gangs = [0, 0, 0]

        context.current_player_idx = context.dealer_idx
        return GameStateType.PLAYER_DECISION


    def exit(self, context: GameContext) -> None:
        """离开初始状态"""
        # log 等后续实现
        pass



if __name__ == "__main__":
    # 创建游戏上下文
    context_init = GameContext()
    context_init.players = [PlayerData(player_id=i) for i in range(4)]
    context_init.round_info.dealer_position = 0  # 设置庄家位置

    # 进入初始状态
    builder = SimpleObservationBuilder()
    engine = Wuhan7P4LRuleEngine(context_init)
    initial_state = InitialState(rule_engine=engine, observation_builder=builder)
    initial_state.enter(context_init)
    initial_state.step(context_init)


    from src.mahjong_rl.visualization import TileVisualization

    visualizer = TileVisualization.TileTextVisualizer()
    # 打印初始状态信息
    print(f"庄家索引: {context_init.dealer_idx}")
    print(f"当前玩家: {context_init.current_player_idx}")
    print(f"赖子牌: {visualizer.format_tile(Tiles(context_init.lazy_tile))}")
    print(f"皮子牌: {[visualizer.format_tile(Tiles(tile)) for tile in context_init.skin_tile]}")
    print(f"牌墙: {visualizer.format_hand(list(context_init.wall), group_by_suit=False)}，牌墙剩余数量: {len(context_init.wall)}")


    for i in range(4):
        hand_tiles = sorted(context_init.players[i].hand_tiles)

        print(f"玩家{i}手牌: {visualizer.format_hand(hand_tiles, group_by_suit=False)}")


