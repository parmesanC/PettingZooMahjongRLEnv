"""
游戏状态序列化器
将 GameContext 转换为前端可用的 JSON 格式
"""
from typing import Dict, Any, List
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType


class StateSerializer:
    """将游戏状态序列化为前端可用的格式"""

    @staticmethod
    def serialize(context: GameContext, observer_player_idx: int = 0) -> Dict[str, Any]:
        """
        将 GameContext 序列化为前端格式

        Args:
            context: 游戏上下文
            observer_player_idx: 观察者玩家索引（用于确定视角）

        Returns:
            前端可用的状态字典
        """
        return {
            'current_state': context.current_state.value if hasattr(context.current_state, 'value') else str(context.current_state),
            'current_player_idx': int(context.current_player_idx),
            'dealer_idx': int(context.dealer_idx) if context.dealer_idx is not None else 0,
            'lazy_tile': int(context.lazy_tile) if context.lazy_tile is not None else None,
            'skin_tiles': [int(t) for t in context.skin_tile] if context.skin_tile else [],
            'wall_count': len(context.wall),
            'players': [
                StateSerializer._serialize_player(p, observer_player_idx)
                for p in context.players
            ],
            'last_discarded_tile': int(context.last_discarded_tile) if context.last_discarded_tile is not None else None,
            'is_win': context.is_win,
            'is_flush': context.is_flush,
            'winner_ids': list(context.winner_ids) if context.winner_ids else []
        }

    @staticmethod
    def _serialize_player(player, observer_idx: int) -> Dict[str, Any]:
        """序列化玩家数据"""
        # 判断是否是观察者自己（决定是否显示手牌）
        is_self = player.player_id == observer_idx

        return {
            'player_id': int(player.player_id),
            'hand_tiles': [int(t) for t in player.hand_tiles] if is_self else [],
            'hand_count': len(player.hand_tiles),  # 对手只显示数量
            'melds': [
                {
                    'action_type': m.action_type.action_type.value,
                    'tiles': [int(t) for t in m.tiles],
                    'from_player': int(m.from_player)
                }
                for m in player.melds
            ],
            'discard_tiles': [int(t) for t in player.discard_tiles],
            'special_gangs': [int(x) for x in player.special_gangs],
            'is_dealer': bool(player.is_dealer),
            'is_win': bool(player.is_win)
        }
