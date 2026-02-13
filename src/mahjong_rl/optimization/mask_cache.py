"""
Action Mask 缓存机制

通过缓存 action_mask 避免重复计算，仅在游戏状态真正改变时重建。
"""

from dataclasses import dataclass
from typing import Callable, Dict
import numpy as np

from src.mahjong_rl.core.GameData import GameContext


@dataclass
class MaskCacheKey:
    """
    用于判断 mask 是否需要重建的缓存键

    包含所有影响 action_mask 的游戏状态要素
    """
    player_id: int
    state: int  # GameStateType 的 value
    hand_hash: int  # 手牌哈希
    discard_hash: int  # 弃牌堆哈希
    melds_hash: int  # 副露哈希

    # ===== 新增：特殊牌信息 =====
    # Bug fix: 添加 lazy_tile 和 skin_tile，确保不同局的赖子皮子产生不同的缓存键
    lazy_tile: int  # 赖子牌 ID
    skin_tile: tuple  # 皮子牌元组（固定2张）

    @classmethod
    def from_context(cls, player_id: int, context: GameContext) -> 'MaskCacheKey':
        """
        从 GameContext 创建缓存键

        Args:
            player_id: 玩家 ID
            context: 游戏上下文

        Returns:
            MaskCacheKey 对象
        """
        player = context.players[player_id]

        # 计算手牌哈希
        hand_hash = hash(tuple(player.hand_tiles))

        # 计算弃牌堆哈希
        discard_hash = hash(tuple(context.discard_pile))

        # 计算副露哈希（考虑每张牌）
        melds_hash = hash(tuple(
            (m.action_type.action_type.value, tuple(m.tiles))
            for m in player.melds
        ))

        # ===== 新增：特殊牌哈希 =====
        lazy_tile = context.lazy_tile if context.lazy_tile is not None else -1
        skin_tile = tuple(context.skin_tile)  # 转为 tuple 以便哈希

        return cls(
            player_id=player_id,
            state=context.current_state.value,
            hand_hash=hand_hash,
            discard_hash=discard_hash,
            melds_hash=melds_hash,
            # 新增字段
            lazy_tile=lazy_tile,
            skin_tile=skin_tile
        )

    def __hash__(self) -> int:
        """使 MaskCacheKey 可用作字典键"""
        return hash((self.player_id, self.state, self.hand_hash,
                     self.discard_hash, self.melds_hash,
                     self.lazy_tile, self.skin_tile))


class ActionMaskCache:
    """
    Action Mask 缓存管理器

    缓存已计算的 action_mask，当游戏状态未改变时直接返回缓存结果。
    """

    def __init__(self):
        self._cache: Dict[MaskCacheKey, np.ndarray] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get_or_build(self, key: MaskCacheKey, builder_func: Callable) -> np.ndarray:
        """
        获取缓存或构建新 mask

        Args:
            key: 缓存键
            builder_func: 构建函数（缓存未命中时调用）

        Returns:
            np.ndarray: action_mask 数组（145 位的二进制数组）
        """
        if key in self._cache:
            self._hits += 1
            # 返回副本避免意外修改缓存
            return self._cache[key].copy()

        self._misses += 1
        mask = builder_func()
        # 存储副本
        self._cache[key] = mask.copy()
        return mask

    def clear(self):
        """
        清空缓存

        通常在 episode 结束时调用
        """
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, int | float]:
        """
        获取缓存统计信息

        Returns:
            包含 hits, misses, hit_rate 的字典
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate
        }

    def size(self) -> int:
        """返回缓存中当前存储的 mask 数量"""
        return len(self._cache)
