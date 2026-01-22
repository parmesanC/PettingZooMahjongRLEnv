from enum import Enum, auto
from functools import lru_cache
from typing import List, Tuple, ClassVar, Dict, Final

CARDS_PER_SUIT: Final[int] = 9
NUM_SUIT_TYPES: Final[int] = 3  # 万、筒、条
NUM_SUIT_CARDS: Final[int] = CARDS_PER_SUIT * NUM_SUIT_TYPES  # 27
NUM_HONOR_CARDS: Final[int] = 7  # 4风 + 3箭
TOTAL_TILES: Final[int] = NUM_SUIT_CARDS + NUM_HONOR_CARDS  # 34

# 自摸、抢杠、杠开、点炮
class WinWay(Enum):
    """胡牌方式"""
    SELF_DRAW = 0  # 自摸
    ROB_KONG = 1  # 抢杠
    KONG_SELF_DRAW = 2  # 杠开
    DISCARD = 3  # 点炮

class WinType(Enum):
    """胡牌类型"""
    # 小胡
    SOFT_WIN = 0  # 软胡
    HARD_WIN = 1  # 硬胡

    # 大胡
    PURE_FLUSH = 2  # 清一色
    PENG_PENG_HU = 3  # 碰碰胡
    ALL_WIND = 4  # 风一色
    ALL_JIANG = 5  # 将一色
    FULLY_MELDED = 6  # 全求人
    FLOWER_ON_KONG = 7  # 杠上花
    LAST_TILE_WIN = 8  # 海底捞月
    ROB_KONG = 9  # 抢杠胡


class ActionType(Enum):
    """
    麻将 agent 可选动作（参数化动作+双头网络）
    基础动作类型（第一层决策）
    """
    DISCARD = 0  # 出牌
    CHOW = 1  # 吃
    PONG = 2  # 碰
    KONG_EXPOSED = 3  # 明杠
    KONG_SUPPLEMENT = 4  # 补杠
    KONG_CONCEALED = 5  # 暗杠
    KONG_RED = 6  # 红中杠
    KONG_SKIN = 7  # 皮子杠
    KONG_LAZY = 8  # 赖子杠
    WIN = 9  # 胡牌
    PASS = 10  # 过牌


class ChowType(Enum):
    """吃牌类型"""
    LEFT = 0  # 吃左
    MIDDLE = 1  # 吃中
    RIGHT = 2  # 吃右


#
# class ActionType(Enum):
#     """"麻将 agent 可选动作"""
#     # 0-33: 打出对应牌
#     DISCARD_ONE_CHAR = 0
#     DISCARD_TWO_CHAR = 1
#     DISCARD_THREE_CHAR = 2
#     DISCARD_FOUR_CHAR = 3
#     DISCARD_FIVE_CHAR = 4
#     DISCARD_SIX_CHAR = 5
#     DISCARD_SEVEN_CHAR = 6
#     DISCARD_EIGHT_CHAR = 7
#     DISCARD_NINE_CHAR = 8
#
#     DISCARD_ONE_BAM = 9
#     DISCARD_TWO_BAM = 10
#     DISCARD_THREE_BAM = 11
#     DISCARD_FOUR_BAM = 12
#     DISCARD_FIVE_BAM = 13
#     DISCARD_SIX_BAM = 14
#     DISCARD_SEVEN_BAM = 15
#     DISCARD_EIGHT_BAM = 16
#     DISCARD_NINE_BAM = 17
#
#     DISCARD_ONE_DOT = 18
#     DISCARD_TWO_DOT = 19
#     DISCARD_THREE_DOT = 20
#     DISCARD_FOUR_DOT = 21
#     DISCARD_FIVE_DOT = 22
#     DISCARD_SIX_DOT = 23
#     DISCARD_SEVEN_DOT = 24
#     DISCARD_EIGHT_DOT = 25
#     DISCARD_NINE_DOT = 26
#
#     DISCARD_EAST = 27
#     DISCARD_SOUTH = 28
#     DISCARD_WEST = 29
#     DISCARD_NORTH = 30
#     DISCARD_RED = 31
#     DISCARD_GREEN = 32
#     DISCARD_WHITE = 33
#
#     # 34-46: 特殊动作
#     CHOW_LEFT = 34  # 吃左
#     CHOW_MIDDLE = 35  # 吃中
#     CHOW_RIGHT = 36  # 吃右
#     PONG = 37  # 碰
#     KONG_EXPOSED = 38  # 明杠
#     KONG_SUPPLEMENT = 39  # 补杠
#     KONG_CONCEALED = 40  # 暗杠
#     KONG_RED = DISCARD_RED  # 红中杠
#     KONG_SKIN = 41  # 皮子杠
#     KONG_LAZY = 42  # 赖子杠
#     WIN = 43  # 胡
#     PASS = 44  # 过
#
#     @property
#     def is_discard_action(self) -> bool:
#         """是否为出牌动作"""
#         return 0 <= self.value <= 33
#
#     @property
#     def is_chow_action(self) -> bool:
#         """是否为吃牌动作"""
#         return 34 <= self.value <= 36
#
#     @property
#     def is_kong_action(self) -> bool:
#         """是否为杠牌动作"""
#         return 38 <= self.value <= 40
#
#     @property
#     def is_pon_action(self) -> bool:
#         """是否为碰牌动作"""
#         return self.value == 37
#
#
# # # --- 1. 定义常量，消除魔数 ---
# # NUM_SUIT_CARDS = 27  # 数字牌总数 (3种花色 * 9张)
# # CARDS_PER_SUIT = 9  # 每种花色的牌数


class TileMappings:
    """
    牌映射数据容器，遵循分离关注点原则。
    所有映射使用整数ID而非枚举实例，避免循环引用和初始化顺序问题。
    """

    # 赖子映射：字牌循环（东风 -> 南风 -> 西风 -> 北风 -> 发财 -> 白板 -> 红中 -> 东风）
    WIND_DRAGON_MAP: ClassVar[Dict[int, int]] = {
        27: 28,  # EAST -> SOUTH
        28: 29,  # SOUTH -> WEST
        29: 30,  # WEST -> NORTH
        30: 32,  # NORTH -> GREEN_DRAGON
        31: 32,  # RED_DRAGON -> GREEN_DRAGON
        32: 33,  # GREEN_DRAGON -> WHITE_DRAGON
        33: 27,  # WHITE_DRAGON -> EAST
    }

    # 皮子映射：每张字牌对应的"皮子"牌列表
    SKIN_MAP: ClassVar[Dict[int, List[int]]] = {
        27: [27, 33],  # EAST -> [EAST, WHITE_DRAGON]
        28: [28, 27],  # SOUTH -> [SOUTH, EAST]
        29: [29, 28],  # WEST -> [WEST, SOUTH]
        30: [30, 29],  # NORTH -> [NORTH, WEST]
        31: [30, 29],  # RED_DRAGON -> [NORTH, WEST] (跳过自身)
        32: [32, 30],  # GREEN_DRAGON -> [GREEN_DRAGON, NORTH]
        33: [33, 32],  # WHITE_DRAGON -> [WHITE_DRAGON, GREEN_DRAGON]
    }


# 牌中文名称映射表（模块级别常量）
_TILE_CHINESE_NAMES: Dict[int, str] = {
    # 万子
    0: "1万", 1: "2万", 2: "3万", 3: "4万", 4: "5万", 5: "6万", 6: "7万", 7: "8万", 8: "9万",
    # 条子
    9: "1条", 10: "2条", 11: "3条", 12: "4条", 13: "5条", 14: "6条", 15: "7条", 16: "8条", 17: "9条",
    # 筒子
    18: "1筒", 19: "2筒", 20: "3筒", 21: "4筒", 22: "5筒", 23: "6筒", 24: "7筒", 25: "8筒", 26: "9筒",
    # 字牌
    27: "东风", 28: "南风", 29: "西风", 30: "北风", 31: "红中", 32: "发财", 33: "白板",
}

# ==================== 枚举类 ====================
class Tiles(TileMappings, Enum):
    """
    麻将牌枚举类。

    设计原则：
    1. 单一职责：仅定义牌的ID和基础行为
    2. 开放封闭：通过Mixin扩展映射逻辑
    3. 里氏替换：所有方法对子类同样适用
    """

    # ==================== 数字牌 ====================
    # 万
    ONE_CHAR = 0
    TWO_CHAR = 1
    THREE_CHAR = 2
    FOUR_CHAR = 3
    FIVE_CHAR = 4
    SIX_CHAR = 5
    SEVEN_CHAR = 6
    EIGHT_CHAR = 7
    NINE_CHAR = 8
    # 条
    ONE_BAM = 9
    TWO_BAM = 10
    THREE_BAM = 11
    FOUR_BAM = 12
    FIVE_BAM = 13
    SIX_BAM = 14
    SEVEN_BAM = 15
    EIGHT_BAM = 16
    NINE_BAM = 17
    # 筒
    ONE_DOT = 18
    TWO_DOT = 19
    THREE_DOT = 20
    FOUR_DOT = 21
    FIVE_DOT = 22
    SIX_DOT = 23
    SEVEN_DOT = 24
    EIGHT_DOT = 25
    NINE_DOT = 26
    # 风
    EAST = 27
    SOUTH = 28
    WEST = 29
    NORTH = 30
    # 箭牌
    RED_DRAGON = 31
    GREEN_DRAGON = 32
    WHITE_DRAGON = 33

    # ==================== 类方法 ====================
    @classmethod
    def get_all_tile_ids(cls) -> List[int]:
        """
        获取所有基础牌ID。
        使用动态计算，避免硬编码。
        """
        return list(range(TOTAL_TILES))

    @classmethod
    def is_number_tile(cls, tile_id: int) -> bool:
        """判断是否为数字牌（万/筒/条）"""
        return 0 <= tile_id < NUM_SUIT_CARDS

    @classmethod
    def is_honor_tile(cls, tile_id: int) -> bool:
        """判断是否为字牌（风/箭）"""
        return NUM_SUIT_CARDS <= tile_id < TOTAL_TILES

    @classmethod
    def get_suit_base(cls, tile_id: int) -> int:
        """获取花色基准值（每花色的第一个ID）"""
        if not cls.is_number_tile(tile_id):
            raise ValueError(f"字牌没有花色: {tile_id}")
        return (tile_id // CARDS_PER_SUIT) * CARDS_PER_SUIT

    @classmethod
    def get_suit(cls, tile_id: int) -> int:
        """获取花色（0=万, 1=条, 2=筒, 3=字）"""
        if tile_id in range(0, 9):
            return 0
        elif tile_id in range(9, 18):
            return 1
        elif tile_id in range(18, 27):
            return 2
        elif tile_id in range(27, 34):
            return 3
        else:
            raise ValueError(f"无效的牌ID: {tile_id}")


    @classmethod
    @lru_cache()
    def generate_special_tiles(cls, first_tile: int) -> Tuple[int, int, int]:
        """
        生成特殊牌（赖子和皮子）。

        赖子规则：
        - 数字牌：同花色的下一张牌（九循环到一）
        - 字牌：按WIND_DRAGON_MAP循环

        皮子规则：
        - 数字牌：当前牌 + 前一张牌
        - 字牌：按SKIN_MAP配置

        :param first_tile: 首张牌ID
        :return: (赖子ID, 皮子ID元组)
        :raises ValueError: 当牌ID无效时
        """
        if not (0 <= first_tile < TOTAL_TILES):
            raise ValueError(f"无效的牌ID: {first_tile}")

        if cls.is_number_tile(first_tile):
            return cls._generate_number_special_tiles(first_tile)
        else:
            return cls._generate_honor_special_tiles(first_tile)

    @classmethod
    def _generate_number_special_tiles(cls, first_tile: int) -> Tuple[int, int, int]:
        """数字牌特殊牌生成（私有方法）"""
        suit_base = cls.get_suit_base(first_tile)
        num_in_suit = first_tile % CARDS_PER_SUIT

        # 赖子：下一张牌（循环）
        lai_zi_num = (num_in_suit + 1) % CARDS_PER_SUIT
        lai_zi = lai_zi_num + suit_base

        # 皮子：当前牌 + 前一张牌（循环）
        prev_num = (num_in_suit - 1) % CARDS_PER_SUIT
        pi_zi1 = first_tile
        pi_zi2 = prev_num + suit_base

        return lai_zi, pi_zi1, pi_zi2

    @classmethod
    def _generate_honor_special_tiles(cls, first_tile: int) -> Tuple[int, int, int]:
        """字牌特殊牌生成（私有方法）"""
        # 赖子：直接查表
        lai_zi = cls.WIND_DRAGON_MAP[first_tile]

        # 皮子：过滤红中（业务规则）
        raw_skins = cls.SKIN_MAP[first_tile]
        skin_tiles = tuple(tile for tile in raw_skins if tile != cls.RED_DRAGON.value)

        return lai_zi, skin_tiles[0], skin_tiles[1]

    @classmethod
    def get_tile_name(cls, tile_id: int) -> str:
        """
        通过ID获取牌的中文名称

        Args:
            tile_id: 牌ID (0-33)

        Returns:
            中文名称，如 "1万", "红中" 等

        Raises:
            ValueError: 当牌ID无效时
        """
        if not (0 <= tile_id < TOTAL_TILES):
            raise ValueError(f"无效的牌ID: {tile_id}")
        return _TILE_CHINESE_NAMES[tile_id]

    def __repr__(self) -> str:
        """更友好的对象表示"""
        return f"{self.__class__.__name__}.{self.name}({self.value})"


class GameStateType(Enum):
    """强化学习专用状态机状态"""
    INITIAL = auto()  # 初始化
    DRAWING = auto()  # 摸牌阶段
    PLAYER_DECISION = auto()  # 摸牌后决策（可杠/胡/出牌）
    DISCARDING = auto()  # 打牌等待响应
    MELD_DECISION = auto()  # 鸣牌后决策（可杠/出牌，不能胡）
    WAITING_RESPONSE = auto()  # 等待响应
    RESPONSES = auto()  # 响应阶段
    PROCESSING_MELD = auto()  # 吃碰杠声明
    GONG = auto()  # 杠牌处理
    DRAWING_AFTER_GONG = auto()  # 杠后补牌
    RESPONSES_AFTER_GONG = auto()  # 杠后响应
    WAIT_ROB_KONG = auto()  # 等待抢杠和决策
    WIN = auto()  # 胡牌
    FLOW_DRAW = auto()  # 荒牌流局
    SETTLE = auto()  # 结算


class ResponsePriority(Enum):
    """响应优先级枚举"""
    WIN = 0  # 和牌最高
    KONG = 1  # 杠
    PONG = 2  # 碰
    CHOW = 3  # 吃
    PASS = 4  # 过


ACTION_SPACE_SIZE = len(ActionType)
#
# # 首张牌是 WHITE_DRAGON (ID=33)
# lai_zi_id, skin_id_1, skin_id_2 = Tiles.generate_special_tiles(Tiles.WHITE_DRAGON.value)
#
# print(f"赖子ID: {lai_zi_id}")  # 预期输出: 赖子ID: 27 (EAST)
# print(f"皮子ID: {skin_id_1}, {skin_id_2}")  # 预期输出: 皮子ID: 33, 32 (WHITE_DRAGON, GREEN_DRAGON)

# ==================== 单元测试 ====================
if __name__ == "__main__":
    def test_tiles():
        """全面的单元测试"""
        print("=" * 50)
        print("麻将牌枚举系统测试")
        print("=" * 50)

        # 测试1：基础属性
        assert len(Tiles) == TOTAL_TILES, f"枚举成员数量应为{TOTAL_TILES}"
        assert Tiles.ONE_CHAR.value == 0
        assert Tiles.WHITE_DRAGON.value == 33
        print("✓ 基础枚举值正确")

        # 测试2：牌类型判断
        assert Tiles.is_number_tile(0) is True  # 一万
        assert Tiles.is_number_tile(26) is True  # 九条
        assert Tiles.is_number_tile(27) is False  # 东风
        assert Tiles.is_honor_tile(31) is True  # 红中
        print("✓ 牌类型判断正确")

        # 测试3：数字牌赖子/皮子
        lai_zi, skin1, skin2 = Tiles.generate_special_tiles(0)  # 一万
        assert lai_zi == 1  # 赖子：二万
        assert skin1 == 0  # 皮子：一万
        assert skin2 == 8  # 皮子：九万
        print(
            f"✓ 数字牌赖子/皮子：一万 -> 赖子={Tiles.get_tile_name(lai_zi)}, 皮子={[Tiles.get_tile_name(tile) for tile in (skin1, skin2)]}")

        lai_zi, skin1, skin2 = Tiles.generate_special_tiles(8)  # 九万
        assert lai_zi == 0  # 赖子：一万
        assert skin1 == 8  # 皮子：九万
        assert skin2 == 7  # 皮子：八万
        print(f"✓ 数字牌循环正确：九万 -> 赖子={Tiles.get_tile_name(lai_zi)}")

        # 测试4：字牌赖子
        lai_zi, skin1, skin2 = Tiles.generate_special_tiles(27)  # 东风
        assert lai_zi == 28  # 南风
        assert skin1 == 27  # 东风
        assert skin2 == 33  # 白板
        print(f"✓ 字牌赖子：东风 -> 赖子={Tiles.get_tile_name(lai_zi)}")

        # 测试5：字牌皮子
        lai_zi, skin1, skin2 = Tiles.generate_special_tiles(31)  # 红中
        assert skin1 == 30  # 北风
        assert skin2 == 29  # 西风
        print(f"✓ 红中皮子规则：皮子={[Tiles.get_tile_name(tile) for tile in (skin1, skin2)]}")

        # 测试6：缓存功能
        import time
        start = time.perf_counter()
        for _ in range(10000):
            Tiles.generate_special_tiles(5)
        cached_time = time.perf_counter() - start
        print(f"✓ 缓存性能：10000次调用耗时 {cached_time:.4f}秒")

        # 测试7：错误处理
        try:
            Tiles.generate_special_tiles(99)
            assert False, "应抛出ValueError"
        except ValueError as e:
            assert "无效的牌ID" in str(e)
            print("✓ 异常处理正确")

        print("\n" + "=" * 50)
        print("所有测试通过！")
        print("=" * 50)


    test_tiles()
