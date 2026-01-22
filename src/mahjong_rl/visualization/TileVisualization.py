# ==================== 独立的可视化策略类 ====================
from typing import Dict, List

from src.mahjong_rl.core.constants import Tiles


class TileTextVisualizer:
    """
    麻将牌文本可视化策略

    设计原则全面体现：
    ✅ SRP: 单一职责，只负责可视化
    ✅ OCP: 无需修改Tiles类即可扩展
    ✅ LSP: Tiles子类行为不受影响
    ✅ ISP: 提供独立的可视化接口
    ✅ DIP: 高层模块依赖抽象(可视化器)，不依赖细节
    ✅ LKP: Tiles类完全不知道可视化器的存在
    """

    def __init__(self):
        # 中文数字映射
        chinese_nums = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]

        # 初始化映射表（内部实现细节）
        self._text_map: Dict[int, str] = {
            # 万 (1-9)
            **{i: f"{chinese_nums[i]}万" for i in range(0, 9)},
            # 条 (1-9)
            **{i: f"{chinese_nums[i - 9]}条" for i in range(9, 18)},
            # 筒 (1-9)
            **{i: f"{chinese_nums[i - 18]}筒" for i in range(18, 27)},
            # 风牌
            27: "东风", 28: "南风", 29: "西风", 30: "北风",
            # 箭牌
            31: "红中", 32: "发财", 33: "白板"
        }

    def format_tile(self, tile: Tiles) -> str:
        """转换单张牌为清晰文本"""
        return self._text_map.get(tile.value, f"?{tile.value}")

    def format_hand(self, tile_ids: List[int],
                    group_by_suit: bool = True,
                    separator: str = " ") -> str:
        """
        格式化手牌，支持按花色分组

        Args:
            tile_ids: 牌ID列表
            group_by_suit: 是否按花色分组显示
            separator: 分隔符
        """
        tiles = [Tiles(tid) for tid in tile_ids]

        if not group_by_suit:
            return separator.join(self.format_tile(t) for t in tiles)

        # 按花色分组（万/条/筒/字）
        suits = {"万": [], "条": [], "筒": [], "字": []}
        for tile in sorted(tiles, key=lambda t: t.value):
            text = self.format_tile(tile)
            if "万" in text:
                suits["万"].append(text)
            elif "条" in text:
                suits["条"].append(text)
            elif "筒" in text:
                suits["筒"].append(text)
            else:
                suits["字"].append(text)

        # 只显示非空花色
        result = []
        for suit_name, cards in suits.items():
            if cards:
                result.append(f"{suit_name}: {' '.join(cards)}")

        return "  |  ".join(result)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # Tiles类完全没有被修改！

    # 创建可视化器实例
    visualizer = TileTextVisualizer()

    # 测试各种牌型
    test_tiles = [
        Tiles.ONE_CHAR,  # 1万
        Tiles.FIVE_BAM,  # 5条
        Tiles.NINE_DOT,  # 9筒
        Tiles.EAST,  # 东
        Tiles.RED_DRAGON  # 红中
    ]

    print("单张牌测试:")
    for tile in test_tiles:
        print(f"{tile.name:12} (ID:{tile.value:2d}) → {visualizer.format_tile(tile)}")

    print("\n" + "=" * 50)

    # 手牌可视化（解决字符太小问题）
    hand = [0, 2, 3, 9, 9, 13, 18, 22, 27, 28, 31, 32]

    print("\n普通列表格式:")
    print(visualizer.format_hand(hand, group_by_suit=False))

    print("\n分组格式（推荐，清晰易读）:")
    print(visualizer.format_hand(hand, group_by_suit=True))

    print("\n特殊牌型示例（七对子）:")
    seven_pairs = [0, 0, 9, 9, 18, 18, 27, 27, 31, 31, 32, 32, 33, 33]
    print(visualizer.format_hand(seven_pairs))
    