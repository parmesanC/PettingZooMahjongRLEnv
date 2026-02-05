"""测试Tiles.get_tile_name输出"""
import sys
sys.path.insert(0, '.')

from src.mahjong_rl.core.constants import Tiles

# 测试几张牌的输出
test_tiles = [0, 5, 8, 9, 17, 18, 26, 27, 31, 32, 33]

print("当前Tiles.get_tile_name的输出:")
print("-" * 50)
for tile_id in test_tiles:
    name = Tiles.get_tile_name(tile_id)
    print(f"ID {tile_id:2d}: {name}")

print("\n" + "=" * 50)
print("问题: 返回的是枚举名称（如ONE_CHAR），而不是中文名称（如1万）")
