import random
import time
from typing import Tuple, List

import mahjong_win_checker as mjc

checker = mjc.MahjongWinChecker()

# 1. 基本胡牌（无赖子）
def test_basic_win():
    # 普通胡牌手牌（清一色筒子）
    hand = [0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8]
    success, min_laizi = checker.get_min_laizi_needed(hand, wild_index=0, unlimited_eye=True)
    print(f"普通胡牌: {success}, 最小赖子: {min_laizi}")  # 应该返回 (True, 0)

# 2. 需要赖子作为万能牌
def test_need_wildcard():
    # 缺一张牌胡牌，需要赖子作为万能牌
    hand = [0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 6, 6, 7, 8]  # 缺一张8筒
    success, min_laizi = checker.get_min_laizi_needed(hand, wild_index=1, unlimited_eye=False)
    print(f"需要万能牌: {success}, 最小赖子: {min_laizi}")  # 应该返回 (True, 1)

# 3. 赖子牌自身作为普通牌
def test_laizi_as_normal():
    # 手牌包含赖子牌，但赖子牌可以作为自身牌使用
    # 假设赖子是8筒，手牌包含1个8筒赖子，需要另一个8筒做将
    hand = [0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 5, 6, 7, 8]  # 8是赖子
    success, min_laizi = checker.get_min_laizi_needed(hand, wild_index=8, unlimited_eye=False)
    print(f"赖子自身使用: {success}, 最小赖子: {min_laizi}")  # 可能返回 (True, 0)

# 4. 混合使用场景
def test_mixed_usage():
    # 有多个赖子，部分作为自身，部分作为万能牌
    hand = [0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 10]  # 3个赖子8筒
    success, min_laizi = checker.get_min_laizi_needed(hand, wild_index=-1, unlimited_eye=False)
    print(f"混合使用: {success}, 最小赖子: {min_laizi}")  # 可能返回 (True, 1)

# 5. 无法胡牌的情况
def test_cannot_win():
    # 明显无法胡牌的手牌
    hand = [0, 0, 0, 0, 1, 1, 1, 1]  # 四个0筒，四个1筒
    success, min_laizi = checker.get_min_laizi_needed(hand, wild_index=0, unlimited_eye=False)
    print(f"无法胡牌: {success}, 最小赖子: {min_laizi}")  # 应该返回 (False, -1)

# 6. 大量赖子的情况
def test_many_laizi():
    # 手牌很差，但有很多赖子
    hand = [0, 0, 0, 0, 0, 0, 0, 0]  # 8个赖子牌
    success, min_laizi = checker.get_min_laizi_needed(hand, wild_index=0, unlimited_eye=False)
    print(f"大量赖子: {success}, 最小赖子: {min_laizi}")

# 7. 十三幺测试（需要unlimited_eye=True）
def test_thirteen_orphans():
    # 十三幺手牌（缺一张）
    hand = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]  # 缺一张
    success, min_laizi = checker.get_min_laizi_needed(hand, wild_index=0, unlimited_eye=True)
    print(f"十三幺: {success}, 最小赖子: {min_laizi}")

# 8. 七对子测试
def test_seven_pairs():
    # 七对子手牌
    hand = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    success, min_laizi = checker.get_min_laizi_needed(hand, wild_index=0, unlimited_eye=False)
    print(f"七对子: {success}, 最小赖子: {min_laizi}")


# 9. 对比新旧函数结果
def test_compare_with_old():
    hand = [0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8]  # 缺一张8筒

    # 旧函数（所有赖子都当万能牌）
    old_result = checker.is_win_hand_fast(hand, wild_index=0, unlimited_eye=False)

    # 新函数（智能使用赖子）
    new_success, min_laizi = checker.get_min_laizi_needed(hand, wild_index=0, unlimited_eye=False)

    print(f"旧函数: {old_result}, 新函数: {new_success}, 最小赖子: {min_laizi}")


# 10. 性能测试
def test_performance():
    import time

    # 复杂手牌
    hand = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]

    start = time.time()
    for _ in range(100000):
        success, min_laizi = checker.get_min_laizi_needed(hand, wild_index=0, unlimited_eye=False)
    end = time.time()

    print(f"100000次调用耗时: {end - start:.2f}秒")
    print(f"缓存大小: {checker.get_cache_size()}")


def run_all_tests():
    print("=" * 50)
    print("开始测试 get_min_laizi_needed 函数")
    print("=" * 50)

    test_basic_win()
    test_need_wildcard()
    test_laizi_as_normal()
    test_mixed_usage()
    test_cannot_win()
    test_many_laizi()
    test_thirteen_orphans()
    test_seven_pairs()
    test_compare_with_old()

    print("=" * 50)
    print("测试完成")
    print("=" * 50)

#
# if __name__ == "__main__":
#     run_all_tests()
#     # test_performance()

def build_random_data(number: int) -> Tuple[List[List[int]], List[int]]:
    """生成 number 组 14 张随机手牌 + number 个随机癞子索引"""
    hand_list = [[random.randint(0, 33) for _ in range(14)] for __ in range(number)]
    wild_list = [random.randint(0, 33) for _ in range(number)]
    return hand_list, wild_list



def benchmark(hand_list: List[List[int]],
              wild_list: List[int],
              unlimited_eye: bool = False) -> float:
    """返回总耗时（秒）"""
    checker = mjc.MahjongWinChecker()
    start = time.perf_counter()
    for hand, wild in zip(hand_list, wild_list):
        # 实际调用处
        _ = checker.get_min_laizi_needed(hand, wild_index=wild, unlimited_eye=unlimited_eye)
    end = time.perf_counter()
    print(f"Cache size: {checker.get_cache_size()}")

    # 清空缓存
    checker.clear_cache()
    print(f"Cache size after clear: {checker.get_cache_size()}")
    return end - start


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    NUMBER = 10_000_000               # 调整样本量
    hand_list, wild_list = build_random_data(NUMBER)

    print(f"开始测试，样本量：{NUMBER}")
    elapsed = benchmark(hand_list, wild_list, unlimited_eye=False)
    print(f"总耗时：{elapsed:.6f} s")
    print(f"平均单局：{elapsed/NUMBER*1000*1000:.6f} μs")
