import mahjong_win_checker


# def test_win_checker():
#     checker = mahjong_win_checker.MahjongWinChecker()
#
#     # 测试基本胡牌
#     hand1 = [0, 0, 0, 1, 2, 3, 9, 9, 9, 27, 27, 27, 31, 31]
#     result1 = checker.is_win_hand_fast(hand1, wild_index=31, unlimited_eye=False)
#     print(f"Hand 1 (basic win): {result1}")
#
#     # 测试需要万能牌的胡牌
#     hand2 = [0, 0, 1, 2, 3, 9, 9, 27, 27, 27, 31, 31]  # 缺一张
#     result2 = checker.is_win_hand_fast(hand2, wild_index=31, unlimited_eye=False)  # 假设33是万能牌
#     print(f"Hand 2 (with wild): {result2}")
#
#     # 测试不能胡牌
#     hand3 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5]  # 不成顺子
#     result3 = checker.is_win_hand_fast(hand3, wild_index=-1, unlimited_eye=False)
#     print(f"Hand 3 (not win): {result3}")
#
#     print(f"Cache size: {checker.get_cache_size()}")
#
#     # 清空缓存
#     checker.clear_cache()
#     print(f"Cache size after clear: {checker.get_cache_size()}")
#
#
# if __name__ == "__main__":
#     test_win_checker()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机手牌性能测试：checker.is_win_hand_fast
"""
import random
import time
from typing import List, Tuple

# ------------------------------------------------------------------
# 假设 checker 模块已存在
# from checker import is_win_hand_fast
# ------------------------------------------------------------------

def build_random_data(number: int) -> Tuple[List[List[int]], List[int]]:
    """生成 number 组 14 张随机手牌 + number 个随机癞子索引"""
    hand_list = [[random.randint(0, 33) for _ in range(14)] for __ in range(number)]
    wild_list = [random.randint(0, 33) for _ in range(number)]
    return hand_list, wild_list


def benchmark(hand_list: List[List[int]],
              wild_list: List[int],
              unlimited_eye: bool = False) -> float:
    """返回总耗时（秒）"""
    checker = mahjong_win_checker.MahjongWinChecker()
    start = time.perf_counter()
    for hand, wild in zip(hand_list, wild_list):
        # 实际调用处
        _ = checker.is_win_hand_fast(hand, wild_index=wild, unlimited_eye=unlimited_eye)
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
