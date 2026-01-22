#include <vector>
#include <array>
#include <unordered_map>
#include <tuple>
#include <functional>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class MahjongWinChecker {
private:
    // 常量定义
    static constexpr std::array<int8_t, 9> EYE_CANDIDATES = {1, 4, 7, 10, 13, 16, 19, 22, 25};
    static constexpr std::array<int8_t, 34> UNLIMITED_EYE_CANDIDATES = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33
    };

    // 全局缓存结构
    struct TupleHash {
        size_t operator()(const std::tuple<std::array<int8_t, 9>, int8_t>& key) const {
            size_t hash = 0;
            const auto& arr = std::get<0>(key);
            for (int i = 0; i < 9; ++i) {
                hash = hash * 31 + arr[i];
            }
            hash = hash * 31 + std::get<1>(key);
            return hash;
        }
    };

    std::unordered_map<std::tuple<std::array<int8_t, 9>, int8_t>, bool, TupleHash> melds_cache_;

public:
    MahjongWinChecker() = default;

    // 清空缓存（如果需要）
    void clear_cache() {
        melds_cache_.clear();
    }

    // 获取缓存大小（用于调试）
    size_t get_cache_size() const {
        return melds_cache_.size();
    }

private:
    bool can_form_melds_with_suit(const std::array<int8_t, 9>& tiles, int8_t wild) {
        auto key = std::make_tuple(tiles, wild);
        auto it = melds_cache_.find(key);
        if (it != melds_cache_.end()) {
            return it->second;
        }

        // 检查数字条件剪枝
        int total_tiles = 0;
        for (int count : tiles) {
            total_tiles += count;
        }

        if (total_tiles == 0) {
            melds_cache_[key] = true;
            return true;
        }

        if ((total_tiles + wild) % 3 != 0) {
            melds_cache_[key] = false;
            return false;
        }

        // 使用DFS搜索
        std::vector<std::pair<std::array<int8_t, 9>, int8_t>> stack;
        std::unordered_map<std::tuple<std::array<int8_t, 9>, int8_t>, bool, TupleHash> visited;

        stack.emplace_back(tiles, wild);

        while (!stack.empty()) {
            auto current = stack.back();
            stack.pop_back();
            auto c = current.first;
            auto w = current.second;

            // 检查是否已经处理过这个状态
            auto state_key = std::make_tuple(c, w);
            if (visited.count(state_key)) {
                continue;
            }
            visited[state_key] = true;

            // 检查是否完成
            bool all_zero = true;
            for (int count : c) {
                if (count != 0) {
                    all_zero = false;
                    break;
                }
            }
            if (all_zero) {
                melds_cache_[key] = true;
                return true;
            }

            if (w < 0) {
                continue;
            }

            // 找到第一个有牌的位置
            int i = 0;
            while (i < 9 && c[i] == 0) {
                i++;
            }

            if (i >= 9) {
                melds_cache_[key] = true;
                return true;
            }

            // 尝试刻子
            int max_k = std::min(3, static_cast<int>(c[i]));
            for (int k = 1; k <= max_k; k++) {
                int need_wild = 3 - k;
                if (need_wild <= w) {
                    auto new_c = c;
                    new_c[i] -= k;
                    stack.emplace_back(new_c, w - need_wild);
                }
            }

            // 尝试顺子
            if (i <= 6) {
                int need_wild = 0;
                auto new_c = c;

                // 计算需要的万能牌
                std::array<int, 3> positions = {i, i + 1, i + 2};
                for (int pos : positions) {
                    if (new_c[pos] > 0) {
                        new_c[pos] -= 1;
                    } else {
                        need_wild += 1;
                    }
                }

                if (need_wild <= w) {
                    stack.emplace_back(new_c, w - need_wild);
                }
            }
        }

        melds_cache_[key] = false;
        return false;
    }

    bool can_form_melds(const std::array<int8_t, 34>& count_arr, int8_t wild_count) {
        // 处理字牌（27-33）
        int8_t wild_remaining = wild_count;
        for (int i = 27; i <= 33; i++) {
            int cnt = count_arr[i];
            if (cnt == 0) {
                continue;
            } else if (cnt == 1) {
                if (wild_remaining < 2) return false;
                wild_remaining -= 2;
            } else if (cnt == 2) {
                if (wild_remaining < 1) return false;
                wild_remaining -= 1;
            } else if (cnt == 3) {
                // 不需要万能牌
            } else {
                return false;
            }
        }

        // 提取三种花色
        std::array<int8_t, 9> dots, bamboo, chars;
        for (int i = 0; i < 9; i++) {
            dots[i] = count_arr[i];
            bamboo[i] = count_arr[i + 9];
            chars[i] = count_arr[i + 18];
        }

        // 分配万能牌到各花色
        for (int w1 = 0; w1 <= wild_remaining; w1++) {
            if (!can_form_melds_with_suit(dots, w1)) {
                continue;
            }
            for (int w2 = 0; w2 <= wild_remaining - w1; w2++) {
                int w3 = wild_remaining - w1 - w2;
                if (can_form_melds_with_suit(bamboo, w2) &&
                    can_form_melds_with_suit(chars, w3)) {
                    return true;
                }
            }
        }
        return false;
    }

    bool _is_winning_core(const std::array<int8_t, 34>& count_arr, int8_t wild_count,
                         const std::vector<int8_t>& eye_candidates) {
        auto temp_count = count_arr;  // 创建副本用于修改

        for (int idx : eye_candidates) {
            // 正常将眼（对子）
            if (temp_count[idx] >= 2) {
                temp_count[idx] -= 2;
                if (can_form_melds(temp_count, wild_count)) {
                    return true;
                }
                temp_count[idx] += 2;
            }

            // 单张+万能牌作为将眼
            if (temp_count[idx] >= 1 && wild_count >= 1) {
                temp_count[idx] -= 1;
                if (can_form_melds(temp_count, wild_count - 1)) {
                    return true;
                }
                temp_count[idx] += 1;
            }

            // 双万能牌作为将眼
            if (wild_count >= 2) {
                if (can_form_melds(temp_count, wild_count - 2)) {
                    return true;
                }
            }
        }
        return false;
    }

public:
    bool is_win_hand_fast(const std::vector<int>& hand_indices, int wild_index, bool unlimited_eye) {
        // 检查手牌数量是否为3n+2（2,5,8,11,14张）
        int hand_size = hand_indices.size();
        if ((hand_size - 2) % 3 != 0 || hand_size < 2 || hand_size > 14) {
            return false; // 相公或牌数不符合胡牌要求
        }

        // 统计牌型
        std::array<int8_t, 34> count_arr = {0};
        for (int idx : hand_indices) {
            if (idx >= 0 && idx < 34) {
                count_arr[idx]++;
            }
        }

        // 处理万能牌
        int8_t wild_count = 0;
        if (wild_index >= 0 && wild_index < 34) {
            wild_count = count_arr[wild_index];
            count_arr[wild_index] = 0;
        }

        // 选择将眼候选
        std::vector<int8_t> eye_candidates;
        if (unlimited_eye) {
            eye_candidates.assign(UNLIMITED_EYE_CANDIDATES.begin(), UNLIMITED_EYE_CANDIDATES.end());
        } else {
            eye_candidates.assign(EYE_CANDIDATES.begin(), EYE_CANDIDATES.end());
        }

        return _is_winning_core(count_arr, wild_count, eye_candidates);
    }

    std::tuple<bool, int> get_min_laizi_needed(const std::vector<int>& hand_indices, int wild_index, bool unlimited_eye) {
        // 1. 检查手牌数量是否为3n+2（2,5,8,11,14张）
        int hand_size = hand_indices.size();
        if ((hand_size - 2) % 3 != 0 || hand_size < 2 || hand_size > 14) {
            return std::make_tuple(false, -1); // 相公或牌数不符合胡牌要求
        }

        // 2. 统计原始手牌，得到count_arr[34]
        std::array<int8_t, 34> count_arr = {0};
        for (int idx : hand_indices) {
            if (idx >= 0 && idx < 34) {
                count_arr[idx]++;
            }
        }

        // 3. 获取赖子牌总数
        int total_wild_tiles = 0;
        if (wild_index >= 0 && wild_index < 34) {
            total_wild_tiles = count_arr[wild_index];
        }

        // 4. 确定将的候选列表
        std::vector<int8_t> eye_candidates;
        if (unlimited_eye) {
            eye_candidates.assign(UNLIMITED_EYE_CANDIDATES.begin(), UNLIMITED_EYE_CANDIDATES.end());
        } else {
            eye_candidates.assign(EYE_CANDIDATES.begin(), EYE_CANDIDATES.end());
        }

        // 5. 核心循环：枚举当作万能牌使用的赖子数量（从小到大）
        // 注意：必须从0开始枚举到total_wild_tiles（包含两端）
        for (int used_wild = 0; used_wild <= total_wild_tiles; ++used_wild) {
            // used_wild: 当作万能牌使用的赖子数量
            // total_wild_tiles - used_wild: 当作普通牌（自身还原）的赖子数量

            // 构建临时手牌状态：
            // 复制原始手牌统计
            std::array<int8_t, 34> temp_counts = count_arr;

            // 重要修正：当赖子牌全部当作万能牌使用时，手牌中不应再包含赖子牌
            // 当部分赖子牌当作普通牌使用时，这些赖子牌作为普通牌留在手牌中
            if (wild_index >= 0 && wild_index < 34) {
                if (used_wild == total_wild_tiles) {
                    // 所有赖子都当作万能牌，手牌中不再有赖子牌
                    temp_counts[wild_index] = 0;
                } else {
                    // 部分赖子当作普通牌，保留在手牌中
                    temp_counts[wild_index] = total_wild_tiles - used_wild;
                }
            }

            // 此时手牌状态为：
            // - temp_counts: 包含还原为普通牌的赖子（如果有）
            // - used_wild: 当作万能牌的数量

            // 调用现有的胡牌判定逻辑
            if (_is_winning_core(temp_counts, used_wild, eye_candidates)) {
                // 由于used_wild是从小到大枚举，第一个找到的就是最小赖子使用量
                return std::make_tuple(true, used_wild);
            }
        }

        // 循环结束仍未找到可行方案，说明无法胡牌
        return std::make_tuple(false, -1);
    }
};

// Python绑定
PYBIND11_MODULE(mahjong_win_checker, m) {
    py::class_<MahjongWinChecker>(m, "MahjongWinChecker")
        .def(py::init<>())
        .def("is_win_hand_fast", &MahjongWinChecker::is_win_hand_fast,
             "Fast check for winning hand",
             py::arg("hand_indices"),
             py::arg("wild_index") = -1,
             py::arg("unlimited_eye") = false)
        .def("get_min_laizi_needed", &MahjongWinChecker::get_min_laizi_needed,
             "Get minimum laizi needed for winning hand",
             py::arg("hand_indices"),
             py::arg("wild_index"),
             py::arg("unlimited_eye") = false,
             py::return_value_policy::automatic)
        .def("clear_cache", &MahjongWinChecker::clear_cache, "Clear cache")
        .def("get_cache_size", &MahjongWinChecker::get_cache_size, "Get cache size");
}