# 可见度掩码修复实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 修复 `example_mahjong_env.py` 中的 `_apply_visibility_mask` 方法，实现课程学习三阶段（全知→渐进掩码→真实状态）

**架构:** 通过 `training_phase` 和 `phase2_progress` 参数控制信息可见度，使用 S 曲线实现渐进式掩码过渡

**技术栈:** Python, NumPy, PettingZoo AECEnv

---

## 背景说明

当前 `_apply_visibility_mask` 方法逻辑与设计意图完全相反：

**当前错误实现：**
- Phase 1: 限制信息最多（只保留私有手牌）
- Phase 2/3: 屏蔽对手手牌

**正确设计意图：**
- **阶段1（全知视角）**：所有信息可见，包括对手手牌、牌墙、暗杠等
- **阶段2（渐进式）**：随机增加掩码，通过 S 曲线随 `phase2_progress` 参数逐渐过渡
- **阶段3（真实状态）**：只可见自己手牌和公共信息

**特殊值约定：**
- `global_hand` 对手手牌：**5**（观测空间限制为 [6]，5 表示未知）
- `wall`：**34**（已有）
- `melds.tiles` 暗杠牌：**34**（与 wall 一致）

---

## Task 1: 添加 phase2_progress 参数

**Files:**
- Modify: `example_mahjong_env.py:63-90` (WuhanMahjongEnv.__init__)

**Step 1: 修改 __init__ 方法签名**

在 `training_phase` 参数后添加 `phase2_progress` 参数：

```python
def __init__(
    self,
    render_mode=None,
    training_phase=3,
    phase2_progress=0.0,  # 新增：阶段2的课程学习进度（0.0-1.0）
    enable_logging=True,
    log_config=None,
    logger=None,
    enable_perf_monitor=False
):
```

**Step 2: 在 __init__ 方法体中初始化参数**

找到 `self.training_phase = training_phase` 行，在其后添加：

```python
self.training_phase = training_phase
self.phase2_progress = max(0.0, min(1.0, float(phase2_progress)))  # 限制在0-1范围
```

**Step 3: 运行测试验证环境可正常创建**

```bash
"D:\DATA\Development\Anaconda\envs\PettingZooRLMahjong\python.exe" -c "from example_mahjong_env import WuhanMahjongEnv; env = WuhanMahjongEnv(training_phase=2, phase2_progress=0.5); print('phase2_progress:', env.phase2_progress)"
```

Expected: `phase2_progress: 0.5`

**Step 4: 提交**

```bash
git add example_mahjong_env.py
git commit -m "feat: add phase2_progress parameter for curriculum learning"
```

---

## Task 2: 添加 S 曲线概率计算方法

**Files:**
- Modify: `example_mahjong_env.py` (在 _apply_visibility_mask 方法前添加)

**Step 1: 在 _apply_visibility_mask 方法前添加新方法**

在 `_apply_visibility_mask` 方法（约660行）之前添加：

```python
def _get_masking_probability(self) -> float:
    """
    根据progress计算S曲线掩码概率

    S曲线特性：
    - progress=0.0 时，概率接近 0（接近阶段1全知）
    - progress=0.5 时，概率=0.5（过渡中点）
    - progress=1.0 时，概率接近 1（接近阶段3完全掩码）

    Returns:
        掩码概率 (0.0-1.0)
    """
    if self.training_phase != 2:
        return 0.0 if self.training_phase == 1 else 1.0

    # S曲线：sigmoid函数，6*(x-0.5) 让曲线在中间变化更明显
    import math
    sigmoid = 1 / (1 + math.exp(-6 * (self.phase2_progress - 0.5)))
    return sigmoid
```

**Step 2: 验证方法在不同阶段的输出**

```bash
"D:\DATA\Development\Anaconda\envs\PettingZooRLMahjong\python.exe" -c "
from example_mahjong_env import WuhanMahjongEnv

# 测试阶段1
env1 = WuhanMahjongEnv(training_phase=1)
print('Phase 1 probability:', env1._get_masking_probability())

# 测试阶段2，不同progress
env2_0 = WuhanMahjongEnv(training_phase=2, phase2_progress=0.0)
print('Phase 2, progress=0.0:', env2_0._get_masking_probability())

env2_5 = WuhanMahjongEnv(training_phase=2, phase2_progress=0.5)
print('Phase 2, progress=0.5:', env2_5._get_masking_probability())

env2_1 = WuhanMahjongEnv(training_phase=2, phase2_progress=1.0)
print('Phase 2, progress=1.0:', env2_1._get_masking_probability())

# 测试阶段3
env3 = WuhanMahjongEnv(training_phase=3)
print('Phase 3 probability:', env3._get_masking_probability())
"
```

Expected:
```
Phase 1 probability: 0.0
Phase 2, progress=0.0: ~0.047 (接近0)
Phase 2, progress=0.5: 0.5
Phase 2, progress=1.0: ~0.953 (接近1)
Phase 3 probability: 1.0
```

**Step 3: 提交**

```bash
git add example_mahjong_env.py
git commit -m "feat: add S-curve masking probability calculation"
```

---

## Task 3: 重写 _apply_visibility_mask 方法

**Files:**
- Modify: `example_mahjong_env.py:660-697` (_apply_visibility_mask 方法)

**Step 1: 完全替换 _apply_visibility_mask 方法**

将现有的 `_apply_visibility_mask` 方法（660-697行）完全替换为：

```python
def _apply_visibility_mask(self, observation: Dict[str, np.ndarray], agent_id: int) -> Dict[str, np.ndarray]:
    """
    应用信息可见度掩码（课程学习）

    阶段设计：
    - 阶段1（全知视角）：所有信息可见
    - 阶段2（渐进式）：随progress逐渐增加掩码
    - 阶段3（真实状态）：只可见自己手牌和公共信息

    特殊值约定：
    - global_hand: 5 表示"未知"
    - wall: 34 表示"未知"
    - melds.tiles: 暗杠的牌设为 34（与wall一致）

    Args:
        observation: 原始观测
        agent_id: 当前agent ID

    Returns:
        掩码后的观测
    """
    if self.training_phase == 1:
        # 阶段1：全知视角，不做任何掩码
        pass

    elif self.training_phase == 2:
        # 阶段2：渐进式随机掩码
        mask_prob = self._get_masking_probability()

        # 按概率掩码对手手牌
        if np.random.random() < mask_prob:
            global_hand = observation['global_hand'].copy()
            for i in range(4):
                if i != agent_id:
                    global_hand[i * 34:(i + 1) * 34] = 5
            observation['global_hand'] = global_hand

        # 按概率掩码牌墙
        if np.random.random() < mask_prob:
            observation['wall'].fill(34)

        # 按概率掩码对手暗杠的牌
        if np.random.random() < mask_prob:
            tiles = observation['melds']['tiles'].copy()
            action_types = observation['melds']['action_types']
            KONG_CONCEALED = 5  # ActionType.KONG_CONCEALED.value

            for player_id in range(4):
                if player_id == agent_id:
                    continue
                for meld_idx in range(4):
                    idx = player_id * 4 + meld_idx
                    if action_types[idx] == KONG_CONCEALED:
                        # 将这个暗杠的4张牌位置设为34
                        base_tile_idx = (player_id * 4 * 4 + meld_idx * 4) * 34
                        for tile_pos in range(4):
                            tile_start = base_tile_idx + tile_pos * 34
                            tiles[tile_start:tile_start + 34] = 34
            observation['melds']['tiles'] = tiles

    elif self.training_phase == 3:
        # 阶段3：真实状态，完全掩码
        # 掩码对手手牌
        global_hand = observation['global_hand'].copy()
        for i in range(4):
            if i != agent_id:
                global_hand[i * 34:(i + 1) * 34] = 5
        observation['global_hand'] = global_hand

        # 掩码牌墙
        observation['wall'].fill(34)

        # 掩码对手暗杠的牌
        tiles = observation['melds']['tiles'].copy()
        action_types = observation['melds']['action_types']
        KONG_CONCEALED = 5

        for player_id in range(4):
            if player_id == agent_id:
                continue
            for meld_idx in range(4):
                idx = player_id * 4 + meld_idx
                if action_types[idx] == KONG_CONCEALED:
                    base_tile_idx = (player_id * 4 * 4 + meld_idx * 4) * 34
                    for tile_pos in range(4):
                        tile_start = base_tile_idx + tile_pos * 34
                        tiles[tile_start:tile_start + 34] = 34
        observation['melds']['tiles'] = tiles

    return observation
```

**Step 2: 提交**

```bash
git add example_mahjong_env.py
git commit -m "fix: rewrite _apply_visibility_mask with correct curriculum learning logic"
```

---

## Task 4: 编写集成测试验证三阶段行为

**Files:**
- Create: `tests/integration/test_visibility_mask.py`

**Step 1: 创建测试文件**

```python
"""
测试可见度掩码的三阶段课程学习
"""
import numpy as np
from example_mahjong_env import WuhanMahjongEnv


def test_phase_1_omniscient():
    """阶段1：全知视角，所有信息应可见"""
    env = WuhanMahjongEnv(training_phase=1)
    obs, _ = env.reset(seed=42)

    # 验证对手手牌可见（不全为5）
    for agent_id in range(4):
        obs = env.observe(f"player_{agent_id}")
        global_hand = obs['global_hand']
        for i in range(4):
            if i != agent_id:
                player_hand = global_hand[i * 34:(i + 1) * 34]
                # 应该有非5的值（实际牌数）
                assert not np.all(player_hand == 5), \
                    f"Phase 1: Agent {agent_id} should see opponent {i}'s hand"

    # 验证牌墙可见（不全为34）
    assert not np.all(obs['wall'] == 34), "Phase 1: Wall should be visible"

    print("✓ Phase 1 (omniscient) test passed")


def test_phase_2_progressive():
    """阶段2：渐进式掩码"""
    # progress=0.0 应接近无掩码
    env1 = WuhanMahjongEnv(training_phase=2, phase2_progress=0.0)
    obs1, _ = env1.reset(seed=42)
    obs1 = env1.observe(env1.agent_selection)
    # 大部分情况下对手手牌应该可见
    opponent_hand = obs1['global_hand'][0:34]  # 假设当前是player_0
    # 由于随机性，多次测试应该有可见的情况
    visible_count = 0
    for _ in range(10):
        obs, _ = env1.reset(seed=42)
        obs = env1.observe(env1.agent_selection)
        if not np.all(obs['global_hand'][34:68] == 5):
            visible_count += 1
    assert visible_count > 0, "Phase 2 progress=0.0: Should sometimes show opponent hands"
    print(f"Phase 2 progress=0.0: Opponent hands visible in {visible_count}/10 resets")

    # progress=1.0 应接近完全掩码
    env2 = WuhanMahjongEnv(training_phase=2, phase2_progress=1.0)
    masked_count = 0
    for _ in range(10):
        obs, _ = env2.reset(seed=42)
        obs = env2.observe(env2.agent_selection)
        if np.all(obs['global_hand'][34:68] == 5) and np.all(obs['wall'] == 34):
            masked_count += 1
    assert masked_count > 0, "Phase 2 progress=1.0: Should sometimes mask opponent hands"
    print(f"Phase 2 progress=1.0: Opponent hands masked in {masked_count}/10 resets")

    print("✓ Phase 2 (progressive) test passed")


def test_phase_3_real_state():
    """阶段3：真实状态，对手信息应被掩码"""
    env = WuhanMahjongEnv(training_phase=3)
    obs, _ = env.reset(seed=42)

    # 检查所有玩家的观测
    for agent_id in range(4):
        obs = env.observe(f"player_{agent_id}")

        # 验证对手手牌被掩码（全为5）
        global_hand = obs['global_hand']
        for i in range(4):
            if i != agent_id:
                player_hand = global_hand[i * 34:(i + 1) * 34]
                assert np.all(player_hand == 5), \
                    f"Phase 3: Agent {agent_id} should NOT see opponent {i}'s hand"

        # 验证牌墙被掩码（全为34）
        assert np.all(obs['wall'] == 34), "Phase 3: Wall should be masked"

    print("✓ Phase 3 (real state) test passed")


def test_concealed_kong_masking():
    """测试暗杠牌的掩码"""
    env = WuhanMahjongEnv(training_phase=3)
    obs, _ = env.reset(seed=42)

    action_types = obs['melds']['action_types']
    tiles = obs['melds']['tiles']
    KONG_CONCEALED = 5

    # 检查是否有暗杠
    has_concealed_kong = False
    for player_id in range(4):
        for meld_idx in range(4):
            idx = player_id * 4 + meld_idx
            if action_types[idx] == KONG_CONCEALED:
                has_concealed_kong = True
                # 检查暗杠的牌是否被掩码为34
                base_tile_idx = (player_id * 4 * 4 + meld_idx * 4) * 34
                for tile_pos in range(4):
                    tile_start = base_tile_idx + tile_pos * 34
                    tile_section = tiles[tile_start:tile_start + 34]
                    # 应该全为34或全为0（取决于是否是该玩家的观测）
                    if np.all(tile_section == 34) or np.all(tile_section == 0):
                        pass  # 正确：被掩码或该位置没有牌
                    else:
                        raise AssertionError(
                            f"Concealed kong tiles should be masked to 34 or 0, "
                            f"got {tile_section}"
                        )

    if has_concealed_kong:
        print("✓ Concealed kong masking test passed")
    else:
        print("⚠ No concealed kong found in test game, skipping masked tile check")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Visibility Mask Curriculum Learning")
    print("=" * 60)

    test_phase_1_omniscient()
    test_phase_2_progressive()
    test_phase_3_real_state()
    test_concealed_kong_masking()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
```

**Step 2: 运行测试**

```bash
"D:\DATA\Development\Anaconda\envs\PettingZooRLMahjong\python.exe" tests/integration/test_visibility_mask.py
```

Expected: 所有测试通过

**Step 3: 提交**

```bash
git add tests/integration/test_visibility_mask.py
git commit -m "test: add visibility mask curriculum learning tests"
```

---

## Task 5: 更新示例代码

**Files:**
- Modify: `example_mahjong_env.py:809-851` (main 代码块)

**Step 1: 更新示例代码中的环境创建**

找到第814行：
```python
env = WuhanMahjongEnv(render_mode='human', training_phase=3)
```

在其上方添加注释说明三阶段用法：

```python
# 创建环境
# training_phase: 1=全知视角, 2=渐进式掩码(需配合phase2_progress), 3=真实状态
env = WuhanMahjongEnv(render_mode='human', training_phase=3)
```

**Step 2: 提交**

```bash
git add example_mahjong_env.py
git commit -m "docs: add training_phase usage comment in example code"
```

---

## 验证清单

完成所有任务后，运行以下命令验证：

```bash
# 1. 运行可见度掩码测试
"D:\DATA\Development\Anaconda\envs\PettingZooRLMahjong\python.exe" tests/integration/test_visibility_mask.py

# 2. 简单环境测试
"D:\DATA\Development\Anaconda\envs\PettingZooRLMahjong\python.exe" example_mahjong_env.py

# 3. 确认git状态
git status
git log --oneline -5
```

---

## 参考文档

- **武汉麻将规则:** `src/mahjong_rl/rules/wuhan_mahjong_rule_engine/wuhan_mahjong_rules.md`
- **状态机文档:** `CLAUDE.md` 中的"状态机转换流程"章节
- **ActionType定义:** `src/mahjong_rl/core/constants.py:44-49` (KONG相关)
