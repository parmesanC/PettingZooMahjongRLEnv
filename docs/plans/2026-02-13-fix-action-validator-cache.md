# ActionMask 和 ActionValidator 缓存 Bug 修复设计

**创建日期：** 2026-02-13
**目的：** 修复跨 episode 的错误缓存复用问题

---

## 问题描述

### Bug 1: ActionMaskCache 缓存键冲突

**现象：**
- Episode 44, Step 74, Player_1 的 action_mask 全为 0
- ActionValidator 检测到了 3 个可用动作
- 手动构建的 mask 有 3 个非零位
- 实际 action_mask 全为 0

**根因：**
- `MaskCacheKey` 只包含：player_id, state, hand_hash, discard_hash, melds_hash
- **缺少特殊牌信息**：lazy_tile, skin_tile
- 不同局的赖子皮子不同，但手牌相同时，产生相同缓存键
- 错误的 action_mask 被缓存并在后续 episodes 中复用

**影响：**
- 导致玩家无法执行任何动作（action_mask 全为 0）
- 随机发生，难以复现
- 破坏游戏体验

### Bug 2: ActionValidator 初始化时机问题

**现象：**
```
validator.laizi: None  ← ❌ 应该是 19
validator.special_tiles: [-1, -1, 31, None]  ← ❌ 应该包含 [18, 26, 31, 19]
```

**根因：**
- `ActionValidator` 在 `reset()` 中创建（第 428 行）
- 这时 `context.lazy_tile` 还是 `None`（默认值）
- 特殊牌在 `InitialState.step()` 中才生成（第 456 行）
- **错误的 validator 被缓存并在所有后续 episodes 中复用**

**影响：**
- ActionValidator 使用错误的特殊牌信息
- 导致动作检测错误

---

## 解决方案

### 第一部分：改进 MaskCacheKey 设计

**文件：** `src/mahjong_rl/optimization/mask_cache.py`

**修改：**

```python
@dataclass
class MaskCacheKey:
    """用于判断 mask 是否需要重建的缓存键"""
    player_id: int
    state: int
    hand_hash: int
    discard_hash: int
    melds_hash: int

    # ===== 新增：特殊牌信息 =====
    lazy_tile: int      # 赖子牌 ID
    skin_tile: tuple     # 皮子牌元组 (固定2张)

    @classmethod
    def from_context(cls, player_id: int, context: GameContext) -> 'MaskCacheKey':
        player = context.players[player_id]

        # 原有字段
        hand_hash = hash(tuple(player.hand_tiles))
        discard_hash = hash(tuple(context.discard_pile))
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
```

**效果：**
- 不同局的赖子皮子不同会产生不同的缓存键
- 避免跨 episode 的错误缓存复用

### 第二部分：让 ActionValidator 支持动态更新

**文件：** `src/mahjong_rl/rules/wuhan_mahjong_rule_engine/action_validator.py`

**新增方法：**

```python
class ActionValidator:
    def __init__(self, game_context):
        self.context = game_context
        self.laizi = game_context.lazy_tile
        self.pizi = game_context.skin_tile
        self.red = game_context.red_dragon
        self.special_tiles = self.pizi + [self.red, self.laizi]

    def update_context(self) -> None:
        """
        更新从 context 缓存的特殊牌信息

        应该在特殊牌生成后（InitialState 完成后）调用，
        确保 laizi, pizi, red 等属性是最新的。
        """
        self.laizi = self.context.lazy_tile
        self.pizi = self.context.skin_tile
        self.red = self.context.red_dragon
        self.special_tiles = self.pizi + [self.red, self.laizi]
```

**效果：**
- ActionValidator 可以动态更新特殊牌信息
- 保持缓存优势，不需要每次重新创建

### 第三部分：在 InitialState 完成后调用 update_context()

**文件：** `src/mahjong_rl/state_machine/states/initial_state.py`

**修改：** 在完成初始化后调用 `validator.update_context()`

**时机：**
- `InitialState.step()` 完成后
- 特殊牌已生成（lazy_tile, skin_tile）
- 调用 `validator.update_context()`

### 第四部分：清理代码

**文件：** `src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py`

**修改：**
- 移除之前添加的 `set_mask_cache_clear()` 方法
- 移除在 `build_action_mask` 中的清空缓存调用
- 恢复正常的缓存机制

---

## 实施计划

### 优先级顺序

1. **高优先级：修复 MaskCacheKey**
   - 文件：`src/mahjong_rl/optimization/mask_cache.py`
   - 添加 `lazy_tile`, `skin_tile` 字段
   - 估计时间：5 分钟

2. **中优先级：添加 ActionValidator.update_context()**
   - 文件：`src/mahjong_rl/rules/wuhan_mahjong_rule_engine/action_validator.py`
   - 添加动态更新方法
   - 估计时间：10 分钟

3. **中优先级：调用 update_context()**
   - 文件：`src/mahjong_rl/state_machine/states/initial_state.py`
   - 在完成初始化后调用
   - 估计时间：10 分钟

4. **低优先级：清理代码**
   - 文件：`src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py`
   - 移除之前的错误修复
   - 估计时间：5 分钟

**总估计时间：** 30 分钟

### 测试与验证

1. **单元测试：**
   - 测试 `MaskCacheKey` 哈希包含特殊牌
   - 测试相同手牌但不同特殊牌产生不同缓存键
   - 测试 `ActionValidator.update_context()` 方法

2. **集成测试：**
   - 运行 `diagnose_check_all.py` 50 个 episodes
   - 验证不再出现 action_mask 全为 0 的 bug
   - 验证缓存命中率合理

3. **性能测试：**
   - 运行 `profile_performance.py`
   - 确认缓存仍然有效（命中率 > 80%）
   - 确保性能没有回退

**验证标准：**
- ✅ 50 个 episodes 不出现 action_mask 全为 0
- ✅ `validator.laizi` 正确反映当前局的赖子
- ✅ 缓存命中率 > 80%（保留性能优势）

### 风险评估

- **低风险：** 只添加新字段，不改变现有逻辑
- **易回退：** 如有问题可快速撤销
- **向后兼容：** 不破坏现有 API

---

## 数据流

```
reset()
  → 创建空的 GameContext
  → InitialState.step() 执行
      → 洗牌、发牌
      → 生成特殊牌（lazy_tile, skin_tile）
      → 调用 validator.update_context()  ← 新增
  → 后续的 action_mask 构建使用正确更新的 validator
  → MaskCacheKey 包含 lazy_tile, skin_tile  ← 新增
  → 不同局产生不同的缓存键
  → 避免错误缓存复用
```
