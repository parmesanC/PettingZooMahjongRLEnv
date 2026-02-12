# 性能优化设计文档

**创建日期**: 2026-02-12
**设计者**: Claude + 汪呜呜
**状态**: 设计完成，待实施

## 概述

通过激进的实例缓存策略，最大化训练速度。主要解决 ActionValidator 和 WuhanMahjongWinChecker 的重复实例化问题，保留 C++ 扩展的缓存优势。

## 问题背景

### 当前性能瓶颈

| 组件 | 每局实例化次数 | 主要创建位置 |
|------|----------------|-------------|
| ActionValidator | ~240次 | ObservationBuilder (3处/次) |
| WuhanMahjongWinChecker | ~40次 | 各状态类 + ActionValidator |
| C++ 扩展缓存 | 不共享 | 每个新实例清空缓存 |

### 典型调用链

```
env.step()
  -> state_machine.step()
    -> State.enter()
      -> observation_builder.build()
        -> ActionValidator(context).detect_xxx()  # 每次创建
          -> WuhanMahjongWinChecker(context)  # 每次创建
            -> mjc.MahjongWinChecker()  # C++ 实例
```

## 优化方案：激进缓存策略

### 核心架构

**1. CachedActionValidator**
- 每个 episode 创建一次（env.reset() 时）
- 存活整个 episode
- 消除 240 次重复实例化 → 1 次/局

**2. CachedWinChecker**
- 每个 episode 创建一次
- 保留 C++ 扩展的 melds_cache_
- 缓存随 episode 增长，win 检测加速

**3. ActionMaskCache**
- 缓存 action_mask 直到游戏状态改变
- 目标命中率 > 60%

## 实现设计

### Section 1: 核心单例架构

**环境类修改** (`example_mahjong_env.py`)
```python
class WuhanMahjongEnv:
    def __init__(self, ...):
        # ... 现有代码 ...
        self._cached_validator = None
        self._cached_win_checker = None
        self._mask_cache = None

    def reset(self, seed=None, options=None):
        # ... 现有代码 ...

        # 创建单例（每个 episode 一次）
        from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.action_validator import ActionValidator
        from src.mahjong_rl.rules.wuhan_mahjong_rule_engine.win_detector import WuhanMahjongWinChecker
        from src.mahjong_rl.optimization.mask_cache import ActionMaskCache

        self._cached_validator = ActionValidator(self.context)
        self._cached_win_checker = WuhanMahjongWinChecker(self.context)
        self._mask_cache = ActionMaskCache()

        # 注入到状态机
        self.state_machine.set_cached_components(
            validator=self._cached_validator,
            win_checker=self._cached_win_checker
        )
```

### Section 2: 集成点修改

**ObservationBuilder** (`wuhan_7p4l_observation_builder.py`)
```python
class Wuhan7P4LObservationBuilder(IObservationBuilder):
    def __init__(self, context: Optional[GameContext] = None):
        self.context = context
        self._cached_validator = None  # 新增

    def set_cached_validator(self, validator):
        """设置缓存的验证器"""
        self._cached_validator = validator

    def _build_decision_mask(self, player, context, mask):
        # 使用缓存的验证器
        validator = self._cached_validator
        if validator is None:
            validator = ActionValidator(context)  # 降级
        actions = validator.detect_available_actions_after_draw(player, None)
        # ... 其余逻辑不变

    # 类似修改 _build_meld_decision_mask 和 _build_response_mask
```

**状态机基类** (`state_machine/base.py` 或 `mahjong_state_machine.py`)
```python
class MahjongStateMachine:
    def __init__(self, rule_engine, observation_builder, logger=None, enable_logging=False):
        # ... 现有代码 ...
        self._cached_validator = None
        self._cached_win_checker = None

    def set_cached_components(self, validator=None, win_checker=None):
        """接收并传递缓存的组件到各状态"""
        self._cached_validator = validator
        self._cached_win_checker = win_checker

        # 传递给当前状态
        current_state = self._current_state
        if hasattr(current_state, 'set_cached_components'):
            current_state.set_cached_components(validator, win_checker)
```

**状态类修改** - 以 PlayerDecisionState 为例
```python
class PlayerDecisionState(GameState):
    def __init__(self, rule_engine, observation_builder):
        super().__init__(rule_engine, observation_builder)
        # ... 现有代码 ...
        self._cached_win_checker = None

    def set_cached_components(self, validator, win_checker):
        """接收缓存的组件"""
        if win_checker is not None:
            self._cached_win_checker = win_checker

    # 在需要检查和牌时使用缓存的 win_checker
    # 替代: WuhanMahjongWinChecker(context)
    # 改为: self._cached_win_checker 或降级创建
```

### Section 3: Action Mask 缓存机制

**新增文件** `src/mahjong_rl/optimization/mask_cache.py`
```python
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

@dataclass
class MaskCacheKey:
    """用于判断 mask 是否需要重建的缓存键"""
    player_id: int
    state: int  # GameStateType
    hand_hash: int
    discard_hash: int
    melds_hash: int

    @classmethod
    def from_context(cls, player_id: int, context: GameContext):
        player = context.players[player_id]
        return cls(
            player_id=player_id,
            state=context.current_state.value,
            hand_hash=hash(tuple(player.hand_tiles)),
            discard_hash=hash(tuple(context.discard_pile)),
            melds_hash=hash(tuple(str(m.tiles) for m in player.melds))
        )

class ActionMaskCache:
    """Action Mask 缓存管理器"""
    def __init__(self):
        self._cache = {}
        self._hits = 0
        self._misses = 0

    def get_or_build(self, key: MaskCacheKey, builder_func: Callable) -> np.ndarray:
        if key in self._cache:
            self._hits += 1
            return self._cache[key].copy()  # 返回副本避免意外修改

        self._misses += 1
        mask = builder_func()
        self._cache[key] = mask.copy()
        return mask

    def clear(self):
        """episode 结束时清空"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self):
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {"hits": self._hits, "misses": self._misses, "hit_rate": hit_rate}
```

**集成到 ObservationBuilder**
```python
from src.mahjong_rl.optimization.mask_cache import MaskCacheKey, ActionMaskCache

class Wuhan7P4LObservationBuilder(IObservationBuilder):
    def __init__(self, context: Optional[GameContext] = None):
        self.context = context
        self._cached_validator = None
        self._mask_cache = ActionMaskCache()  # 新增

    def build_action_mask(self, player_id: int, context: GameContext) -> np.ndarray:
        # ... 现有代码 ...

        # 构建缓存键
        cache_key = MaskCacheKey.from_context(player_id, context)

        # 使用缓存或构建
        def build_mask():
            return self._build_mask_uncached(player_id, context)

        return self._mask_cache.get_or_build(cache_key, build_mask)

    def _build_mask_uncached(self, player_id: int, context: GameContext) -> np.ndarray:
        """原有构建逻辑（重命名）"""
        # ... 原有 build_action_mask 的逻辑 ...
```

### Section 4: 文件修改清单

**优先级 P0（核心性能瓶颈）：**

| 文件 | 修改内容 | 新增行数 |
|------|---------|---------|
| `example_mahjong_env.py` | 添加缓存属性；reset() 中创建单例 | +15 |
| `wuhan_7p4l_observation_builder.py` | 使用缓存 validator；添加 mask 缓存 | +40 |
| 新增 `src/mahjong_rl/optimization/mask_cache.py` | MaskCacheKey, ActionMaskCache | +80 |
| 新增 `src/mahjong_rl/optimization/__init__.py` | 模块初始化 | +5 |

**优先级 P1（状态机集成）：**

| 文件 | 修改内容 | 新增行数 |
|------|---------|---------|
| `state_machine/mahjong_state_machine.py` | 添加 set_cached_components() | +15 |
| `states/player_decision_state.py` | 使用缓存 win_checker | +10 |
| `states/drawing_state.py` | 同上 | +8 |
| `states/drawing_after_gong_state.py` | 同上 | +8 |
| `states/wait_rob_kong_state.py` | 同上 | +8 |
| `states/win_state.py` | 同上 | +8 |

**优先级 P2（监控与验证）：**

| 文件 | 修改内容 | 新增行数 |
|------|---------|---------|
| `profile_performance.py` | 添加缓存命中率统计 | +30 |
| 新增 `tests/integration/test_cache_optimization.py` | 集成测试 | +100 |

## 验证计划

### 性能指标

| 指标 | 目标 | 测量方法 |
|------|------|---------|
| ActionValidator 实例化减少率 | >96% | 计数器对比 |
| WinChecker 实例化减少率 | >97.5% | 计数器对比 |
| Episode 时间改善 | >30% | time.perf_counter() |
| Mask 缓存命中率 | >60% | ActionMaskCache.stats() |

### 正确性验证

1. **行为一致性测试** - 优化前后游戏结果相同
2. **缓存失效测试** - 状态变化时缓存正确失效
3. **回归测试** - 所有现有集成测试通过

### 实施顺序

1. **Batch 1**: P0 修改（环境 + ObservationBuilder + 基础缓存类）
2. **Batch 2**: P1 修改（状态机集成）
3. **Batch 3**: P2 修改（监控测试）

## 预期效果

- 每局减少 ~280 次对象实例化
- C++ 缓存跨所有 win 检测共享
- 训练速度提升 30%+
- 内存增加 <100MB/环境

## 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 缓存状态不同步 | 使用不可变 key (hash) |
| 内存泄漏 | episode 结束强制 clear() |
| 行为不一致 | 完整的回归测试套件 |
| 线程安全 | 保持单进程设计 (AECEnv) |
