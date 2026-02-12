# Fast Mode 设计文档

**日期**: 2026-02-13
**作者**: Claude
**目标**: 通过 `fast_mode` 参数禁用状态快照功能，加速训练速度

---

## 背景

当前状态机在每次状态转换时都会执行 `deepcopy(context)` 来保存快照（`machine.py:381`），用于支持状态回滚功能。在训练场景下，这种深拷贝会带来显著的性能开销：

- 每次状态转换都保存完整 GameContext 副本
- 一局游戏约 100-200 次状态转换
- 训练时可能有数百万次状态转换
- 状态回滚功能在训练时完全不需要

## 设计目标

**主要目标**: 加速训练速度

**次要目标**:
- 降低内存占用
- 保持向后兼容性
- 为后续增量快照优化预留空间

## 实现方案

### 1. 状态机层面修改

**MahjongStateMachine 类**:

- **构造函数添加 `fast_mode` 参数**
  ```python
  def __init__(
      self,
      rule_engine: IRuleEngine,
      observation_builder: IObservationBuilder,
      logger: Optional[ILogger] = None,
      enable_logging: bool = True,
      fast_mode: bool = False  # 新增
  ):
      self.fast_mode = fast_mode
  ```

- **修改 `_save_snapshot` 方法**
  ```python
  def _save_snapshot(self, context: GameContext):
      if self.fast_mode:
          return  # 快速模式下直接返回
      # 原有逻辑保持不变
      snapshot = {
          'state_type': self.current_state_type,
          'context': deepcopy(context),
          'timestamp': time.time()
      }
      self.state_history.append(snapshot)
      ...
  ```

- **修改 `rollback` 方法**
  ```python
  def rollback(self, steps: int = 1) -> GameContext:
      if self.fast_mode:
          raise RuntimeError("Cannot rollback in fast_mode - snapshots are disabled")
      # 原有逻辑保持不变
      ...
  ```

### 2. 环境层面修改

**WuhanMahjongEnv 类**:

- **构造函数暴露 `fast_mode` 参数**
  ```python
  def __init__(
      self,
      render_mode=None,
      debug_mode=False,
      fast_mode=False,  # 新增
      max_cycles=1000
  ):
      # 传递给状态机
      self.state_machine = MahjongStateMachine(
          rule_engine=self.rule_engine,
          observation_builder=self.observation_builder,
          logger=logger,
          enable_logging=debug_mode,
          fast_mode=fast_mode  # 传递
      )
  ```

### 3. 使用示例

```python
# 正常游戏/调试（保留完整功能）
env = WuhanMahjongEnv()

# 训练模式（禁用快照）
env = WuhanMahjongEnv(fast_mode=True)

# 最快训练配置（同时禁用日志）
env = WuhanMahjongEnv(fast_mode=True, debug_mode=False)
```

## 测试策略

### 单元测试

创建 `tests/unit/test_fast_mode.py`：

1. **验证快照禁用**
   ```python
   def test_fast_mode_disables_snapshots():
       env = WuhanMahjongEnv(fast_mode=True)
       env.reset()
       for _ in range(10):
           env.step(env.action_space(env.agent_selection))
       assert len(env.state_machine.get_history()) == 0
   ```

2. **验证 rollback 抛出异常**
   ```python
   def test_fast_mode_rollback_raises_error():
       env = WuhanMahjongEnv(fast_mode=True)
       env.reset()
       with pytest.raises(RuntimeError, match="fast_mode"):
           env.state_machine.rollback(1)
   ```

3. **验证普通模式保留功能**
   ```python
   def test_normal_mode_keeps_snapshots():
       env = WuhanMahjongEnv(fast_mode=False)
       env.reset()
       for _ in range(5):
           env.step(env.action_space(env.agent_selection))
       assert len(env.state_machine.get_history()) > 0
   ```

### 性能基准测试

```python
def test_fast_mode_performance():
    import time

    # 测试普通模式
    env_normal = WuhanMahjongEnv(fast_mode=False)
    start = time.time()
    run_n_episodes(env_normal, episodes=100)
    time_normal = time.time() - start

    # 测试快速模式
    env_fast = WuhanMahjongEnv(fast_mode=True)
    start = time.time()
    run_n_episodes(env_fast, episodes=100)
    time_fast = time.time() - start

    # 验证加速效果
    assert time_fast < time_normal
    speedup = time_normal / time_fast
    print(f"Speedup: {speedup:.2f}x")
```

## 边缘情况与注意事项

1. **向后兼容性**
   - `fast_mode` 默认值为 `False`
   - 现有代码无需修改

2. **日志独立性**
   - `fast_mode` 不影响 `external_logger` 的功能
   - 完全静默需同时设置 `enable_logging=False`

3. **内存管理**
   - 快速模式下 `state_history` 保持为空列表
   - 无需额外清理逻辑

4. **文档更新**
   - 更新 `example_mahjong_env.py` 的 docstring
   - 在 CLAUDE.md 中添加训练配置建议

## 预期效果

- **性能提升**: 减少 20-40% 的状态转换开销
- **内存节省**: 消除快照列表的内存占用
- **代码质量**: 最小化修改，保持清晰的职责分离

## 后续优化方向

1. **增量快照** (未来实现)
   - 只记录变化的部分（手牌、弃牌堆等）
   - 使用 `__getstate__` / `__setstate__` 自定义序列化
   - 在普通模式下提供更高效的快照机制

2. **性能分析**
   - 使用 profiler 验证实际加速效果
   - 识别其他性能瓶颈

3. **其他优化点**
   - 观测构建的缓存优化
   - 批量动作处理的优化
