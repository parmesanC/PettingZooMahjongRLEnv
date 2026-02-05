# 重构计划：GongState 显式设计模式改造

## TL;DR

> 重构 `gong_state.py`，消除 `hasattr` 隐式类型判断，改为使用显式的 **策略模式 + 统一 ActionRecord 接口**。提升代码可维护性和类型安全。
> 
> **影响范围**：4 个核心文件（gong_state.py, GameData.py, player_decision_state.py, wait_response_state.py）
> **预计耗时**：中等（2-3 小时）
> **风险等级**：中（涉及核心游戏逻辑，需充分测试）

---

## 核心问题

### 当前设计缺陷
```python
# gong_state.py:88-114
if hasattr(context, 'selected_responder') and context.selected_responder is not None:
    # 明杠逻辑（通过隐式属性判断）
    ...
else:
    # 其他杠类型
    if not hasattr(context, 'pending_kong_action'):  # 动态属性！
        raise ValueError(...)
```

**问题清单**：
1. ❌ 隐式类型判断（hasattr）
2. ❌ `pending_kong_action` 未在 GameContext 定义中声明（动态添加）
3. ❌ 6种杠类型混杂在一个状态
4. ❌ 两种完全不同的进入路径混在一起
5. ❌ 违背单一职责原则

### 目标设计
使用 **策略模式（Strategy Pattern）** + **统一接口（Unified Interface）**：
```python
# 重构后的理想代码
handlers = {
    ActionType.KONG_EXPOSED: self._handle_exposed_kong,
    ActionType.KONG_SUPPLEMENT: self._handle_supplement_kong,
    ActionType.KONG_CONCEALED: self._handle_concealed_kong,
    # ...
}
handler = handlers[context.pending_gong_action.action_type]
return handler(context, record)
```

---

## 改动清单

### 文件 1: `src/mahjong_rl/core/GameData.py`

**改动内容**：
1. 在 dataclass 中明确定义 `pending_gong_action: Optional[ActionRecord]`
2. 移除 `selected_responder` 的冗余使用（明杠时也使用 `pending_gong_action`）
3. 可选：添加类型注解确保类型安全

**具体位置**：
- 第 70 行附近添加字段定义
- 第 39 行附近移除 `pending_kong_action`（如果存在）

**风险评估**：
- ⚠️ 低风险：纯字段添加，不影响现有逻辑
- ⚠️ 注意：需检查其他代码是否依赖 `pending_kong_action` 字段名

### 文件 2: `src/mahjong_rl/state_machine/states/player_decision_state.py`

**改动内容**：
1. 所有杠类型处理器（_handle_supplement_kong, _handle_concealed_kong 等）统一设置 `context.pending_gong_action`
2. 不再设置 `context.pending_kong_action`（动态属性）

**具体位置**：
- 第 136-206 行：所有 _handle_*_kong 方法

**修改示例**：
```python
def _handle_concealed_kong(self, context, action, player_data):
    # 当前代码
    context.pending_kong_action = action  # ❌ 动态属性
    
    # 改为
    context.pending_gong_action = ActionRecord(
        action_type=action,
        tile=action.parameter,
        player_id=player_data.player_id
    )
    return GameStateType.GONG
```

**风险评估**：
- ⚠️ 中风险：改动状态转换逻辑
- ✅ 测试保障：所有杠类型测试需通过

### 文件 3: `src/mahjong_rl/state_machine/states/wait_response_state.py`

**改动内容**：
1. `_select_best_response` 方法中，当选择 KONG_EXPOSED 时，设置 `pending_gong_action`
2. 不再依赖 `selected_responder` 判断杠类型

**具体位置**：
- 第 220-223 行：明杠状态转换

**修改示例**：
```python
current
if best_response.action_type == ActionType.KONG_EXPOSED:
    context.selected_responder = best_response.player_id
    return GameStateType.GONG

改为
if best_response.action_type == ActionType.KONG_EXPOSED:
    context.pending_gong_action = ActionRecord(
        action_type=MahjongAction(ActionType.KONG_EXPOSED, best_response.parameter),
        tile=best_response.parameter,
        player_id=best_response.player_id
    )
    context.selected_responder = best_response.player_id  # 保留，ProcessMeldState 仍需要
    return GameStateType.GONG
```

**风险评估**：
- ⚠️ 中风险：改动状态转换
- ⚠️ 注意：`selected_responder` 仍需保留（ProcessMeldState 需要）

### 文件 4: `src/mahjong_rl/state_machine/states/gong_state.py`（核心）

**改动内容**：
1. **重构 step() 方法**：使用策略模式替代条件判断
2. **创建统一处理器**：6个杠类型各自的处理方法
3. **简化逻辑**：移除 hasattr 判断
4. **类型安全**：使用显式 ActionRecord

**具体位置**：
- 第 62-154 行：整个 step() 方法重写
- 第 156-175 行：新增处理辅助方法

**重构后结构**：
```python
class GongState(GameState):
    def __init__(self, rule_engine, observation_builder):
        super().__init__(rule_engine, observation_builder)
        # 策略模式：杠类型处理器映射
        self.kong_handlers = {
            ActionType.KONG_EXPOSED: self._handle_kong_exposed,
            ActionType.KONG_SUPPLEMENT: self._handle_kong_supplement,
            ActionType.KONG_CONCEALED: self._handle_kong_concealed,
            ActionType.KONG_RED: self._handle_kong_special,
            ActionType.KONG_SKIN: self._handle_kong_special,
            ActionType.KONG_LAZY: self._handle_kong_special,
        }
    
    def step(self, context, action):
        # 验证
        if context.pending_gong_action is None:
            raise ValueError("No pending_gong_action")
        
        record = context.pending_gong_action
        kong_type = record.action_type.action_type
        
        # 策略分发
        handler = self.kong_handlers.get(kong_type)
        if handler is None:
            raise ValueError(f"Unknown kong type: {kong_type}")
        
        # 补杠特殊处理（需要抢杠检查）
        if kong_type == ActionType.KONG_SUPPLEMENT:
            return self._handle_supplement_kong_with_rob_check(context, record)
        
        # 其他杠类型
        return handler(context, record)
```

**辅助方法**：
- `_handle_kong_exposed(context, record)` - 明杠
- `_handle_kong_concealed(context, record)` - 暗杠  
- `_handle_kong_supplement_with_rob_check(context, record)` - 补杠（抢杠检查）
- `_handle_kong_special(context, record, special_type)` - 特殊杠

**风险评估**：
- 🔴 **高风险**：核心游戏状态改动
- 🔴 需确保所有杠类型流程正确
- 🔴 需测试抢杠和、杠上开花等边界情况

---

## 测试策略

### 测试要求（必须遵守）

**MANDATORY: ZERO HUMAN INTERVENTION**
- ❌ 禁止："Run the game and manually test..."
- ❌ 禁止："User opens browser and checks..."
- ✅ **所有测试必须通过自动化脚本执行**

### 测试覆盖清单

#### 1. 现有测试检查
```bash
# 运行现有测试验证基准
python tests/integration/test_win_by_discard.py
python tests/integration/test_rob_kong.py
python tests/integration/test_auto_skip_state.py
python test_state_machine.py
```

**预期**：所有现有测试必须在重构前通过（建立基准）

#### 2. 单元测试（需编写/更新）

**GongState 单元测试**：
```python
def test_gong_state_with_exposed_kong():
    """测试明杠流程"""
    # 创建 mock context
    # 设置 pending_gong_action = ActionRecord(KONG_EXPOSED, ...)
    # 调用 step()
    # 验证： DRAWING_AFTER_GONG 状态
    # 验证：牌已从弃牌堆移除
    # 验证：副露已添加

def test_gong_state_with_supplement_kong():
    """测试补杠流程（进入抢杠检查）"""
    # 设置 pending_gong_action = ActionRecord(KONG_SUPPLEMENT, ...)
    # 调用 step()
    # 验证：WAIT_ROB_KONG 状态

def test_gong_state_with_concealed_kong():
    """测试暗杠流程"""
    # 类似明杠测试

def test_gong_state_invalid_no_pending_action():
    """测试无 pending_gong_action 时抛出异常"""
    # context.pending_gong_action = None
    # 调用 step() 应抛出 ValueError
```

**状态转换测试**：
```python
def test_player_decision_to_gong_transition():
    """测试 PLAYER_DECISION → GONG 的 ActionRecord 设置"""
    
def test_wait_response_to_gong_transition():
    """测试 WAITING_RESPONSE → GONG 的 ActionRecord 设置"""
```

#### 3. 集成测试（使用现有）

**必须运行的集成测试**：
- `tests/integration/test_rob_kong.py` - 抢杠和流程
- `tests/integration/test_win_by_discard.py` - 和牌流程
- `tests/integration/test_auto_skip_state.py` - 自动跳过

**预期结果**：所有测试通过后，输出 PASS

### 测试命令

```bash
# 1. 建立基准（重构前）
python tests/integration/test_rob_kong.py
python tests/integration/test_win_by_discard.py
python test_state_machine.py

# 2. 重构后验证
python -m pytest tests/ -v  # 如果有 pytest
# 或
python tests/integration/test_rob_kong.py
python tests/integration/test_win_by_discard.py
python tests/integration/test_auto_skip_state.py

# 3. 游戏流程测试（自动化）
python play_mahjong.py --mode human_vs_ai --renderer cli --test-mode  # 假设有测试模式
```

---

## 实施步骤

### Wave 1: 准备阶段（顺序执行）

**TODO-1: 分析现有测试覆盖**
- [ ] 1.1 运行所有现有测试，建立通过基准
- [ ] 1.2 识别需要新增的测试场景
- [ ] 1.3 确认 GameContext 字段定义

**TODO-2: GameContext 字段添加（低风险）**
- [ ] 2.1 在 GameData.py 添加 `pending_gong_action: Optional[ActionRecord]`
- [ ] 2.2 添加类型注解
- [ ] 2.3 验证 dataclass 正确性（实例化测试）

### Wave 2: 上游状态修改（中等风险，依赖 Wave 1）

**TODO-3: player_decision_state.py 重构**
- [ ] 3.1 修改 _handle_supplement_kong
- [ ] 3.2 修改 _handle_concealed_kong
- [ ] 3.3 修改 _handle_red_kong
- [ ] 3.4 修改 _handle_skin_kong
- [ ] 3.5 修改 _handle_lazy_kong
- [ ] 3.6 运行测试验证

**TODO-4: wait_response_state.py 重构**
- [ ] 4.1 修改 _select_best_response 中的 KONG_EXPOSED 处理
- [ ] 4.2 确保仍设置 selected_responder（ProcessMeldState 需要）
- [ ] 4.3 运行测试验证

### Wave 3: 核心重构（高风险，依赖 Wave 2）

**TODO-5: GongState 策略模式重构**
- [ ] 5.1 在 __init__ 中定义 kong_handlers 映射
- [ ] 5.2 重写 step() 方法
- [ ] 5.3 重构 _handle_kong_exposed 适配新接口
- [ ] 5.4 重构 _handle_kong_concealed 适配新接口
- [ ] 5.5 创建 _handle_kong_supplement_with_rob_check
- [ ] 5.6 重构 _handle_special_kong 适配新接口
- [ ] 5.7 清理旧的 hasattr 代码
- [ ] 5.8 运行单元测试

### Wave 4: 验证阶段（依赖 Wave 3）

**TODO-6: 测试验证**
- [ ] 6.1 运行所有现有集成测试
- [ ] 6.2 执行游戏流程测试（自动化）
- [ ] 6.3 边界情况测试（抢杠和、杠上开花）

**TODO-7: 代码审查**
- [ ] 7.1 自我代码审查（对照本计划）
- [ ] 7.2 更新状态机 README 文档
- [ ] 7.3 添加代码注释说明新设计模式

---

## 风险分析与缓解

| 风险 | 影响 | 可能性 | 缓解措施 |
|------|------|--------|----------|
| 重构引入 bug | 高 | 中 | 充分测试，分阶段实施，每次改动后验证 |
| 遗漏字段引用 | 中 | 中 | 全局搜索 `pending_kong_action`，确保全部替换 |
| ProcessMeldState 受影响 | 中 | 低 | 保留 selected_responder，仅改动 GONG 相关逻辑 |
| 抢杠和逻辑出错 | 高 | 低 | 专项测试 test_rob_kong.py 必须通过 |
| 类型不匹配 | 中 | 低 | 添加类型注解，运行时验证 |

### 回滚策略

**触发条件**：
- 任何集成测试失败且无法快速修复（>30分钟）
- 游戏流程测试发现严重 bug
- 发现遗漏的依赖关系

**回滚步骤**：
1. 使用 git 回滚到重构前提交
2. 恢复所有修改的文件
3. 重新运行测试确认基准恢复

---

## 验收标准

### 功能验收
- [ ] ✅ 所有6种杠类型流程正确（明杠、暗杠、补杠、红中杠、皮子杠、赖子杠）
- [ ] ✅ 抢杠和流程正确（补杠 → 等待抢杠 → 杠后补牌）
- [ ] ✅ 杠上开花检测正确
- [ ] ✅ 现有所有集成测试通过

### 代码质量验收
- [ ] ✅ 无 `hasattr` 隐式判断
- [ ] ✅ 使用策略模式分发
- [ ] ✅ 统一使用 `pending_gong_action: Optional[ActionRecord]`
- [ ] ✅ 类型注解完整
- [ ] ✅ 代码注释清晰

### 性能验收
- [ ] ✅ 无性能退化（策略模式 vs 条件判断性能相当）

---

## 后续建议

### 可选优化（超出本次范围）
1. **将策略模式提取到基类**：其他状态（如 ProcessMeldState）也可使用策略模式
2. **使用枚举定义处理器映射**：更类型安全
3. **引入依赖注入**：将 handler 映射作为构造函数参数

### 代码审查关注点
1. 确保 `selected_responder` 仍在 ProcessMeldState 需要的地方设置
2. 确保抢杠和特殊逻辑完整保留
3. 确保所有杠类型的牌操作正确（手牌移除、副露添加）

---

## 附录：字段引用检查清单

重构前需全局搜索确认：
```bash
# 搜索所有 pending_kong_action 引用
grep -r "pending_kong_action" --include="*.py" src/

# 搜索所有 selected_responder 引用（确认 ProcessMeldState 依赖）
grep -r "selected_responder" --include="*.py" src/

# 搜索 gong_state 相关的 hasattr
grep -r "hasattr.*context" --include="*.py" src/
```

**预期结果**：
- `pending_kong_action` 只出现在 player_decision_state.py（待删除）
- `selected_responder` 出现在 wait_response_state.py 和 process_meld_state.py（保留）
- `hasattr` 只在 gong_state.py 使用（本次重构目标）

---

## 执行命令

开始执行时运行：
```bash
/start-work
```

这将激活计划跟踪，确保每个 TODO 按顺序完成。

---

**计划制定完成** ✅
**准备就绪，等待执行指令**
