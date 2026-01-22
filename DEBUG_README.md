# KONG_LAZY 错误调试说明

## 问题背景
在 MELD_DECISION 状态下，当玩家选择 KONG_LAZY（赖子杠）动作时，出现错误：
```
list.remove(x): x not in list
```

## 已添加的调试代码

### 1. ActionValidator 调试输出
**文件**: `src/mahjong_rl/rules/wuhan_mahjong_rule_engine/action_validator.py`

在 `detect_available_actions_after_draw` 方法中添加了以下调试输出：
- 玩家ID、draw_tile、手牌
- 特殊牌信息（lazy_tile、skin_tile、red_dragon）
- self.special_tiles 构成
- self.pizi 和 context.skin_tile 的 id（检查是否是同一引用）
- temp_hand_counter 内容
- 每个特殊牌的处理过程
- 检测到的可用动作列表

### 2. GongState 调试输出
**文件**: `src/mahjong_rl/state_machine/states/gong_state.py`

在 `_handle_special_kong` 方法中添加了以下调试输出：
- kong_type、要移除的牌
- 玩家手牌（移除前）
- context 中的特殊牌信息
- 检查牌是否在手牌中
- 玩家手牌（移除后）
- special_gangs 计数

## 测试脚本

### 1. test_kong_lazy_debug.py
**用途**: 完整的游戏运行测试，重现 KONG_LAZY 错误

**运行方式**:
```bash
cd D:\DATA\Python_Project\Code\PettingZooRLENVMahjong
python test_kong_lazy_debug.py
```

**功能**:
- 创建环境并运行最多50步
- 打印每一步的详细状态
- 捕获错误并打印完整的堆栈信息
- 打印错误发生时的游戏状态

### 2. test_action_validator_init.py
**用途**: 单独测试 ActionValidator 的初始化时机

**运行方式**:
```bash
python test_action_validator_init.py
```

**功能**:
- 场景1：在 skin_tile 初始化之前创建 ActionValidator
- 场景2：在 skin_tile 初始化之后创建 ActionValidator
- 场景3：测试赖子杠检测逻辑（玩家有赖子牌）
- 场景4：测试赖子杠检测逻辑（玩家没有赖子牌）

### 3. test_obs_builder_validator.py
**用途**: 测试 ObservationBuilder 中的 ActionValidator

**运行方式**:
```bash
python test_obs_builder_validator.py
```

**功能**:
- 测试 ObservationBuilder 构建时的 ActionValidator 初始化
- 测试 MELD_DECISION 状态下的 action_mask 构建
- 验证赖子杠检测是否正确

## 运行顺序建议

### 步骤1：运行单独的 ActionValidator 测试
```bash
python test_action_validator_init.py
```

这个脚本会验证：
- ActionValidator 的初始化时机问题
- self.pizi 是否正确引用 context.skin_tile
- 赖子杠的检测逻辑是否正确

### 步骤2：运行 ObservationBuilder 测试
```bash
python test_obs_builder_validator.py
```

这个脚本会验证：
- ObservationBuilder 中的 ActionValidator 是否正确初始化
- MELD_DECISION 状态下的 action_mask 是否正确

### 步骤3：运行完整游戏测试
```bash
python test_kong_lazy_debug.py
```

这个脚本会：
- 运行完整的游戏
- 捕获错误并打印调试信息
- 帮助定位问题

## 需要粘贴回来的信息

### 来自 test_action_validator_init.py 的输出
特别关注：
- 场景1中，更新 skin_tile 后，validator1.special_tiles 的值
- 场景4中，玩家没有赖子牌时，是否检测到赖子杠

### 来自 test_obs_builder_validator.py 的输出
特别关注：
- MELD_DECISION 状态下，是否检测到赖子杠
- 调用 ActionValidator 时的调试输出

### 来自 test_kong_lazy_debug.py 的输出
特别关注：
- 错误发生时的完整堆栈
- ActionValidator 的调试输出（特别是 special_tiles 和 temp_hand_counter）
- GongState._handle_special_kong 的调试输出

## 预期的问题根源

根据分析，问题可能是以下之一：

### 问题A：ActionValidator 初始化时机问题
**症状**: self.special_tiles 包含初始值 [-1, -1]，导致检测错误

**检查方法**: 查看 test_action_validator_init.py 的输出
```
id(validator1.pizi) == id(context1.skin_tile)  # 应该是 True
```

### 问题B：玩家手牌中没有赖子牌，但检测到 KONG_LAZY
**症状**: temp_hand_counter 中没有赖子牌，但仍添加 KONG_LAZY 动作

**检查方法**: 查看 ActionValidator 的调试输出
```
temp_hand_counter: {牌ID: 数量, ...}  # 检查赖子牌的计数
```

### 问题C：检测和执行之间手牌被修改
**症状**: ActionValidator 检测时牌在手牌中，但 GongState 执行时牌不在手牌中

**检查方法**: 对比两个调试输出的手牌状态

## 移除调试代码

调试完成后，需要移除调试代码。可以使用 git diff 查看修改，然后手动移除：
```bash
git diff src/mahjong_rl/rules/wuhan_mahjong_rule_engine/action_validator.py
git diff src/mahjong_rl/state_machine/states/gong_state.py
```
