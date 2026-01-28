# SimpleGameRunner 重构设计文档

> **目标:** 重构 `simple_game_runner.py` 继承 `ManualController` 基类，添加动作验证，采用异步事件驱动模式

**日期:** 2026-01-28

---

## 一、整体架构

### 1.1 核心改动

`SimpleGameRunner` 继承 `ManualController`，采用**异步事件驱动**模式替代原有的同步阻塞循环。

### 1.2 架构要点

1. **继承关系**：`SimpleGameRunner(ManualController)` 复用基类的 AI 处理、回合管理等逻辑
2. **事件驱动**：WebSocket 收到动作后立即触发验证和执行，不再阻塞等待
3. **动作验证层**：在 `on_action_received()` 回调中验证人类和 AI 动作的合法性
4. **状态同步**：动作执行后主动推送状态到前端，而非轮询

### 1.3 与基类的差异处理

- 基类 `run()` 方法是同步阻塞循环，不适合 WebSocket
- **方案**：重写 `run()` 方法，使用 FastAPI 的事件循环驱动游戏
- **复用**：基类的 `_get_ai_action()` 和 `_is_human_player()` 方法

### 1.4 组件协作流程

```
FastAPI Server → WebSocket 收到动作 → on_action_received() 回调
    ↓
验证动作合法性 → 非法则拒绝并返回错误提示
    ↓
env.step(action) → 状态机执行 → render_env() 推送新状态
    ↓
如果是 AI 玩家 → 调用基类 _get_ai_action() → 继续循环
```

---

## 二、动作验证层

### 2.1 验证逻辑复用

从 CLI 控制器的 `_validate_action()` 方法提取验证逻辑，创建可复用的验证器。

### 2.2 验证时机

在 `on_action_received()` 回调中立即验证，**在任何动作执行之前**。

### 2.3 验证内容

1. **玩家身份验证**：检查 `player_id` 是否与 `context.current_player_idx` 匹配
2. **动作类型验证**：检查 `action_type` 是否在当前状态下可用（基于 action_mask）
3. **参数验证**：检查 `parameter` 是否在有效范围内（如牌ID 0-33）
4. **资源验证**：对于杠牌动作，检查玩家手牌中是否有对应的牌

### 2.4 错误反馈机制

```python
def on_action_received(self, action, player_id=None):
    # 1. 验证玩家身份
    if player_id != current_player_idx:
        return self._send_error("不是你的回合")

    # 2. 验证动作合法性
    action_type, parameter = action
    if not self._validate_action(action_type, parameter, action_mask):
        action_name = self._get_action_name(action_type)
        return self._send_error(f"{action_name} 当前不可用")

    # 3. 验证通过，执行动作
    self._execute_action(action)
```

### 2.5 前端集成

扩展 WebSocket 消息协议，支持错误消息类型：

```json
{
  "type": "error",
  "message": "皮子杠 当前不可用"
}
```

---

## 三、异步事件驱动与 AI 玩家处理

### 3.1 核心问题

基类的 `run()` 是同步阻塞循环，但 WebSocket 需要异步事件驱动。

### 3.2 解决方案

重写 `run()` 方法，使用**步进式执行**而非循环。

### 3.3 实现方式

```python
def run(self):
    """启动 FastAPI 服务器（阻塞）"""
    self.server.start()

def on_action_received(self, action, player_id=None):
    """WebSocket 回调 - 驱动游戏前进一步"""
    # 1. 验证动作
    if not self._validate_and_set_action(action, player_id):
        return  # 验证失败，已发送错误

    # 2. 执行当前玩家动作
    obs, reward, terminated, truncated, info = self.env.step(self.pending_action)

    # 3. 自动处理 AI 玩家和自动状态
    self._process_auto_players(obs)

    # 4. 渲染新状态
    self.render_env()

def _process_auto_players(self, initial_obs):
    """处理 AI 玩家和自动状态推进"""
    while not self.env.unwrapped.context.is_win:
        current_agent = self.env.agent_selection

        # 如果是人类玩家，停止自动推进
        if self._is_human_player(current_agent):
            break

        # 获取 AI 动作（带验证）
        obs, reward, terminated, truncated, info = self.env.last()
        action = self._get_ai_action_with_validation(obs, info)

        # 执行 AI 动作
        obs, reward, terminated, truncated, info = self.env.step(action)

        # AI 延迟（观察用）
        if self.ai_delay > 0:
            time.sleep(self.ai_delay)

        # 发送状态更新
        self.render_env()
```

### 3.4 关键点

- 人类动作触发后，自动推进所有 AI 玩家
- 每个玩家动作后都 `render_env()` 推送状态
- AI 动作也经过验证（调用 `_get_ai_action_with_validation()`）

---

## 四、增强重启功能与配置保留

### 4.1 重启触发方式

1. 前端发送特殊动作 `(-1, 0)` 表示重启请求
2. 游戏结束后自动提示是否重启

### 4.2 配置保留策略

```python
def _restart_game(self, confirmed=True):
    """重启游戏，保留配置"""
    if not confirmed:
        return self._send_restart_confirmation_request()

    # 保留的配置
    saved_config = {
        'port': self.port,
        'max_episodes': self.max_episodes,
        'strategies': self.strategies,
        'ai_delay': self.ai_delay
    }

    # 重置环境
    self.env.reset()

    # 发送新状态到所有前端
    self.render_env()

    # 发送重启成功消息
    self._send_message("游戏已重启，配置已保留")
```

### 4.3 重启确认流程

```json
// 前端收到确认请求
{
  "type": "restart_confirmation",
  "message": "确定要重启游戏吗？"
}

// 前端发送确认
{
  "type": "restart_confirmed"
}
```

### 4.4 边界情况处理

- 游戏进行中重启：保留当前回合数
- 游戏结束后重启：重置回合数
- 多个前端连接：通知所有客户端重启

---

## 五、错误处理与测试策略

### 5.1 错误处理边界情况

1. **WebSocket 断连处理**
   - 玩家断开不影响游戏继续（AI 接管）
   - 重新连接后同步当前状态

2. **超时处理**
   - 人类玩家超时（如 5 分钟无响应）自动 PASS
   - 防止游戏卡死

3. **并发冲突**
   - 多个玩家同时发送动作 → 只接受当前回合玩家的动作
   - 动作执行期间锁定，防止重复提交

4. **状态不一致**
   - 定期同步前端状态
   - 检测到不一致时强制刷新

### 5.2 测试策略

1. **单元测试**
   - 测试动作验证逻辑（各种非法动作）
   - 测试 AI 玩家动作验证

2. **集成测试**
   - 完整游戏循环（4 个人类玩家）
   - 混合玩家（人类 + AI）
   - 重启功能测试

3. **边界测试**
   - WebSocket 断连重连
   - 超时处理
   - 并发动作提交

### 5.3 实现注意事项

- 保持与现有 `fastapi_server.py` 的兼容性
- 动作验证逻辑可以提取到 `utils/action_validator.py` 供复用
- 前端需要扩展错误消息处理
