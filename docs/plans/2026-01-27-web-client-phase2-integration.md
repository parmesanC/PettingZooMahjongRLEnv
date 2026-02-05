# 武汉麻将网页客户端阶段2实施计划 - 游戏逻辑集成

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 集成真实的游戏逻辑（WuhanMahjongEnv）到Phaser前端，实现前后端WebSocket通信和完整的游戏动作处理。

**架构:**
- 后端使用现有的 FastAPI 服务器扩展 JSON 格式状态传输
- 前端 Phaser 客户端通过 WebSocket 连接后端，接收游戏状态并发送动作
- 使用状态适配器模式转换 GameContext 到前端需要的格式

**技术栈:**
- 后端: FastAPI, WebSocket, WuhanMahjongEnv
- 前端: Phaser.js 3.x, WebSocket API
- 通信: JSON 格式消息

---

## 前置准备

### Task 0: 创建状态序列化器

**文件:**
- Create: `src/mahjong_rl/web/state_serializer.py`

**Step 1: 创建状态序列化器模块**

```python
"""
游戏状态序列化器
将 GameContext 转换为前端可用的 JSON 格式
"""
from typing import Dict, Any, List
from src.mahjong_rl.core.GameData import GameContext
from src.mahjong_rl.core.constants import GameStateType


class StateSerializer:
    """将游戏状态序列化为前端可用的格式"""

    @staticmethod
    def serialize(context: GameContext, observer_player_idx: int = 0) -> Dict[str, Any]:
        """
        将 GameContext 序列化为前端格式

        Args:
            context: 游戏上下文
            observer_player_idx: 观察者玩家索引（用于确定视角）

        Returns:
            前端可用的状态字典
        """
        return {
            'current_state': context.current_state.value if hasattr(context.current_state, 'value') else str(context.current_state),
            'current_player_idx': int(context.current_player_idx),
            'dealer_idx': int(context.dealer_idx) if context.dealer_idx is not None else 0,
            'lazy_tile': int(context.lazy_tile) if context.lazy_tile is not None else None,
            'skin_tiles': [int(t) for t in context.skin_tile] if context.skin_tile else [],
            'wall_count': len(context.wall),
            'players': [
                StateSerializer._serialize_player(p, observer_player_idx)
                for p in context.players
            ],
            'last_discarded_tile': int(context.last_discarded_tile) if context.last_discarded_tile is not None else None,
            'is_win': context.is_win,
            'is_flush': context.is_flush,
            'winner_ids': list(context.winner_ids) if context.winner_ids else []
        }

    @staticmethod
    def _serialize_player(player, observer_idx: int) -> Dict[str, Any]:
        """序列化玩家数据"""
        # 判断是否是观察者自己（决定是否显示手牌）
        is_self = player.player_id == observer_idx

        return {
            'player_id': int(player.player_id),
            'hand_tiles': [int(t) for t in player.hand_tiles] if is_self else [],
            'hand_count': len(player.hand_tiles),  # 对手只显示数量
            'melds': [
                {
                    'action_type': m.action_type.action_type.value,
                    'tiles': [int(t) for t in m.tiles],
                    'from_player': int(m.from_player)
                }
                for m in player.melds
            ],
            'discard_tiles': [int(t) for t in player.discard_tiles],
            'special_gangs': [int(x) for x in player.special_gangs],
            'is_dealer': bool(player.is_dealer),
            'is_win': bool(player.is_win)
        }
```

**Step 2: 验证文件创建成功**

Run: `cat src/mahjong_rl/web/state_serializer.py`
Expected: 显示完整的序列化器代码

**Step 3: 提交序列化器**

```bash
git add src/mahjong_rl/web/state_serializer.py
git commit -m "feat(web): add state serializer for frontend communication"
```

---

## 模块1: 后端WebSocket通信扩展

### Task 1: 扩展WebSocket消息格式

**文件:**
- Modify: `src/mahjong_rl/web/fastapi_server.py`

**Step 1: 添加JSON状态广播方法**

在 `MahjongFastAPIServer` 类中添加：

```python
def send_json_state(self, context: GameContext, observer_player_idx: int = 0):
    """
    发送JSON格式的游戏状态

    Args:
        context: 游戏上下文
        observer_player_idx: 观察者玩家索引
    """
    from .state_serializer import StateSerializer

    state_dict = StateSerializer.serialize(context, observer_player_idx)

    message = {
        'type': 'game_state',
        'state': state_dict
    }

    self.websocket_manager.broadcast_sync(message)
    print(f"📡 已发送游戏状态 (玩家{observer_player_idx}视角)")
```

**Step 2: 修改WebSocket端点支持JSON动作**

在 `websocket_endpoint` 函数中修改消息处理：

```python
@self.app.websocket("/ws/{player_id}")
async def websocket_endpoint(websocket: WebSocket, player_id: int):
    """WebSocket端点 - 支持玩家ID参数"""
    await websocket.accept()
    self.websocket_manager.active_connections.append(websocket)
    print(f"✓ 玩家{player_id}连接，总连接数: {len(self.websocket_manager.active_connections)}")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message['type'] == 'action':
                # 解析动作
                action_type = message['action_type']
                parameter = message.get('parameter', 0)

                # 调用控制器处理动作
                self.controller.on_action_received((action_type, parameter), player_id)

            elif message['type'] == 'get_state':
                # 请求当前状态
                if hasattr(self.controller, 'get_current_context'):
                    context = self.controller.get_current_context()
                    self.send_json_state(context, player_id)

    except WebSocketDisconnect:
        if websocket in self.websocket_manager.active_connections:
            self.websocket_manager.active_connections.remove(websocket)
        print(f"✓ 玩家{player_id}断开连接")

    except Exception as e:
        print(f"WebSocket错误 (玩家{player_id}): {e}")
        if websocket in self.websocket_manager.active_connections:
            self.websocket_manager.active_connections.remove(websocket)
```

**Step 3: 提交WebSocket扩展**

```bash
git add src/mahjong_rl/web/fastapi_server.py
git commit -m "feat(web): extend WebSocket for JSON state and action handling"
```

---

## 模块2: 前端WebSocket客户端

### Task 2: 创建WebSocket管理器

**文件:**
- Create: `src/mahjong_rl/web/phaser_client/js/utils/WebSocketManager.js`

**Step 1: 创建WebSocket管理器**

```javascript
/**
 * WebSocket通信管理器
 * 处理与后端的WebSocket连接和消息
 */

export class WebSocketManager {
    constructor(url, onMessageCallback) {
        this.url = url;
        this.onMessageCallback = onMessageCallback;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }

    /**
     * 连接WebSocket
     */
    connect(playerId = 0) {
        const wsUrl = `${this.url}/${playerId}`;
        console.log(`正在连接WebSocket: ${wsUrl}`);

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('✓ WebSocket连接成功');
            this.reconnectAttempts = 0;
        };

        this.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                console.log('收到消息:', message.type);

                if (this.onMessageCallback) {
                    this.onMessageCallback(message);
                }
            } catch (e) {
                console.error('解析消息失败:', e, event.data);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket错误:', error);
        };

        this.ws.onclose = () => {
            console.log('WebSocket连接关闭');
            this.attemptReconnect(playerId);
        };
    }

    /**
     * 尝试重连
     */
    attemptReconnect(playerId) {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * this.reconnectAttempts;

            console.log(`${delay}ms后尝试重连 (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

            setTimeout(() => {
                this.connect(playerId);
            }, delay);
        } else {
            console.error('达到最大重连次数，放弃重连');
        }
    }

    /**
     * 发送动作
     */
    sendAction(actionType, parameter = 0) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const message = {
                type: 'action',
                action_type: actionType,
                parameter: parameter
            };

            this.ws.send(JSON.stringify(message));
            console.log('发送动作:', message);
        } else {
            console.error('WebSocket未连接，无法发送动作');
        }
    }

    /**
     * 请求当前状态
     */
    requestState() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const message = { type: 'get_state' };
            this.ws.send(JSON.stringify(message));
        }
    }

    /**
     * 断开连接
     */
    disconnect() {
        if (this.ws) {
            this.reconnectAttempts = this.maxReconnectAttempts; // 防止重连
            this.ws.close();
        }
    }
}
```

**Step 2: 提交WebSocket管理器**

```bash
git add src/mahjong_rl/web/phaser_client/js/utils/WebSocketManager.js
git commit -m "feat(phaser-client): add WebSocket manager for backend communication"
```

---

### Task 3: 集成WebSocket到MahjongScene

**文件:**
- Modify: `src/mahjong_rl/web/phaser_client/js/scenes/MahjongScene.js`

**Step 1: 添加WebSocket管理器**

在文件顶部的导入部分添加：

```javascript
import { WebSocketManager } from '../utils/WebSocketManager.js';
```

**Step 2: 在构造函数中初始化WebSocket**

在 `constructor()` 中添加：

```javascript
// WebSocket管理器
this.wsManager = null;
this.playerId = 0;  // 默认为玩家0
```

**Step 3: 在create()中连接WebSocket**

在 `create()` 方法的末尾添加：

```javascript
// 初始化WebSocket连接
this.initWebSocket();
```

**Step 4: 添加WebSocket初始化方法**

在类中添加新方法：

```javascript
/**
 * 初始化WebSocket连接
 */
initWebSocket() {
    const wsUrl = `ws://${window.location.hostname}:8011/ws`;

    this.wsManager = new WebSocketManager(wsUrl, (message) => {
        this.handleWebSocketMessage(message);
    });

    this.wsManager.connect(this.playerId);
}

/**
 * 处理WebSocket消息
 */
handleWebSocketMessage(message) {
    switch (message.type) {
        case 'game_state':
            this.updateState(message.state);
            break;

        case 'initial_state':
            if (message.state) {
                this.updateState(message.state);
            }
            break;

        case 'action_prompt':
            // TODO: 显示动作提示UI
            console.log('动作提示:', message);
            break;

        case 'game_over':
            // TODO: 显示游戏结束UI
            console.log('游戏结束:', message);
            break;

        default:
            console.log('未知消息类型:', message.type);
    }
}
```

**Step 5: 修改updateState方法以适配后端格式**

修改 `updateState` 方法：

```javascript
/**
 * 更新游戏状态
 */
updateState(newState) {
    // 兼容后端返回的状态格式
    if (newState.current_state !== undefined) {
        // 后端状态：将数字转换为字符串
        const stateNames = {
            0: 'INITIAL',
            1: 'DRAWING',
            2: 'PLAYER_DECISION',
            3: 'DISCARDING',
            4: 'WAITING_RESPONSE',
            5: 'GONG',
            6: 'WIN',
            7: 'FLOW_DRAW'
        };

        newState.current_state = stateNames[newState.current_state] || 'INITIAL';
    }

    this.gameState = { ...this.gameState, ...newState };
    this.render();
}
```

**Step 6: 修改打牌方法以发送WebSocket动作**

修改 `updateAfterDiscard` 方法：

```javascript
/**
 * 打牌后更新游戏状态
 */
updateAfterDiscard(tileId, index, sortedTiles) {
    // 通过WebSocket发送打牌动作
    if (this.wsManager) {
        // ActionType.DISCARD = 0
        this.wsManager.sendAction(0, tileId);
    }

    // 本地临时更新（等待服务器确认后会覆盖）
    const player = this.gameState.players[0];
    const originalIndex = player.hand_tiles.indexOf(tileId);
    if (originalIndex > -1) {
        player.hand_tiles.splice(originalIndex, 1);
    }
    player.discard_tiles.push(tileId);

    // 重新渲染
    this.render();

    console.log(`Discarded tile ${tileId}. Waiting for server confirmation...`);
}
```

**Step 7: 提交WebSocket集成**

```bash
git add src/mahjong_rl/web/phaser_client/js/scenes/MahjongScene.js
git commit -m "feat(phaser-client): integrate WebSocket for real-time game state"
```

---

## 模块3: 游戏控制器集成

### Task 4: 创建简单游戏运行器

**文件:**
- Create: `src/mahjong_rl/web/simple_game_runner.py`

**Step 1: 创建游戏运行器**

```python
"""
简单的游戏运行器
用于启动FastAPI服务器并运行游戏循环
"""
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.mahjong_rl.web.fastapi_server import MahjongFastAPIServer
from example_mahjong_env import WuhanMahjongEnv


class SimpleGameRunner:
    """简单的游戏运行器"""

    def __init__(self, port=8011):
        self.port = port
        self.env = None
        self.server = None
        self.current_context = None

    def setup(self):
        """设置环境和服务器"""
        print("初始化武汉麻将环境...")

        # 创建环境
        self.env = WuhanMahjongEnv(
            render_mode=None,
            training_phase=3,  # 完全信息
            enable_logging=False
        )

        # 重置环境获取初始状态
        obs, info = self.env.reset()
        self.current_context = self.env.unwrapped.context

        print(f"✓ 环境初始化完成")
        print(f"  - 当前玩家: {self.current_context.current_player_idx}")
        print(f"  - 赖子: {self.current_context.lazy_tile}")
        print(f"  - 皮子: {self.current_context.skin_tile}")

    def on_action_received(self, action, player_id=None):
        """
        处理接收到的动作

        Args:
            action: (action_type, parameter) 元组
            player_id: 发送动作的玩家ID
        """
        current_player = self.env.agent_selection

        if player_id is not None and player_id != self.env.possible_agents.index(current_player):
            print(f"警告: 玩家{player_id}尝试在玩家{current_player}的回合行动")
            return

        action_type, parameter = action
        print(f"收到动作: type={action_type}, param={parameter}, player={current_player}")

        # 执行动作
        try:
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.current_context = self.env.unwrapped.context

            # 发送新状态到前端
            self.send_state_to_all()

            if terminated or truncated:
                print(f"\n游戏结束! 终止={terminated}, 截断={truncated}")
                if self.current_context.winner_ids:
                    print(f"获胜者: {self.current_context.winner_ids}")

        except Exception as e:
            print(f"执行动作失败: {e}")
            import traceback
            traceback.print_exc()

    def get_current_context(self):
        """获取当前游戏上下文"""
        return self.current_context

    def send_state_to_all(self):
        """发送状态给所有连接的客户端"""
        if self.server and self.current_context:
            # 给每个玩家发送对应视角的状态
            for player_idx in range(4):
                self.server.send_json_state(self.current_context, player_idx)

    def start(self):
        """启动服务器"""
        if not self.env:
            self.setup()

        # 创建控制器（将自身作为控制器传入）
        controller = self
        self.server = MahjongFastAPIServer(
            env=self.env,
            controller=controller,
            port=self.port
        )

        # 发送初始状态
        self.send_state_to_all()

        # 启动服务器
        self.server.start()


if __name__ == "__main__":
    runner = SimpleGameRunner(port=8011)
    runner.start()
```

**Step 2: 提交游戏运行器**

```bash
git add src/mahjong_rl/web/simple_game_runner.py
git commit -m "feat(web): add simple game runner for WebSocket testing"
```

---

## 测试点: 端到端测试

### Task 5: 端到端测试

**文件:**
- Test: 手动测试

**Step 1: 启动后端服务器**

```bash
cd D:\DATA\Python_Project\Code\PettingZooRLENVMahjong
python src/mahjong_rl/web/simple_game_runner.py
```

Expected: 服务器启动，显示：
```
初始化武汉麻将环境...
✓ 环境初始化完成
============================================================
🌐 FastAPI麻将游戏服务器
============================================================
📌 游戏地址: http://localhost:8011
📚 API文档: http://localhost:8011/docs
🔌 端点: /ws/{player_id}
============================================================
```

**Step 2: 修改前端连接地址**

修改 `src/mahjong_rl/web/phaser_client/js/utils/WebSocketManager.js` 中的连接地址：
- 将 `ws://${window.location.hostname}:8011/ws`
- 改为 `ws://localhost:8011/ws`

**Step 3: 启动前端**

```bash
cd src/mahjong_rl/web/phaser_client
python -m http.server 8080
```

**Step 4: 浏览器测试**

打开: `http://localhost:8080/index.html`

Expected:
- 控制台显示 "WebSocket连接成功"
- 看到真实的游戏状态（不再是测试数据）
- 点击手牌可以发送打牌动作到后端
- 后端接收动作并更新状态

**Step 5: 提交测试配置**

```bash
git add src/mahjong_rl/web/phaser_client/js/utils/WebSocketManager.js
git commit -m "fix(phaser-client): update WebSocket URL for local testing"
```

---

## 模块4: 完善动作处理

### Task 6: 实现完整的动作处理

**文件:**
- Modify: `src/mahjong_rl/web/phaser_client/js/scenes/MahjongScene.js`

**Step 1: 添加动作按钮UI**

在 `createGameBoard()` 方法后添加：

```javascript
/**
 * 创建动作按钮
 */
createActionButtons() {
    const scale = window.GLOBAL_SCALE_RATE;
    const centerX = this.cameras.main.width / 2;
    const buttonY = this.cameras.main.height - 250 * scale;

    // 按钮配置
    const buttons = [
        { text: '过', action: 10, x: centerX - 150 * scale },
        { text: '碰', action: 2, x: centerX - 75 * scale },
        { text: '杠', action: 3, x: centerX },
        { text: '胡', action: 9, x: centerX + 75 * scale }
    ];

    this.actionButtons = [];

    buttons.forEach(btn => {
        const button = this.add.text(btn.x, buttonY, btn.text, {
            fontFamily: 'Microsoft YaHei',
            fontSize: 24 * scale + 'px',
            color: '#ffffff',
            backgroundColor: '#4CAF50',
            padding: { x: 15 * scale, y: 10 * scale }
        }).setOrigin(0.5).setDepth(1500);

        button.setData('action', btn.action);
        button.setInteractive();
        button.setVisible(false);  // 默认隐藏

        button.on('pointerdown', () => {
            this.onActionButtonClick(btn.action);
        });

        this.actionButtons.push(button);
        this.layers.ui.add(button);
    });
}

/**
 * 动作按钮点击处理
 */
onActionButtonClick(actionType) {
    console.log('动作按钮点击:', actionType);

    if (this.wsManager) {
        this.wsManager.sendAction(actionType, 0);
    }

    // 隐藏所有按钮
    this.hideActionButtons();
}

/**
 * 显示动作按钮
 */
showActionButtons(availableActions) {
    // 根据可用动作显示对应按钮
    // TODO: 根据action_mask显示可用按钮
    this.actionButtons.forEach(btn => {
        btn.setVisible(true);
    });
}

/**
 * 隐藏动作按钮
 */
hideActionButtons() {
    this.actionButtons.forEach(btn => {
        btn.setVisible(false);
    });
}
```

**Step 2: 在create()中调用**

在 `create()` 方法中添加：

```javascript
// 创建动作按钮
this.createActionButtons();
```

**Step 3: 处理action_prompt消息**

在 `handleWebSocketMessage()` 方法中添加：

```javascript
case 'action_prompt':
    // 显示动作按钮
    this.showActionButtons(message.action_mask);
    break;
```

**Step 4: 提交动作按钮功能**

```bash
git add src/mahjong_rl/web/phaser_client/js/scenes/MahjongScene.js
git commit -m "feat(phaser-client): add action buttons for game interaction"
```

---

## 模块5: 游戏结束和重启

### Task 7: 实现游戏结束处理

**文件:**
- Modify: `src/mahjong_rl/web/phaser_client/js/scenes/MahjongScene.js`

**Step 1: 添加游戏结束UI**

```javascript
/**
 * 显示游戏结束界面
 */
showGameOverScreen(winnerIds) {
    const scale = window.GLOBAL_SCALE_RATE;
    const centerX = this.cameras.main.width / 2;
    const centerY = this.cameras.main.height / 2;

    // 半透明遮罩
    const overlay = this.add.graphics();
    overlay.fillStyle(0x000000, 0.7);
    overlay.fillRect(0, 0, this.cameras.main.width, this.cameras.main.height);
    overlay.setDepth(2000);

    // 结果文本
    const resultText = winnerIds.length > 0
        ? `玩家 ${winnerIds.join(', ')} 获胜!`
        : '流局';

    const text = this.add.text(centerX, centerY, resultText, {
        fontFamily: 'Microsoft YaHei',
        fontSize: 48 * scale + 'px',
        color: '#FFD700',
        fontStyle: 'bold',
        backgroundColor: '#000000',
        padding: { x: 30 * scale, y: 20 * scale }
    }).setOrigin(0.5).setDepth(2001);

    // 重启按钮
    const restartBtn = this.add.text(centerX, centerY + 100 * scale, '再来一局', {
        fontFamily: 'Microsoft YaHei',
        fontSize: 28 * scale + 'px',
        color: '#ffffff',
        backgroundColor: '#4CAF50',
        padding: { x: 20 * scale, y: 15 * scale }
    }).setOrigin(0.5).setDepth(2001).setInteractive();

    restartBtn.on('pointerdown', () => {
        this.requestRestart();
    });

    // 保存引用以便清理
    this.gameOverUI = { overlay, text, restartBtn };
}

/**
 * 请求重新开始
 */
requestRestart() {
    // 清理游戏结束UI
    if (this.gameOverUI) {
        this.gameOverUI.overlay.destroy();
        this.gameOverUI.text.destroy();
        this.gameOverUI.restartBtn.destroy();
        this.gameOverUI = null;
    }

    // 发送重启请求（通过WebSocket）
    if (this.wsManager) {
        this.wsManager.sendAction(-1, 0);  // 使用-1表示重启
    }
}
```

**Step 2: 在handleWebSocketMessage中处理game_over**

```javascript
case 'game_over':
    this.showGameOverScreen(message.winner_ids || []);
    break;
```

**Step 3: 提交游戏结束处理**

```bash
git add src/mahjong_rl/web/phaser_client/js/scenes/MahjongScene.js
git commit -m "feat(phaser-client): add game over screen and restart functionality"
```

---

## 文档和清理

### Task 8: 更新文档

**文件:**
- Create: `src/mahjong_rl/web/phaser_client/TEST_PHASE2.md`

**Step 1: 创建阶段2测试文档**

```markdown
# 阶段2测试报告 - 游戏逻辑集成

## 测试日期
2026-01-27

## 测试内容

### 后端WebSocket服务器

```bash
python src/mahjong_rl/web/simple_game_runner.py
```

### 前端Phaser客户端

```bash
cd src/mahjong_rl/web/phaser_client
python -m http.server 8080
```

### 浏览器访问

```
http://localhost:8080/index.html
```

## 已实现功能

- ✅ 前后端WebSocket通信
- ✅ 真实游戏状态同步
- ✅ 打牌动作发送到后端
- ✅ 游戏结束处理
- ✅ 重新开始功能

## 已知问题

- 待补充
```

**Step 2: 提交文档**

```bash
git add src/mahjong_rl/web/phaser_client/TEST_PHASE2.md
git commit -m "docs(phaser-client): add Phase 2 test documentation"
```

---

## 重要注意事项

1. **WebSocket URL**: 确保 `WebSocketManager.js` 中的URL与后端服务器地址匹配
2. **CORS**: FastAPI已配置允许所有来源，生产环境需限制
3. **错误处理**: WebSocket断开会自动重连（最多5次）
4. **状态同步**: 前端显示的是从后端接收的真实游戏状态

## 参考文档

- FastAPI: https://fastapi.tiangolo.com/
- WebSocket API: https://developer.mozilla.org/en-US/docs/Web/API/WebSocket
- Phaser 3: https://photonstorm.github.io/phaser3-docs/
- 武汉麻将规则: `src/mahjong_rl/rules/wuhan_mahjong_rule_engine/wuhan_mahjong_rules.md`
