# Phaser 渲染系统重构设计文档

**日期**: 2026-01-28
**作者**: Claude
**状态**: 设计中

## 问题概述

当前 Phaser 客户端存在严重的渲染问题：

1. **前端不实时更新** - AI 动作后前端完全不更新，只有手动刷新才能看到
2. **消息接收正常** - 日志显示 WebSocket 消息正确到达，`updateState()` 和 `render()` 都被调用
3. **渲染失败** - 虽然代码执行了，但界面没有变化

**根本原因**：
- 当前实现每次都销毁并重建所有游戏对象（`clear(true, true)`）
- 这种方式不符合 Phaser 的最佳实践
- 新创建的对象可能没有正确显示或被覆盖

## 设计方案

### 核心理念

**数据驱动的渲染系统**：将游戏状态（data）与显示层（view）分离，通过更新现有对象属性而不是销毁重建来刷新界面。

### 架构组件

```
┌─────────────────────────────────────────────────────────┐
│                      MahjongScene                        │
│  - 处理 WebSocket 消息                                   │
│  - 调用 GameView.update()                                │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                      GameView                            │
│  - 协调所有 PlayerView                                   │
│  - 状态对比和差量更新                                     │
│  - 管理 TileManager                                      │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
┌───────▼────┐ ┌────▼─────┐ ┌────▼─────┐
│ PlayerView  │ │PlayerView│ │PlayerView│
│  (玩家0)    │ │ (玩家1)  │ │ (玩家2)  │
│             │ └──────────┘ └──────────┘
│ - 手牌容器  │              ...
│ - 弃牌河    │
│ - 副露区    │
└──────┬──────┘
       │
┌──────▼───────────────────────────────────┐
│           TileManager                     │
│  - 对象池管理                             │
│  - 牌对象创建/复用                        │
│  - 属性更新                               │
└──────────────────────────────────────────┘
```

## 组件详细设计

### 1. TileManager

**职责**：管理所有麻将牌对象的创建、复用和更新。

**核心方法**：
```javascript
class TileManager {
    constructor(scene) {
        this.scene = scene;
        this.pool = [];      // 空闲对象池
        this.active = [];    // 活跃对象列表
    }

    // 获取或创建一张牌
    getTile(tileId, x, y, rotation, scale) {
        const tile = this.pool.pop() || this.createNewTile();
        this.updateTile(tile, tileId, x, y, rotation, scale);
        tile.setVisible(true);
        this.active.push(tile);
        return tile;
    }

    // 释放牌回对象池
    releaseTile(tile) {
        tile.setVisible(false);
        const index = this.active.indexOf(tile);
        if (index > -1) {
            this.active.splice(index, 1);
            this.pool.push(tile);
        }
    }

    // 更新牌的属性
    updateTile(tile, tileId, x, y, rotation, scale) {
        const frameIndex = getTileFrameIndex(tileId);
        tile.setTexture('tiles0', frameIndex);
        tile.setPosition(x, y);
        tile.setRotation(rotation);
        tile.setScale(scale);
    }
}
```

**对象池优化**：
- 预创建常用牌（如手牌背面）
- 使用 `setVisible(false)` 代替 `destroy()`
- 避免频繁的纹理加载

### 2. PlayerView

**职责**：单个玩家的视图容器，管理该玩家的所有牌显示。

**核心结构**：
```javascript
class PlayerView {
    constructor(scene, playerId, position, tileManager) {
        this.scene = scene;
        this.playerId = playerId;
        this.position = position;  // 0=自己, 1=下家, 2=对家, 3=上家
        this.tileManager = tileManager;

        // 容器
        this.container = scene.add.container();
        this.handTiles = [];      // 手牌对象数组
        this.riverTiles = [];     // 弃牌河对象数组
        this.meldTiles = [];      // 副露对象数组

        // 状态缓存
        this.lastHandTiles = [];
        this.lastRiverTiles = [];
        this.lastMelds = [];
    }

    // 更新手牌（差量更新）
    updateHand(handTiles, isMyTurn) {
        const scale = window.GLOBAL_SCALE_RATE;

        // 移除不再存在的牌
        while (this.handTiles.length > handTiles.length) {
            const tile = this.handTiles.pop();
            this.tileManager.releaseTile(tile);
        }

        // 添加新牌或更新现有牌
        for (let i = 0; i < handTiles.length; i++) {
            const tileId = handTiles[i];
            const x = this.calculateHandX(i);
            const y = this.calculateHandY();

            if (this.handTiles[i]) {
                // 更新现有牌
                this.tileManager.updateTile(this.handTiles[i], tileId, x, y, 0, scale);
            } else {
                // 创建新牌
                const tile = this.tileManager.getTile(tileId, x, y, 0, scale);
                this.setupTileInteraction(tile);
                this.handTiles.push(tile);
            }

            // 非自己回合时变暗
            this.handTiles[i].setAlpha(isMyTurn ? 1.0 : 0.6);
        }
    }

    // 更新弃牌河（差量更新）
    updateRiver(discardTiles) {
        // 类似逻辑...
    }

    // 更新副露区（差量更新）
    updateMelds(melds) {
        // 类似逻辑...
    }
}
```

**差量更新策略**：
- 比较新旧数组长度
- 移除多余的牌
- 添加缺失的牌
- 更新变化的牌

### 3. GameView

**职责**：主控制器，协调所有 PlayerView 并监听 gameState 变化。

**核心结构**：
```javascript
class GameView {
    constructor(scene) {
        this.scene = scene;
        this.tileManager = new TileManager(scene);
        this.playerViews = [];

        // 创建4个玩家的视图
        for (let i = 0; i < 4; i++) {
            this.playerViews.push(new PlayerView(scene, i, i, this.tileManager));
        }

        // 上一帧的状态缓存
        this.lastState = null;
    }

    // 首次渲染（创建所有对象）
    renderAll(gameState) {
        for (let i = 0; i < 4; i++) {
            const player = gameState.players[i];
            const isMyTurn = gameState.current_player_idx === i;
            this.playerViews[i].updateHand(player.hand_tiles, isMyTurn);
            this.playerViews[i].updateRiver(player.discard_tiles);
            this.playerViews[i].updateMelds(player.melds);
        }
        this.renderCenterArea(gameState);
        this.lastState = clone(gameState);
    }

    // 更新视图（差量更新）
    update(gameState, actionMask) {
        // 检查每个玩家的变化
        for (let i = 0; i < 4; i++) {
            const player = gameState.players[i];
            const lastPlayer = this.lastState?.players[i];

            if (!lastPlayer ||
                !arraysEqual(player.hand_tiles, lastPlayer.hand_tiles)) {
                const isMyTurn = gameState.current_player_idx === i;
                this.playerViews[i].updateHand(player.hand_tiles, isMyTurn);
            }

            if (!lastPlayer ||
                !arraysEqual(player.discard_tiles, lastPlayer.discard_tiles)) {
                this.playerViews[i].updateRiver(player.discard_tiles);
            }

            if (!lastPlayer ||
                !arraysEqual(player.melds, lastPlayer.melds)) {
                this.playerViews[i].updateMelds(player.melds);
            }
        }

        // 更新中央区域
        this.updateCenterArea(gameState);

        // 更新 UI（按钮等）
        this.updateUI(gameState, actionMask);

        // 保存当前状态
        this.lastState = clone(gameState);
    }
}
```

**状态对比**：
```javascript
function arraysEqual(arr1, arr2) {
    if (arr1?.length !== arr2?.length) return false;
    for (let i = 0; i < arr1.length; i++) {
        if (arr1[i] !== arr2[i]) return false;
    }
    return true;
}
```

### 4. 集成到 MahjongScene

**修改现有代码**：

1. **在 `create()` 中初始化**：
```javascript
create() {
    // ... 现有代码（背景等）

    // 创建新的渲染系统
    this.gameView = new GameView(this);

    // 初始化 WebSocket
    this.initWebSocket();

    // 创建动作按钮（不放在 gameView 中）
    this.createActionButtons();

    // 首次渲染
    this.gameView.renderAll(this.gameState);
}
```

2. **简化 `updateState()`**：
```javascript
updateState(newState, actionMask) {
    console.log('updateState 被调用');

    // 更新状态
    this.gameState = { ...this.gameState, ...newState };

    // 兼容数字状态
    if (typeof newState.current_state === 'number') {
        const stateNames = {
            0: 'INITIAL', 1: 'DRAWING', 2: 'PLAYER_DECISION',
            3: 'DISCARDING', 4: 'WAITING_RESPONSE', 5: 'GONG',
            6: 'WIN', 7: 'FLOW_DRAW'
        };
        this.gameState.current_state = stateNames[newState.current_state] || 'INITIAL';
    }

    // 移除等待消息
    this.hideWaitingMessage();

    // 使用 GameView 更新（差量更新，不销毁对象）
    this.gameView.update(this.gameState, actionMask);
}
```

3. **移除旧的 `render()` 方法**：
   - 不再需要每次清空重建
   - GameView 内部处理所有更新

## 实施计划

### 阶段 1：核心组件实现
1. 创建 `TileManager` 类
2. 创建 `PlayerView` 类
3. 创建 `GameView` 类

### 阶段 2：功能迁移
1. 实现手牌显示和更新
2. 实现弃牌河显示和更新
3. 实现副露区显示和更新

### 阶段 3：集成和优化
1. 修改 `MahjongScene` 使用新系统
2. 测试差量更新
3. 优化性能

### 阶段 4：清理
1. 移除旧的 `render()` 方法
2. 移除不再需要的代码
3. 添加单元测试

## 测试计划

1. **初始状态测试** - 刷新页面能看到正确的初始界面
2. **实时更新测试** - AI 动作后前端实时更新
3. **差量更新测试** - 只更新变化的部分
4. **性能测试** - 确保没有性能退化
5. **边缘情况测试** - 游戏结束、重连等

## 预期收益

1. **实时更新** - AI 动作立即反映到前端
2. **性能提升** - 减少对象创建/销毁
3. **代码清晰** - 分离关注点，易于维护
4. **符合最佳实践** - 遵循 Phaser 推荐模式

---

**接下来**: 实施阶段...
