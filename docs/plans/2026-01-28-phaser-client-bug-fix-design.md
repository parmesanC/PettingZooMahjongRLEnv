# Phaser Client Bug 修复设计文档

**日期**: 2026-01-28
**作者**: Claude
**状态**: 设计中

## 问题概述

phaser_client 存在以下严重影响游戏体验的 Bug：

1. **不是自己的回合也能出牌** - 交互控制缺失状态检查
2. **该自己出牌但点击后不能正常出牌** - 状态同步和按钮显示问题
3. **看不到是否能吃碰杠** - 动作按钮没有根据 action_mask 正确显示
4. **特殊杠牌动作错误** - 打出赖子/皮子/红中时发送了 DISCARD 而非对应的杠动作

这些问题频繁发生（每局多次），严重影响了游戏体验。

## 根本原因分析

### Bug 1: 动作按钮显示错误

**位置**: `MahjongScene.js:260-266` 和 `MahjongScene.js:1235`

```javascript
// showActionButtons 方法没有使用传入的参数
showActionButtons(availableActions) {
    // TODO: 根据action_mask显示可用按钮
    this.actionButtons.forEach(btn => {
        btn.setVisible(true);  // 只是简单显示所有按钮
    });
}
```

**问题**: `showActionButtons` 接收了 `action_mask` 参数但完全没有使用，只是简单地把所有按钮设为可见。

### Bug 2: 手牌交互没有状态检查

**位置**: `MahjongScene.js:933-948`

```javascript
setHandTileInteractivity(tile, index, tileId, sortedTiles) {
    tile.setInteractive();  // 没有检查是否轮到自己
    // ...
}
```

**问题**: 无论是否轮到自己出牌，手牌都是可交互的。

### Bug 3: 特殊杠牌动作类型错误（核心问题）

**问题本质**：武汉麻将中，**赖子杠、皮子杠、红中杠**都是通过"打出对应牌"来触发的：

| 牌类型 | 应该的动作类型 | 当前发送的动作 |
|--------|---------------|---------------|
| 赖子牌 | KONG_LAZY (8) | DISCARD (0) ❌ |
| 皮子牌 | KONG_SKIN (7) | DISCARD (0) ❌ |
| 红中 | KONG_RED (6) | DISCARD (0) ❌ |
| 普通牌 | DISCARD (0) | DISCARD (0) ✓ |

**位置**: `MahjongScene.js:994-1004`

```javascript
updateAfterDiscard(tileId, index, sortedTiles) {
    if (this.wsManager) {
        // 总是发送 DISCARD (0)，这是错误的！
        this.wsManager.sendAction(0, tileId);
    }
    // ...
}
```

## 修复方案

### 修复 1: 添加特殊牌动作转换逻辑

创建一个新方法来处理打牌动作的转换：

```javascript
/**
 * 获取打牌动作类型
 * 根据牌ID判断是普通打牌还是特殊杠牌
 * @param {number} tileId - 牌ID
 * @returns {number} 动作类型
 */
getDiscardActionType(tileId) {
    const { lazy_tile, skin_tiles } = this.gameState;

    // 检查是否为赖子
    if (lazy_tile !== null && isLazyTile(tileId, lazy_tile)) {
        return 8; // KONG_LAZY
    }

    // 检查是否为皮子
    if (skin_tiles.length > 0 && isSkinTile(tileId, skin_tiles)) {
        return 7; // KONG_SKIN
    }

    // 检查是否为红中
    if (isRedDragon(tileId)) {
        return 6; // KONG_RED
    }

    return 0; // DISCARD
}
```

### 修复 2: 修改 updateAfterDiscard 使用转换逻辑

```javascript
updateAfterDiscard(tileId, index, sortedTiles) {
    if (this.wsManager) {
        // 使用转换后的动作类型
        const actionType = this.getDiscardActionType(tileId);
        this.wsManager.sendAction(actionType, tileId);
    }

    console.log(`已发送动作: type=${this.getDiscardActionType(tileId)}, tileId=${tileId}`);
}
```

### 修复 3: 添加手牌交互状态检查

```javascript
setHandTileInteractivity(tile, index, tileId, sortedTiles) {
    // 只在轮到自己时设置交互
    const isMyTurn = this.gameState.current_player_idx === this.playerId;

    tile.setInteractive();

    // 添加交互反馈
    tile.on('pointerover', () => {
        if (isMyTurn) {
            tile.y -= 15 * window.GLOBAL_SCALE_RATE;
        }
    });

    tile.on('pointerout', () => {
        if (isMyTurn) {
            tile.y += 15 * window.GLOBAL_SCALE_RATE;
        }
    });

    tile.on('pointerdown', () => {
        // 检查是否轮到自己
        if (!isMyTurn) {
            console.log('不是你的回合');
            this.showErrorNotification('不是你的回合');
            return;
        }

        console.log(`Tile ${index} (${tileId}) clicked`);
        this.playDiscardAnimation(tile, tileId, index, sortedTiles);
    });
}
```

### 修复 4: 移除无用的 action_prompt 处理

由于后端不发送 `action_prompt` 消息，移除无用的处理分支：

```javascript
handleWebSocketMessage(message) {
    switch (message.type) {
        case 'game_state':
            if (message.observer_player_idx !== undefined && message.observer_player_idx !== this.playerId) {
                console.log(`跳过玩家${message.observer_player_idx}视角的状态更新`);
                return;
            }
            this.updateState(message.state, message.action_mask);
            break;

        case 'initial_state':
            if (message.state) {
                this.updateState(message.state, message.action_mask);
            }
            break;

        // 移除 action_prompt 处理（后端不发送此消息）

        case 'game_over':
            this.showGameOverScreen(message.winner_ids || []);
            break;

        case 'error':
            console.error('收到错误:', message.message);
            this.showErrorNotification(message.message);
            break;

        case 'info':
            console.log('收到消息:', message.message);
            this.showInfoNotification(message.message);
            break;

        default:
            console.log('未知消息类型:', message.type);
    }
}
```

### 修复 5: 优化按钮更新逻辑

确保 `updateState` 正确更新按钮：

```javascript
updateState(newState, actionMask) {
    // 移除等待消息
    this.hideWaitingMessage();

    // 兼容后端返回的状态格式
    if (newState.current_state !== undefined) {
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

    // 根据action_mask显示/隐藏动作按钮
    const isMyTurn = this.gameState.current_player_idx === this.playerId;

    if (isMyTurn && actionMask) {
        this.updateActionButtons(actionMask);
    } else {
        this.hideActionButtons();
    }

    // 重新渲染
    this.render();
}
```

### 修复 6: 添加视觉反馈

在 `renderHandTiles` 中添加非活动状态视觉提示：

```javascript
renderSelfHand(handTiles, scale) {
    // ... 现有代码

    const isMyTurn = this.gameState.current_player_idx === this.playerIdx;

    for (let i = 0; i < sortedTiles.length; i++) {
        // ...

        const tile = this.add.image(x, y, 'tiles4', frameIndex)
            .setScale(tileScale)
            .setDepth(1000);

        // 非自己回合时，手牌变暗
        if (!isMyTurn) {
            tile.setAlpha(0.6);
        }

        // ...
    }
}
```

## 实施计划

### 第一步：修复特殊杠牌动作转换（高优先级）
1. 添加 `getDiscardActionType` 方法
2. 修改 `updateAfterDiscard` 使用转换逻辑

### 第二步：添加手牌交互状态检查（高优先级）
1. 修改 `setHandTileInteractivity` 添加回合检查
2. 添加错误提示

### 第三步：优化按钮更新逻辑（中优先级）
1. 修复 `updateState` 中的按钮更新
2. 移除无用的 `action_prompt` 处理

### 第四步：添加视觉反馈（低优先级）
1. 非活动状态手牌变暗
2. 添加当前回合指示器

## 测试计划

1. 测试打出赖子牌发送 KONG_LAZY 动作
2. 测试打出皮子牌发送 KONG_SKIN 动作
3. 测试打出红中发送 KONG_RED 动作
4. 测试打出普通牌发送 DISCARD 动作
5. 测试非自己回合时手牌不可点击
6. 测试自己回合时手牌可正常点击
7. 测试动作按钮根据 action_mask 正确显示/隐藏

---

**接下来**: 实施修复...
