# Phaser 渲染系统重构实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 重构 Phaser 客户端渲染系统，使用对象池和差量更新替代当前的销毁/重建模式，解决前端不实时更新的问题。

**架构:** 数据驱动的渲染系统，将游戏状态与显示层分离。使用 TileManager 管理对象池，PlayerView 管理每个玩家的视图，GameView 协调所有更新。

**技术栈:** Phaser 3, JavaScript ES6, WebSocket

---

## 任务结构概览

1. 创建 TileManager 类（对象池管理）
2. 创建 PlayerView 类（单玩家视图）
3. 创建 GameView 类（主控制器）
4. 修改 MahjongScene 集成新系统
5. 测试和验证
6. 清理旧代码

---

### Task 1: 创建 TileManager 类

**Files:**
- Create: `src/mahjong_rl/web/phaser_client/js/utils/TileManager.js`

**Step 1: 创建 TileManager 类文件**

创建 `src/mahjong_rl/web/phaser_client/js/utils/TileManager.js`：

```javascript
/**
 * TileManager - 麻将牌对象池管理器
 * 负责牌对象的创建、复用和更新，避免频繁的销毁和重建
 */

import { getTileFrameIndex } from './TileUtils.js';

export class TileManager {
    constructor(scene) {
        this.scene = scene;
        this.pool = [];      // 空闲对象池
        this.active = [];    // 活跃对象列表
    }

    /**
     * 获取或创建一张牌
     */
    getTile(tileId, x, y, rotation = 0, scale = 1, texture = 'tiles0') {
        const tile = this.pool.pop() || this.createNewTile(texture);
        this.updateTile(tile, tileId, x, y, rotation, scale, texture);
        tile.setVisible(true);
        this.active.push(tile);
        return tile;
    }

    /**
     * 创建新的牌对象
     */
    createNewTile(texture = 'tiles0') {
        const tile = this.scene.add.image(0, 0, texture, 0);
        tile.setInteractive();
        return tile;
    }

    /**
     * 释放牌回对象池
     */
    releaseTile(tile) {
        if (!tile) return;

        tile.setVisible(false);
        const index = this.active.indexOf(tile);
        if (index > -1) {
            this.active.splice(index, 1);
            this.pool.push(tile);
        }
    }

    /**
     * 更新牌的属性
     */
    updateTile(tile, tileId, x, y, rotation = 0, scale = 1, texture = 'tiles0') {
        if (!tile) return;

        const frameIndex = getTileFrameIndex(tileId);

        // 更新纹理（如果需要）
        if (tile.texture.key !== texture) {
            tile.setTexture(texture, frameIndex);
        } else {
            tile.setFrame(frameIndex);
        }

        // 更新位置和变换
        tile.setPosition(x, y);
        tile.setRotation(rotation);
        tile.setScale(scale);
    }

    /**
     * 清空所有牌
     */
    clear() {
        // 将所有活跃对象移回池中
        while (this.active.length > 0) {
            const tile = this.active.pop();
            tile.setVisible(false);
            this.pool.push(tile);
        }
    }

    /**
     * 获取活跃对象数量
     */
    getActiveCount() {
        return this.active.length;
    }

    /**
     * 获取池中空闲对象数量
     */
    getPoolCount() {
        return this.pool.length;
    }
}
```

**Step 2: 验证文件创建**

Run: `ls -la src/mahjong_rl/web/phaser_client/js/utils/TileManager.js`
Expected: 文件存在

**Step 3: 提交**

```bash
git add src/mahjong_rl/web/phaser_client/js/utils/TileManager.js
git commit -m "feat(phaser): add TileManager class for object pool management

- Implement getTile/releaseTile for object reuse
- Add updateTile method for property updates
- Use setVisible(false) instead of destroy() for performance
"
```

---

### Task 2: 创建 PlayerView 类（框架）

**Files:**
- Create: `src/mahjong_rl/web/phaser_client/js/views/PlayerView.js`

**Step 1: 创建 PlayerView 类文件**

创建 `src/mahjong_rl/web/phaser_client/js/views/PlayerView.js`：

```javascript
/**
 * PlayerView - 单个玩家的视图容器
 * 管理该玩家的手牌、弃牌河、副露区的显示和更新
 */

import { getTileFrameIndex, getRelativePosition } from '../utils/TileUtils.js';
import { isLazyTile, isSkinTile, isRedDragon } from '../utils/TileUtils.js';
import { COLORS } from '../utils/Constants.js';

export class PlayerView {
    constructor(scene, playerId, position, tileManager) {
        this.scene = scene;
        this.playerId = playerId;
        this.position = position;  // 0=自己, 1=下家, 2=对家, 3=上家
        this.tileManager = tileManager;

        // 牌对象数组
        this.handTiles = [];
        this.riverTiles = [];
        this.meldTiles = [];

        // 状态缓存（用于差量更新）
        this.lastHandTiles = [];
        this.lastRiverTiles = [];
        this.lastMelds = [];
    }

    /**
     * 更新手牌
     */
    updateHand(handTiles, isMyTurn, selfPlayerIdx) {
        const scale = window.GLOBAL_SCALE_RATE;
        const relativePos = getRelativePosition(this.playerId, selfPlayerIdx);

        // 只渲染自己的手牌（正面），其他玩家显示背面
        if (relativePos === 0) {
            this.updateSelfHand(handTiles, scale, isMyTurn);
        } else if (relativePos === 1) {
            this.updateRightHand(handTiles.length, scale);
        } else if (relativePos === 2) {
            this.updateOppoHand(handTiles.length, scale);
        } else if (relativePos === 3) {
            this.updateLeftHand(handTiles.length, scale);
        }

        this.lastHandTiles = [...handTiles];
    }

    /**
     * 更新自己的手牌（正面显示）
     */
    updateSelfHand(handTiles, scale, isMyTurn) {
        let x = 40 * scale;
        const y = this.scene.cameras.main.height - 60 * scale;
        const tileScale = scale * 0.45;

        // 先排序
        const sortedTiles = [...handTiles].sort((a, b) => a - b);

        // 移除多余的牌
        while (this.handTiles.length > sortedTiles.length) {
            const tile = this.handTiles.pop();
            this.tileManager.releaseTile(tile);
        }

        // 更新或创建牌
        for (let i = 0; i < sortedTiles.length; i++) {
            x += (117 + 2) * tileScale;
            const tileId = sortedTiles[i];

            if (this.handTiles[i]) {
                // 更新现有牌
                this.tileManager.updateTile(
                    this.handTiles[i],
                    tileId,
                    x,
                    y,
                    0,
                    tileScale,
                    'tiles4'
                );
            } else {
                // 创建新牌
                const tile = this.tileManager.getTile(
                    tileId,
                    x,
                    y,
                    0,
                    tileScale,
                    'tiles4'
                );
                tile.setDepth(1000 + i);
                this.handTiles.push(tile);
            }

            // 非自己回合时变暗
            this.handTiles[i].setAlpha(isMyTurn ? 1.0 : 0.6);
        }
    }

    /**
     * 更新右侧手牌（背面）
     */
    updateRightHand(length, scale) {
        const tileScale = scale * 0.5;
        const bottom = this.scene.cameras.main.height - 450 * tileScale;
        const imageWidth = 57 * tileScale;
        const imageHeight = 132 * tileScale;

        // 移除多余的牌
        while (this.handTiles.length > length) {
            const tile = this.handTiles.pop();
            this.tileManager.releaseTile(tile);
        }

        // 添加或更新牌
        for (let i = this.handTiles.length; i < length; i++) {
            const x = this.scene.cameras.main.width - imageWidth;
            const y = bottom - (imageHeight - 60 * tileScale) * (i + 1);

            if (this.handTiles[i]) {
                this.tileManager.updateTile(
                    this.handTiles[i],
                    30,  // 背面frame
                    x,
                    y,
                    0,
                    tileScale,
                    'tiles2'
                );
                this.handTiles[i].setDepth(i + 1);
            } else {
                const tile = this.tileManager.getTile(
                    30,  // 背面frame
                    x,
                    y,
                    0,
                    tileScale,
                    'tiles2'
                );
                tile.setDepth(i + 1);
                this.handTiles.push(tile);
            }
        }
    }

    /**
     * 更新对家手牌（背面）
     */
    updateOppoHand(length, scale) {
        // TODO: 实现对家手牌更新
        this.updateGenericHand(length, scale, 'tiles2', 30);
    }

    /**
     * 更新左侧手牌（背面）
     */
    updateLeftHand(length, scale) {
        // TODO: 实现左侧手牌更新
        this.updateGenericHand(length, scale, 'tiles2', 30);
    }

    /**
     * 通用手牌更新（背面）
     */
    updateGenericHand(length, scale, texture, frame) {
        while (this.handTiles.length > length) {
            const tile = this.handTiles.pop();
            this.tileManager.releaseTile(tile);
        }
        // 简化实现，后续完善
    }

    /**
     * 更新弃牌河
     */
    updateRiver(discardTiles, selfPlayerIdx) {
        // TODO: 实现弃牌河更新
        while (this.riverTiles.length > discardTiles.length) {
            const tile = this.riverTiles.pop();
            this.tileManager.releaseTile(tile);
        }
        this.lastRiverTiles = [...discardTiles];
    }

    /**
     * 更新副露区
     */
    updateMelds(melds, selfPlayerIdx) {
        // TODO: 实现副露区更新
        while (this.meldTiles.length > melds.length) {
            const tile = this.meldTiles.pop();
            this.tileManager.releaseTile(tile);
        }
        this.lastMelds = [...melds];
    }

    /**
     * 清空所有牌
     */
    clear() {
        this.handTiles.forEach(tile => this.tileManager.releaseTile(tile));
        this.riverTiles.forEach(tile => this.tileManager.releaseTile(tile));
        this.meldTiles.forEach(tile => this.tileManager.releaseTile(tile));
        this.handTiles = [];
        this.riverTiles = [];
        this.meldTiles = [];
    }
}
```

**Step 2: 创建 views 目录**

Run: `mkdir -p src/mahjong_rl/web/phaser_client/js/views`
Expected: 目录创建成功

**Step 3: 提交**

```bash
git add src/mahjong_rl/web/phaser_client/js/views/PlayerView.js
git commit -m "feat(phaser): add PlayerView class for individual player rendering

- Add updateHand method with support for all 4 positions
- Implement difference-based updates (add/remove/update tiles)
- Add updateRiver and updateMelds stubs
- Support visual feedback for non-active turns (alpha 0.6)
"
```

---

### Task 3: 创建 GameView 类

**Files:**
- Create: `src/mahjong_rl/web/phaser_client/js/views/GameView.js`

**Step 1: 创建 GameView 类文件**

创建 `src/mahjong_rl/web/phaser_client/js/views/GameView.js`：

```javascript
/**
 * GameView - 游戏视图主控制器
 * 协调所有 PlayerView，管理中央区域和 UI 更新
 */

import { TileManager } from '../utils/TileManager.js';
import { PlayerView } from './PlayerView.js';
import { GAME_STATES } from '../utils/Constants.js';

export class GameView {
    constructor(scene, actionButtonsCallback) {
        this.scene = scene;
        this.actionButtonsCallback = actionButtonsCallback;

        // 创建 TileManager
        this.tileManager = new TileManager(scene);

        // 创建 4 个 PlayerView
        this.playerViews = [];
        for (let i = 0; i < 4; i++) {
            this.playerViews.push(new PlayerView(scene, i, i, this.tileManager));
        }

        // 状态缓存
        this.lastState = null;
        this.centralElements = {};  // 中央区域元素缓存
    }

    /**
     * 首次渲染（创建所有对象）
     */
    renderAll(gameState) {
        console.log('GameView.renderAll() - 首次渲染');

        // 渲染所有玩家
        for (let i = 0; i < 4; i++) {
            const player = gameState.players[i];
            const isMyTurn = gameState.current_player_idx === i;
            this.playerViews[i].updateHand(player.hand_tiles || [], isMyTurn, 0);
            this.playerViews[i].updateRiver(player.discard_tiles || [], 0);
            this.playerViews[i].updateMelds(player.melds || [], 0);
        }

        // 渲染中央区域
        this.renderCenterArea(gameState);

        // 保存状态
        this.lastState = JSON.parse(JSON.stringify(gameState));
    }

    /**
     * 更新视图（差量更新）
     */
    update(gameState, actionMask) {
        console.log('GameView.update() - 差量更新');

        // 首次调用，使用 renderAll
        if (!this.lastState) {
            this.renderAll(gameState);
            this.updateUI(gameState, actionMask);
            return;
        }

        // 检查每个玩家的变化
        for (let i = 0; i < 4; i++) {
            const player = gameState.players[i];
            const lastPlayer = this.lastState.players[i];

            // 检查手牌变化
            if (!this.arraysEqual(player.hand_tiles || [], lastPlayer?.hand_tiles || [])) {
                const isMyTurn = gameState.current_player_idx === i;
                this.playerViews[i].updateHand(player.hand_tiles || [], isMyTurn, 0);
            }

            // 检查弃牌河变化
            if (!this.arraysEqual(player.discard_tiles || [], lastPlayer?.discard_tiles || [])) {
                this.playerViews[i].updateRiver(player.discard_tiles || [], 0);
            }

            // 检查副露变化
            if (!this.meldsEqual(player.melds || [], lastPlayer?.melds || [])) {
                this.playerViews[i].updateMelds(player.melds || [], 0);
            }
        }

        // 更新中央区域
        this.updateCenterArea(gameState);

        // 更新 UI
        this.updateUI(gameState, actionMask);

        // 保存当前状态
        this.lastState = JSON.parse(JSON.stringify(gameState));
    }

    /**
     * 渲染中央区域
     */
    renderCenterArea(gameState) {
        const centerX = this.scene.cameras.main.width / 2;
        const centerY = this.scene.cameras.main.height / 2;
        const scale = window.GLOBAL_SCALE_RATE;

        // 创建/更新赖子指示器
        if (gameState.lazy_tile !== null) {
            this.renderLazyIndicator(gameState.lazy_tile, centerX, centerY, scale);
        }

        // 创建/更新皮子指示器
        if (gameState.skin_tiles && gameState.skin_tiles.length > 0) {
            this.renderSkinIndicator(gameState.skin_tiles, centerX, centerY, scale);
        }

        // 创建/更新剩余牌数
        this.renderWallCount(gameState.wall_count, centerX, centerY, scale);

        // 创建/更新当前玩家指示器
        this.renderCurrentPlayerIndicator(gameState.current_player_idx, centerX, centerY, scale);

        // 创建/更新游戏状态文本
        this.renderGameStateText(gameState.current_state, centerX, centerY, scale);
    }

    /**
     * 更新中央区域
     */
    updateCenterArea(gameState) {
        // 简化实现：直接重新渲染
        // 后续可优化为差量更新
        this.renderCenterArea(gameState);
    }

    /**
     * 渲染赖子指示器
     */
    renderLazyIndicator(lazyTileId, centerX, centerY, scale) {
        // TODO: 实现赖子指示器渲染
    }

    /**
     * 渲染皮子指示器
     */
    renderSkinIndicator(skinTiles, centerX, centerY, scale) {
        // TODO: 实现皮子指示器渲染
    }

    /**
     * 渲染剩余牌数
     */
    renderWallCount(wallCount, centerX, centerY, scale) {
        // TODO: 实现剩余牌数渲染
    }

    /**
     * 渲染当前玩家指示器
     */
    renderCurrentPlayerIndicator(currentPlayerIdx, centerX, centerY, scale) {
        // TODO: 实现当前玩家指示器渲染
    }

    /**
     * 渲染游戏状态文本
     */
    renderGameStateText(currentState, centerX, centerY, scale) {
        // TODO: 实现游戏状态文本渲染
    }

    /**
     * 更新 UI（按钮等）
     */
    updateUI(gameState, actionMask) {
        if (this.actionButtonsCallback) {
            this.actionButtonsCallback(gameState, actionMask);
        }
    }

    /**
     * 比较两个数组是否相等
     */
    arraysEqual(arr1, arr2) {
        if (!arr1 || !arr2) return arr1 === arr2;
        if (arr1.length !== arr2.length) return false;
        for (let i = 0; i < arr1.length; i++) {
            if (arr1[i] !== arr2[i]) return false;
        }
        return true;
    }

    /**
     * 比较两个副落数组是否相等
     */
    meldsEqual(melds1, melds2) {
        if (!melds1 || !melds2) return melds1 === melds2;
        if (melds1.length !== melds2.length) return false;
        for (let i = 0; i < melds1.length; i++) {
            const m1 = melds1[i];
            const m2 = melds2[i];
            if (m1.action_type !== m2.action_type ||
                m1.from_player !== m2.from_player ||
                !this.arraysEqual(m1.tiles, m2.tiles)) {
                return false;
            }
        }
        return true;
    }

    /**
     * 清空所有视图
     */
    clear() {
        this.playerViews.forEach(pv => pv.clear());
        this.centralElements = {};
    }
}
```

**Step 2: 提交**

```bash
git add src/mahjong_rl/web/phaser_client/js/views/GameView.js
git commit -m "feat(phaser): add GameView class as main rendering controller

- Add renderAll() for initial rendering
- Add update() for difference-based updates
- Implement state comparison (arraysEqual, meldsEqual)
- Add stubs for center area rendering
- Coordinate all PlayerView instances
"
```

---

### Task 4: 修改 MahjongScene 集成新系统

**Files:**
- Modify: `src/mahjong_rl/web/phaser_client/js/scenes/MahjongScene.js`

**Step 1: 在 create() 方法中初始化 GameView**

在 `MahjongScene.js` 的 `create()` 方法开始添加：

```javascript
create() {
    // 创建背景
    this.createBackground();

    // 创建新的渲染系统
    this.gameView = new GameView(this, (gameState, actionMask) => {
        // 回调：更新动作按钮
        const isMyTurn = gameState.current_player_idx === this.playerId;
        if (isMyTurn && actionMask) {
            this.updateActionButtons(actionMask);
        } else {
            this.hideActionButtons();
        }
    });

    // 创建图层组（保留用于其他用途）
    this.createLayers();

    // 创建游戏桌面
    this.createGameBoard();

    // 初始化 WebSocket
    this.initWebSocket();

    // 创建动作按钮
    this.createActionButtons();

    // 显示等待连接提示
    this.showWaitingMessage();
}
```

**Step 2: 修改 updateState() 方法**

将现有的 `updateState()` 方法简化为：

```javascript
updateState(newState, actionMask) {
    console.log('updateState 被调用 - 使用 GameView');

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

    // 更新状态
    this.gameState = { ...this.gameState, ...newState };

    // 使用 GameView 更新（差量更新）
    this.gameView.update(this.gameState, actionMask);
}
```

**Step 3: 注释掉旧的 render() 方法**

在 `render()` 方法前添加注释：

```javascript
/**
 * @deprecated 使用 GameView.update() 代替
 * 保留此方法用于调试
 */
/*
render() {
    // ... 原有代码
}
*/
```

**Step 4: 添加 import 语句**

在文件顶部添加：

```javascript
import { GameView } from '../views/GameView.js';
```

**Step 5: 提交**

```bash
git add src/mahjong_rl/web/phaser_client/js/scenes/MahjongScene.js
git commit -m "refactor(phaser): integrate GameView into MahjongScene

- Initialize GameView in create()
- Simplify updateState() to use GameView.update()
- Add callback for action button updates
- Keep old render() for debugging (deprecated)
"
```

---

### Task 5: 测试基础功能

**Files:**
- Test: Browser console + Manual testing

**Step 1: 启动后端服务器**

Run:
```bash
cd D:\DATA\Python_Project\Code\PettingZooRLENVMahjong
python src/mahjong_rl/web/simple_game_runner.py --human 1 --ai-delay 1.0
```

Expected: 服务器启动在 8011 端口

**Step 2: 启动前端服务器**

Run:
```bash
cd src/mahjong_rl/web/phaser_client
python -m http.server 8080
```

Expected: 前端服务器启动在 8080 端口

**Step 3: 打开浏览器**

Navigate: `http://localhost:8080/index.html`

Open: Browser DevTools (F12) → Console tab

**Step 4: 观察初始渲染**

Expected:
- 控制台显示 "GameView.renderAll() - 首次渲染"
- 能看到初始游戏界面（手牌、赖子等）
- 没有错误

**Step 5: 测试实时更新**

Trigger: 等待 AI 动作

Expected:
- 控制台显示 "GameView.update() - 差量更新"
- 前端界面实时更新（不需要手动刷新）
- 能看到 AI 打出的牌

**Step 6: 如果失败，记录问题**

如果前端不更新，记录：
- 控制台错误信息
- update() 是否被调用
- PlayerView.updateHand() 是否被调用

**Step 7: 提交测试结果**

Create: `docs/plans/2026-01-28-phaser-rendering-test-results.md`

记录测试结果和发现的问题。

```bash
git add docs/plans/2026-01-28-phaser-rendering-test-results.md
git commit -m "test(phaser): document initial rendering system test results"
```

---

### Task 6: 完善中央区域渲染

**Files:**
- Modify: `src/mahjong_rl/web/phaser_client/js/views/GameView.js`

**Step 1: 实现 renderLazyIndicator()**

```javascript
renderLazyIndicator(lazyTileId, centerX, centerY, scale) {
    const frameIndex = getTileFrameIndex(lazyTileId);

    if (!this.centralElements.lazyIndicator) {
        const tile = this.scene.add.image(
            centerX - 80 * scale,
            centerY,
            'tiles0',
            frameIndex
        ).setScale(scale * 0.4).setDepth(600);

        const label = this.scene.add.text(
            centerX - 80 * scale,
            centerY + 50 * scale,
            '赖子',
            {
                fontFamily: 'Microsoft YaHei',
                fontSize: 16 * scale + 'px',
                color: '#FFD700',
                fontStyle: 'bold'
            }
        ).setOrigin(0.5).setDepth(600);

        this.centralElements.lazyIndicator = { tile, label };
    } else {
        // 更新赖子牌
        const frameIndex = getTileFrameIndex(lazyTileId);
        this.centralElements.lazyIndicator.tile.setFrame(frameIndex);
    }
}
```

**Step 2: 实现其他中央区域方法**

类似地实现：
- `renderSkinIndicator()`
- `renderWallCount()`
- `renderCurrentPlayerIndicator()`
- `renderGameStateText()`

**Step 3: 提交**

```bash
git add src/mahjong_rl/web/phaser_client/js/views/GameView.js
git commit -m "feat(phaser): implement center area rendering in GameView

- Add renderLazyIndicator for lazy tile display
- Add renderSkinIndicator for skin tiles display
- Add renderWallCount for remaining tiles count
- Add renderCurrentPlayerIndicator for turn indication
- Add renderGameStateText for game state display
"
```

---

### Task 7: 清理和优化

**Files:**
- Modify: `src/mahjong_rl/web/phaser_client/js/scenes/MahjongScene.js`

**Step 1: 移除旧的 render() 方法**

删除或注释掉整个 `render()` 方法及其相关辅助方法：
- `renderPlayer()`
- `renderHandTiles()`
- `renderRiver()`
- `renderMelds()`
- `renderSelfHand()`
- `renderRightHand()`
- `renderOppoHand()`
- `renderLeftHand()`
- `renderSelfRiver()`
- `renderRightRiver()`
- `renderOppoRiver()`
- `renderLeftRiver()`
- `renderSelfMelds()`
- `renderRightMelds()`
- `renderOppoMelds()`
- `renderLeftMelds()`
- `renderCenterArea()`
- `renderLazyIndicator()`
- `renderSkinIndicator()`
- `renderWallCount()`
- `renderCurrentPlayerIndicator()`
- `renderGameState()`

**Step 2: 移除不再需要的 import**

清理不再使用的工具函数导入。

**Step 3: 移除 createLayers() 和相关图层管理**

GameView 不再需要图层系统，可以移除。

**Step 4: 测试清理后的代码**

重复 Task 5 的测试步骤。

**Step 5: 提交**

```bash
git add src/mahjong_rl/web/phaser_client/js/scenes/MahjongScene.js
git commit -m "refactor(phaser): remove old rendering code after GameView migration

- Remove old render() method and all helper methods
- Remove unused layer management code
- Clean up imports
- Code is now fully migrated to GameView system
"
```

---

## 测试检查清单

完成实施后，验证以下功能：

- [ ] 初始状态正确显示
- [ ] AI 动作后前端实时更新
- [ ] 手牌正确显示（正面）
- [ ] 其他玩家手牌正确显示（背面）
- [ ] 弃牌河正确更新
- [ ] 副露区正确显示
- [ ] 赖子/皮子指示器正确
- [ ] 当前玩家指示器正确
- [ ] 动作按钮根据 action_mask 正确显示
- [ ] 非自己回合时手牌变暗
- [ ] 点击手牌有悬停效果
- [ ] 游戏结束正确显示

---

## 故障排除

如果前端仍然不更新：

1. **检查控制台日志**
   - 确认 `GameView.update()` 被调用
   - 确认 `PlayerView.updateHand()` 被调用
   - 查看是否有 JavaScript 错误

2. **检查对象状态**
   - 在 `updateSelfHand()` 中添加 `console.log('handTiles:', handTiles)`
   - 确认 `handTiles` 数据正确

3. **检查 Phaser 对象**
   - 在 `getTile()` 中添加 `console.log('Tile created/reused:', tile)`
   - 确认对象被正确创建或复用

4. **检查可见性和深度**
   - 确认 `setVisible(true)` 被调用
   - 确认 `setDepth()` 值正确
   - 确认没有其他对象遮挡

---

## 完成

实施完成后：
1. 更新设计文档标记为完成
2. 创建测试报告
3. 提交最终代码
