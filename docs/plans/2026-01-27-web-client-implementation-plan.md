# 武汉麻将网页客户端实施计划 - 阶段1：基础渲染

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 基于 Phaser.js 搭建武汉麻将网页客户端的基础渲染系统，实现四方位手牌显示、弃牌河、副露区和武汉麻将特殊牌（赖子/皮/红中）的视觉标识。

**架构:** 前端使用 Phaser.js 游戏引擎，复用 Mahjong-AI 项目的麻将牌 sprite sheet。后端暂时使用本地模拟数据，为后续阶段集成真实游戏逻辑做准备。

**技术栈:**
- 前端: Phaser.js 3.x, HTML5, CSS3, JavaScript (ES6+)
- 构建: 无需构建工具，直接引入 Phaser.js CDN
- 测试: 手动浏览器测试
- 参考: Mahjong-AI 项目 (`D:\DATA\Python_Project\Code\PettingZooRLENVMahjong\Mahjong-AI\online_game\web_client`)

---

## 前置准备

### Task 0: 项目初始化和环境准备

**文件:**
- Create: `src/mahjong_rl/web/phaser_client/index.html`
- Create: `src/mahjong_rl/web/phaser_client/css/styles.css`
- Create: `src/mahjong_rl/web/phaser_client/js/main.js`
- Create: `src/mahjong_rl/web/phaser_client/js/scenes/MahjongScene.js`
- Create: `src/mahjong_rl/web/phaser_client/js/utils/TileUtils.js`
- Create: `src/mahjong_rl/web/phaser_client/js/utils/Constants.js`

**Step 1: 创建项目目录结构**

```bash
# 在 src/mahjong_rl/web/ 下创建 phaser_client 目录
mkdir -p src/mahjong_rl/web/phaser_client/js/scenes
mkdir -p src/mahjong_rl/web/phaser_client/js/utils
mkdir -p src/mahjong_rl/web/phaser_client/css
mkdir -p src/mahjong_rl/web/phaser_client/assets/images
```

**Step 2: 验证目录创建成功**

运行: `ls -la src/mahjong_rl/web/phaser_client/`
Expected: 显示 css/, js/, assets/ 目录

**Step 3: 提交目录结构**

```bash
git add src/mahjong_rl/web/phaser_client/
git commit -m "feat(phaser-client): create project directory structure"
```

---

## 模块1: 核心常量和工具函数

### Task 1: 创建游戏常量定义

**文件:**
- Create: `src/mahjong_rl/web/phaser_client/js/utils/Constants.js`

**Step 1: 创建常量文件**

```javascript
/**
 * 武汉麻将网页客户端 - 游戏常量
 * 与后端 constants.py 保持一致
 */

// ============ 牌相关常量 ============
const TILE_TYPES = {
    WAN: 0,      // 万子 (0-8)
    TIAO: 1,     // 条子 (9-17)
    TONG: 2,     // 筒子 (18-26)
    FENG: 3,     // 风牌 (27-30: 东、南、西、北)
    JIAN: 4      // 箭牌 (31-33: 红中、发财、白板)
};

// 牌ID范围
const TILE_RANGES = {
    WAN: { start: 0, end: 8 },
    TIAO: { start: 9, end: 17 },
    TONG: { start: 18, end: 26 },
    FENG: { start: 27, end: 30 },
    JIAN: { start: 31, end: 33 }
};

// 风牌映射
const WIND_TILES = {
    EAST: 27,
    SOUTH: 28,
    WEST: 29,
    NORTH: 30
};

// 箭牌映射
const DRAGON_TILES = {
    RED: 31,    // 红中
    GREEN: 32,  // 发财
    WHITE: 33   // 白板
};

// ============ 动作类型常量 ============
// 与后端 ActionType 枚举对应
const ACTION_TYPES = {
    DISCARD: 0,
    CHOW: 1,
    PONG: 2,
    KONG_EXPOSED: 3,
    KONG_SUPPLEMENT: 4,
    KONG_CONCEALED: 5,
    KONG_RED: 6,
    KONG_SKIN: 7,
    KONG_LAZY: 8,
    WIN: 9,
    PASS: 10
};

// ============ 游戏状态常量 ============
// 与后端 GameStateType 枚举对应
const GAME_STATES = {
    INITIAL: 'INITIAL',
    DRAWING: 'DRAWING',
    PLAYER_DECISION: 'PLAYER_DECISION',
    DISCARDING: 'DISCARDING',
    MELD_DECISION: 'MELD_DECISION',
    WAITING_RESPONSE: 'WAITING_RESPONSE',
    PROCESSING_MELD: 'PROCESSING_MELD',
    GONG: 'GONG',
    DRAWING_AFTER_GONG: 'DRAWING_AFTER_GONG',
    WAIT_ROB_KONG: 'WAIT_ROB_KONG',
    WIN: 'WIN',
    FLOW_DRAW: 'FLOW_DRAW'
};

// ============ 视觉常量 ============
// 屏幕适配
const DEFAULT_DIMENSION = 889;
let GLOBAL_SCALE_RATE = 1;

// 麻将牌尺寸 (参考 Mahjong-AI)
const TILE_SIZES = {
    width0: 72,
    height0: 109,
    width1: 89,
    height1: 97
};

// 颜色定义
const COLORS = {
    LAZY_BORDER: 0xFFD700,    // 金色边框
    LAZY_GLOW: 0xFFAA00,      // 金色光晕
    SKIN_BORDER: 0xC0C0C0,    // 银色边框
    RED_GLOW: 0xFF0000,       // 红色光效
    TEXT_SHADOW: 0x000000     // 文字阴影
};

// ============ 布局常量 ============
const LAYOUT = {
    CENTER_X: 0.5,
    CENTER_Y: 0.5,
    BOARD_SIZE: 250,
    HAND_OFFSET: 40,
    RIVER_OFFSET: 10
};

// 导出常量
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        TILE_TYPES,
        TILE_RANGES,
        WIND_TILES,
        DRAGON_TILES,
        ACTION_TYPES,
        GAME_STATES,
        TILE_SIZES,
        COLORS,
        LAYOUT
    };
}
```

**Step 2: 验证文件创建成功**

运行: `cat src/mahjong_rl/web/phaser_client/js/utils/Constants.js`
Expected: 显示完整的常量定义

**Step 3: 提交常量文件**

```bash
git add src/mahjong_rl/web/phaser_client/js/utils/Constants.js
git commit -m "feat(phaser-client): add game constants matching backend"
```

---

### Task 2: 创建工具函数

**文件:**
- Create: `src/mahjong_rl/web/phaser_client/js/utils/TileUtils.js`

**Step 1: 创建工具函数文件**

```javascript
/**
 * 麻将牌工具函数
 * 处理牌ID到sprite frame的映射等
 */

import { TILE_TYPES, TILE_RANGES, WIND_TILES, DRAGON_TILES } from './Constants.js';

/**
 * 获取牌的sprite frame索引
 * 参考 Mahjong-AI 的 getTileFrameIndex 方法
 *
 * @param {number} tileId - 牌ID (0-33)
 * @returns {number} sprite frame索引
 */
function getTileFrameIndex(tileId) {
    // 红色牌（akas）特殊处理
    const akas = [16, 52, 88];

    if (akas.includes(tileId)) {
        // 红色牌使用特殊frame
        return akas.indexOf(tileId) * 10;
    }

    // 计算行和列
    const row = Math.floor(tileId / 9);
    const col = tileId % 9 + 1;

    // frameIndex = row * 10 + col
    return row * 10 + col;
}

/**
 * 判断是否为数字牌
 * @param {number} tileId - 牌ID
 * @returns {boolean}
 */
function isNumberTile(tileId) {
    return tileId >= 0 && tileId <= 26;
}

/**
 * 判断是否为字牌
 * @param {number} tileId - 牌ID
 * @returns {boolean}
 */
function isHonorTile(tileId) {
    return tileId >= 27 && tileId <= 33;
}

/**
 * 获取牌的类型
 * @param {number} tileId - 牌ID
 * @returns {string} 牌类型 (WAN/TIAO/TONG/FENG/JIAN)
 */
function getTileType(tileId) {
    if (tileId >= 0 && tileId <= 8) return 'WAN';
    if (tileId >= 9 && tileId <= 17) return 'TIAO';
    if (tileId >= 18 && tileId <= 26) return 'TONG';
    if (tileId >= 27 && tileId <= 30) return 'FENG';
    if (tileId >= 31 && tileId <= 33) return 'JIAN';
    return 'UNKNOWN';
}

/**
 * 获取牌的中文名称
 * @param {number} tileId - 牌ID
 * @returns {string} 中文名称
 */
function getTileName(tileId) {
    const names = [
        '1万', '2万', '3万', '4万', '5万', '6万', '7万', '8万', '9万',
        '1条', '2条', '3条', '4条', '5条', '6条', '7条', '8条', '9条',
        '1筒', '2筒', '3筒', '4筒', '5筒', '6筒', '7筒', '8筒', '9筒',
        '东风', '南风', '西风', '北风', '红中', '发财', '白板'
    ];
    return names[tileId] || '未知';
}

/**
 * 判断是否为赖子牌
 * @param {number} tileId - 牌ID
 * @param {number} lazyTile - 赖子牌ID
 * @returns {boolean}
 */
function isLazyTile(tileId, lazyTile) {
    return tileId === lazyTile;
}

/**
 * 判断是否为皮子牌
 * @param {number} tileId - 牌ID
 * @param {number[]} skinTiles - 皮子牌ID数组
 * @returns {boolean}
 */
function isSkinTile(tileId, skinTiles) {
    return skinTiles.includes(tileId);
}

/**
 * 判断是否为红中
 * @param {number} tileId - 牌ID
 * @returns {boolean}
 */
function isRedDragon(tileId) {
    return tileId === DRAGON_TILES.RED;
}

/**
 * 计算相对位置（相对于自己）
 * @param {number} targetPlayer - 目标玩家位置
 * @param {number} selfPlayer - 自己的位置
 * @returns {number} 相对位置 (0=自己, 1=下家, 2=对家, 3=上家)
 */
function getRelativePosition(targetPlayer, selfPlayer) {
    return (targetPlayer - selfPlayer + 4) % 4;
}

/**
 * 获取sprite image set key
 * 根据相对位置选择合适的sprite sheet
 * 参考 Mahjong-AI 的 imageSet 逻辑
 *
 * @param {number} relativePos - 相对位置
 * @returns {string} sprite key
 */
function getSpriteKey(relativePos) {
    const spriteMap = {
        0: 'tiles0',  // 自己 - 竖直显示
        1: 'tiles1',  // 下家 - 横向显示
        2: 'tiles2',  // 对家 - 竖直显示（背面）
        3: 'tiles3'   // 上家 - 横向显示
    };
    return spriteMap[relativePos] || 'tiles0';
}

// 导出函数
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        getTileFrameIndex,
        isNumberTile,
        isHonorTile,
        getTileType,
        getTileName,
        isLazyTile,
        isSkinTile,
        isRedDragon,
        getRelativePosition,
        getSpriteKey
    };
}
```

**Step 2: 验证文件创建成功**

运行: `cat src/mahjong_rl/web/phaser_client/js/utils/TileUtils.js`
Expected: 显示完整的工具函数定义

**Step 3: 提交工具函数文件**

```bash
git add src/mahjong_rl/web/phaser_client/js/utils/TileUtils.js
git commit -m "feat(phaser-client): add tile utility functions"
```

---

## 模块2: HTML和CSS基础

### Task 3: 创建主HTML页面

**文件:**
- Create: `src/mahjong_rl/web/phaser_client/index.html`

**Step 1: 创建HTML页面**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>武汉麻将 - Wuhan Mahjong</title>
    <link rel="stylesheet" href="css/styles.css">
</head>
<body>
    <!-- Phaser 游戏容器 -->
    <div id="game-container"></div>

    <!-- 加载提示 -->
    <div id="loading-screen">
        <div class="loading-content">
            <h1>武汉麻将</h1>
            <p>正在加载资源...</p>
            <div class="loading-bar">
                <div class="loading-progress"></div>
            </div>
        </div>
    </div>

    <!-- Phaser.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/phaser@3.60.0/dist/phaser.min.js"></script>

    <!-- 游戏脚本 -->
    <script type="module" src="js/main.js"></script>
</body>
</html>
```

**Step 2: 验证文件创建成功**

运行: `cat src/mahjong_rl/web/phaser_client/index.html`
Expected: 显示完整的HTML结构

**Step 3: 提交HTML文件**

```bash
git add src/mahjong_rl/web/phaser_client/index.html
git commit -m "feat(phaser-client): add main HTML page"
```

---

### Task 4: 创建CSS样式

**文件:**
- Create: `src/mahjong_rl/web/phaser_client/css/styles.css`

**Step 1: 创建CSS文件**

```css
/**
 * 武汉麻将网页客户端 - 全局样式
 */

/* ============ 全局重置 ============ */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body {
    width: 100%;
    height: 100%;
    overflow: hidden;
    background-color: #1a1a1a;
    font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
}

/* ============ 游戏容器 ============ */
#game-container {
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    background: linear-gradient(135deg, #2c3e50 0%, #1a1a1a 100%);
}

/* ============ 加载屏幕 ============ */
#loading-screen {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.9);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
    transition: opacity 0.5s ease;
}

#loading-screen.hidden {
    opacity: 0;
    pointer-events: none;
}

.loading-content {
    text-align: center;
    color: #ffffff;
}

.loading-content h1 {
    font-size: 48px;
    margin-bottom: 20px;
    color: #f0e68c;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
}

.loading-content p {
    font-size: 18px;
    margin-bottom: 30px;
    color: #cccccc;
}

.loading-bar {
    width: 300px;
    height: 6px;
    background-color: #333333;
    border-radius: 3px;
    overflow: hidden;
    margin: 0 auto;
}

.loading-progress {
    width: 0%;
    height: 100%;
    background-color: #f0e68c;
    transition: width 0.3s ease;
    animation: loading 2s ease-in-out infinite;
}

@keyframes loading {
    0% { width: 0%; }
    50% { width: 70%; }
    100% { width: 100%; }
}

/* ============ 隐藏滚动条 ============ */
::-webkit-scrollbar {
    display: none;
}

/* ============ Phaser Canvas 样式 ============ */
canvas {
    display: block;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
}
```

**Step 2: 验证文件创建成功**

运行: `cat src/mahjong_rl/web/phaser_client/css/styles.css`
Expected: 显示完整的CSS样式定义

**Step 3: 提交CSS文件**

```bash
git add src/mahjong_rl/web/phaser_client/css/styles.css
git commit -m "feat(phaser-client): add global CSS styles"
```

---

## 模块3: Phaser场景基础

### Task 5: 创建Phaser配置和主入口

**文件:**
- Create: `src/mahjong_rl/web/phaser_client/js/main.js`

**Step 1: 创建主入口文件**

```javascript
/**
 * 武汉麻将网页客户端 - 主入口
 * 初始化 Phaser 游戏实例
 */

import { DEFAULT_DIMENSION } from './utils/Constants.js';
import MahjongScene from './js/scenes/MahjongScene.js';

// 计算全局缩放比例
function calculateGlobalScale() {
    const windowWidth = window.innerWidth || document.documentElement.clientWidth;
    const windowHeight = window.innerHeight || document.documentElement.clientHeight;
    const minDimension = Math.min(windowWidth, windowHeight);
    return minDimension / DEFAULT_DIMENSION;
}

// 更新全局缩放比例
function updateGlobalScale() {
    window.GLOBAL_SCALE_RATE = calculateGlobalScale();
}

// 窗口大小改变时更新缩放
window.addEventListener('resize', () => {
    updateGlobalScale();
});

// 初始化全局缩放
updateGlobalScale();

// Phaser 配置
const config = {
    type: Phaser.AUTO,
    parent: 'game-container',
    width: Math.min(window.innerWidth, window.innerHeight),
    height: Math.min(window.innerWidth, window.innerHeight),
    backgroundColor: '#000000',
    scale: {
        mode: Phaser.Scale.FIT,
        autoCenter: Phaser.Scale.CENTER_BOTH
    },
    scene: [MahjongScene],
    physics: {
        default: 'arcade',
        arcade: {
            debug: false
        }
    },
    // 禁用右键菜单
    disableContextMenu: true
};

// 创建游戏实例
const game = new Phaser.Game(config);

// 隐藏加载屏幕
window.addEventListener('load', () => {
    setTimeout(() => {
        const loadingScreen = document.getElementById('loading-screen');
        if (loadingScreen) {
            loadingScreen.classList.add('hidden');
            setTimeout(() => {
                loadingScreen.style.display = 'none';
            }, 500);
        }
    }, 1000);
});

// 导出游戏实例（用于调试）
window.game = game;
```

**Step 2: 验证文件创建成功**

运行: `cat src/mahjong_rl/web/phaser_client/js/main.js`
Expected: 显示完整的主入口代码

**Step 3: 提交主入口文件**

```bash
git add src/mahjong_rl/web/phaser_client/js/main.js
git commit -m "feat(phaser-client): add Phaser game entry point"
```

---

### Task 6: 创建MahjongScene基础结构

**文件:**
- Create: `src/mahjong_rl/web/phaser_client/js/scenes/MahjongScene.js`

**Step 1: 创建场景文件**

```javascript
/**
 * 武汉麻将主场景
 * 负责游戏界面的渲染和交互
 */

import { TILE_SIZES, COLORS, LAYOUT, GAME_STATES } from '../utils/Constants.js';
import { getTileFrameIndex, getSpriteKey } from '../utils/TileUtils.js';

export default class MahjongScene extends Phaser.Scene {
    constructor() {
        super({ key: 'MahjongScene' });

        // 游戏状态
        this.gameState = {
            current_state: GAME_STATES.INITIAL,
            current_player_idx: 0,
            dealer_idx: 0,
            lazy_tile: null,
            skin_tiles: [],
            wall_count: 0,
            players: [],
            debug_mode: false
        };

        // 图层组
        this.layers = {};

        // 加载进度
        this.loadProgress = 0;
    }

    /**
     * 预加载资源
     */
    preload() {
        this.createLoadingBar();
        this.loadAssets();
    }

    /**
     * 创建加载进度条
     */
    createLoadingBar() {
        const width = this.cameras.main.width;
        const height = this.cameras.main.height;

        // 监听加载进度
        this.load.on('progress', (progress) => {
            this.loadProgress = progress;
            this.updateLoadingBar();
        });

        this.load.on('complete', () => {
            this.hideLoadingBar();
        });
    }

    /**
     * 更新加载进度条
     */
    updateLoadingBar() {
        const progressBar = document.querySelector('.loading-progress');
        if (progressBar) {
            progressBar.style.width = `${this.loadProgress * 100}%`;
        }
    }

    /**
     * 隐藏加载进度条
     */
    hideLoadingBar() {
        // 由主入口的 window.load 处理
    }

    /**
     * 加载游戏资源
     */
    loadAssets() {
        // TODO: 从 Mahjong-AI 复制麻将牌 sprite sheet
        // 临时使用占位符

        // 加载麻将牌图片（需要在后续任务中添加真实资源）
        for (let i = 0; i <= 4; i++) {
            this.load.image(`tiles${i}`, 'assets/images/placeholder.png');
        }

        // 加载背景图
        this.load.image('background', 'assets/images/background.jpg');

        // 加载音效（可选）
        // this.load.audio('draw', 'assets/audio/draw.wav');
    }

    /**
     * 创建场景
     */
    create() {
        // 创建背景
        this.createBackground();

        // 创建图层组
        this.createLayers();

        // 创建游戏界面元素
        this.createGameBoard();

        // 初始化测试数据（后续替换为真实数据）
        this.initTestData();

        // 渲染初始状态
        this.render();
    }

    /**
     * 创建背景
     */
    createBackground() {
        const bg = this.add.image(
            this.cameras.main.width / 2,
            this.cameras.main.height / 2,
            'background'
        );

        const scale = Math.max(
            this.cameras.main.width / bg.width,
            this.cameras.main.height / bg.height
        );
        bg.setScale(scale).setDepth(0);
    }

    /**
     * 创建图层组
     */
    createLayers() {
        this.layers = {
            background: this.add.group(),    // 背景元素
            board: this.add.group(),          // 游戏桌面
            players: this.add.group(),        // 玩家信息
            tiles: this.add.group(),          // 麻将牌
            ui: this.add.group(),             // UI元素
            effects: this.add.group()         // 特效
        };
    }

    /**
     * 创建游戏桌面
     */
    createGameBoard() {
        const centerX = this.cameras.main.width / 2;
        const centerY = this.cameras.main.height / 2;

        // TODO: 创建中央区域（赖子/皮子指示器等）
        // 后续任务中实现

        // 创建占位框
        const boardSize = 250 * window.GLOBAL_SCALE_RATE;
        const graphics = this.add.graphics();

        graphics.lineStyle(2, 0xFFFFFF, 0.3);
        graphics.strokeRoundedRect(
            centerX - boardSize / 2,
            centerY - boardSize / 2,
            boardSize,
            boardSize,
            15
        );

        this.layers.board.add(graphics);
    }

    /**
     * 初始化测试数据
     */
    initTestData() {
        // 创建测试玩家数据
        this.gameState.players = [
            {
                player_id: 0,
                hand_tiles: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                melds: [],
                discard_tiles: [],
                special_gangs: [0, 0, 0],
                is_dealer: true,
                is_win: false
            },
            {
                player_id: 1,
                hand_tiles: [],
                melds: [],
                discard_tiles: [],
                special_gangs: [0, 0, 0],
                is_dealer: false,
                is_win: false
            },
            {
                player_id: 2,
                hand_tiles: [],
                melds: [],
                discard_tiles: [],
                special_gangs: [0, 0, 0],
                is_dealer: false,
                is_win: false
            },
            {
                player_id: 3,
                hand_tiles: [],
                melds: [],
                discard_tiles: [],
                special_gangs: [0, 0, 0],
                is_dealer: false,
                is_win: false
            }
        ];

        // 设置测试特殊牌
        this.gameState.lazy_tile = 5;  // 假设赖子是6万
        this.gameState.skin_tiles = [4, 6];  // 假设皮子是5万和7万
        this.gameState.wall_count = 67;
    }

    /**
     * 渲染游戏界面
     */
    render() {
        // 清空所有图层
        Object.values(this.layers).forEach(layer => {
            layer.clear(true, true);
        });

        // 重新创建图层
        this.createLayers();

        // 渲染各玩家
        for (let i = 0; i < 4; i++) {
            this.renderPlayer(i);
        }

        // 渲染中央区域
        this.renderCenterArea();
    }

    /**
     * 渲染单个玩家
     */
    renderPlayer(playerId) {
        // TODO: 后续任务中实现
        console.log(`Rendering player ${playerId}`);
    }

    /**
     * 渲染中央区域
     */
    renderCenterArea() {
        // TODO: 后续任务中实现
        console.log('Rendering center area');
    }

    /**
     * 更新游戏状态
     */
    updateState(newState) {
        this.gameState = { ...this.gameState, ...newState };
        this.render();
    }
}
```

**Step 2: 验证文件创建成功**

运行: `cat src/mahjong_rl/web/phaser_client/js/scenes/MahjongScene.js`
Expected: 显示完整的场景类定义

**Step 3: 提交场景文件**

```bash
git add src/mahjong_rl/web/phaser_client/js/scenes/MahjongScene.js
git commit -m "feat(phaser-client): add basic MahjongScene structure"
```

---

## 测试点1: 验证基础结构

### Task 7: 测试Phaser基础渲染

**文件:**
- Test: 手动浏览器测试

**Step 1: 启动本地HTTP服务器**

```bash
# 进入项目目录
cd src/mahjong_rl/web/phaser_client

# 启动Python HTTP服务器
python -m http.server 8080
```

**Step 2: 打开浏览器访问**

在浏览器中打开: `http://localhost:8080/index.html`

**Expected:**
- 显示黑色背景
- 显示"武汉麻将"标题和加载提示
- 加载进度条动画
- 最终显示游戏界面（虽然内容为空，但Phaser canvas已创建）

**Step 3: 检查浏览器控制台**

按F12打开开发者工具，查看Console标签

**Expected:**
- 无JavaScript错误
- 显示 Phaser 版本信息
- 显示 "Rendering player 0-3" 和 "Rendering center area" 日志

**Step 4: 提交测试结果**

如果测试通过，创建测试说明文档：

```bash
cat > TEST_PHASE1.md << 'EOF'
# 阶段1测试结果

## 测试日期
2026-01-27

## 测试环境
- 浏览器: Chrome/Edge/Firefox
- Phaser版本: 3.60.0

## 测试结果
- ✅ 页面正常加载
- ✅ Phaser游戏实例创建成功
- ✅ 场景初始化成功
- ✅ 无JavaScript错误

## 已知问题
- 麻将牌图片使用占位符（待添加）
- 游戏界面内容待实现
EOF

git add TEST_PHASE1.md
git commit -m "test(phaser-client): document Phase 1 test results"
```

---

## 后续阶段预览

完成阶段1后，你将拥有：
- ✅ 完整的项目目录结构
- ✅ 游戏常量和工具函数
- ✅ Phaser场景基础框架
- ✅ 加载屏幕和基础UI

**下一步（阶段1剩余部分）：**
1. 从 Mahjong-AI 复制麻将牌 sprite sheet
2. 实现手牌渲染（正面和背面）
3. 实现弃牌河渲染
4. 实现副露区渲染
5. 实现特殊牌视觉效果（赖子/皮/红中）
6. 实现明牌模式切换

**后续阶段概览：**
- **阶段2**: 集成 WuhanMahjongEnv，实现本地游戏逻辑
- **阶段3**: 实现回放系统
- **阶段4**: 实现 FastAPI WebSocket 局域网对战

---

## 重要注意事项

1. **资源复用**: 从 Mahjong-AI 项目复制麻将牌图片和音频资源到 `assets/` 目录
2. **浏览器兼容性**: 使用现代浏览器（Chrome 90+, Edge 90+, Firefox 88+, Safari 14+）
3. **调试**: 使用浏览器开发者工具进行调试
4. **性能**: 确保在大屏幕和小屏幕上都能正常显示

## 参考文档

- Mahjong-AI 项目: `D:\DATA\Python_Project\Code\PettingZooRLENVMahjong\Mahjong-AI\online_game\web_client`
- Phaser.js 文档: https://photonstorm.github.io/phaser3-docs/
- 武汉麻将规则: `src/mahjong_rl/rules/wuhan_mahjong_rule_engine/wuhan_mahjong_rules.md`
