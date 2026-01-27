/**
 * 武汉麻将主场景
 * 负责游戏界面的渲染和交互
 */

import { TILE_SIZES, COLORS, GAME_STATES } from '../utils/Constants.js';
import { getTileFrameIndex, getRelativePosition, isLazyTile, isSkinTile, isRedDragon } from '../utils/TileUtils.js';
import { WebSocketManager } from '../utils/WebSocketManager.js';

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

        // 自己的玩家位置（假设自己是玩家0）
        this.selfPlayerIdx = 0;

        // 图层组
        this.layers = {};

        // 玩家手牌组
        this.handGroups = [null, null, null, null];

        // 弃牌河组
        this.riverGroups = [null, null, null, null];

        // 副露区组
        this.meldGroups = [null, null, null, null];

        // 加载进度
        this.loadProgress = 0;

        // WebSocket管理器
        this.wsManager = null;
        this.playerId = 0;  // 默认为玩家0
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
        this.load.on('progress', (progress) => {
            this.loadProgress = progress;
            this.updateLoadingBar();
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
        // 加载麻将牌 sprite sheets
        this.load.spritesheet('tiles0', 'assets/images/tiles0.png', {
            frameWidth: 72,
            frameHeight: 109
        });
        this.load.spritesheet('tiles1', 'assets/images/tiles1.png', {
            frameWidth: 97,
            frameHeight: 89
        });
        this.load.spritesheet('tiles2', 'assets/images/tiles2.png', {
            frameWidth: 72,
            frameHeight: 109
        });
        this.load.spritesheet('tiles3', 'assets/images/tiles3.png', {
            frameWidth: 97,
            frameHeight: 89
        });
        this.load.spritesheet('tiles4', 'assets/images/tiles4.png', {
            frameWidth: 117,
            frameHeight: 177
        });

        // 加载手牌背面图
        this.load.image('right_tile', 'assets/images/right_tile.png');
        this.load.image('left_tile', 'assets/images/left_tile.png');

        // 加载背景图
        this.load.image('background', 'assets/images/background.png');
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

        // 初始化WebSocket连接
        this.initWebSocket();

        // 创建动作按钮
        this.createActionButtons();

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
            background: this.add.group(),
            board: this.add.group(),
            players: this.add.group(),
            tiles: this.add.group(),
            ui: this.add.group(),
            effects: this.add.group()
        };

        // 创建玩家手牌组
        for (let i = 0; i < 4; i++) {
            this.handGroups[i] = this.add.group();
            this.riverGroups[i] = this.add.group();
            this.meldGroups[i] = this.add.group();
        }
    }

    /**
     * 创建游戏桌面
     */
    createGameBoard() {
        const centerX = this.cameras.main.width / 2;
        const centerY = this.cameras.main.height / 2;
        const scale = window.GLOBAL_SCALE_RATE;

        // 创建中央信息区
        const boardSize = 250 * scale;
        const graphics = this.add.graphics();

        graphics.lineStyle(2, 0xFFFFFF, 0.3);
        graphics.strokeRoundedRect(
            centerX - boardSize / 2,
            centerY - boardSize / 2,
            boardSize,
            boardSize,
            15 * scale
        );

        this.layers.board.add(graphics);
    }

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

    /**
     * 初始化测试数据
     */
    initTestData() {
        // 创建更丰富的测试数据
        this.gameState.players = [
            {
                player_id: 0,
                hand_tiles: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                // 测试副露：碰（3张相同）、吃（顺子）、杠（4张相同）
                melds: [
                    { action_type: 2, tiles: [0, 0, 0], from_player: 1 },  // 碰：1万 x3
                    { action_type: 1, tiles: [9, 10, 11], from_player: 2 }  // 吃：1条2条3条
                ],
                discard_tiles: [13, 14, 15],
                special_gangs: [0, 0, 0],
                is_dealer: true,
                is_win: false
            },
            {
                player_id: 1,
                hand_tiles: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  // 10张背面牌（副露后少了）
                melds: [
                    { action_type: 3, tiles: [5, 5, 5, 5], from_player: 0 }  // 明杠：6万 x4
                ],
                discard_tiles: [16, 17],
                special_gangs: [0, 0, 0],
                is_dealer: false,
                is_win: false
            },
            {
                player_id: 2,
                hand_tiles: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  // 10张背面牌
                melds: [
                    { action_type: 2, tiles: [1, 1, 1], from_player: 3 },  // 碰：2万 x3
                    { action_type: 1, tiles: [18, 19, 20], from_player: 0 }  // 吃：1筒2筒3筒
                ],
                discard_tiles: [18, 19, 20],
                special_gangs: [0, 0, 0],
                is_dealer: false,
                is_win: false
            },
            {
                player_id: 3,
                hand_tiles: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  // 10张背面牌
                melds: [],
                discard_tiles: [21, 22],
                special_gangs: [0, 0, 0],
                is_dealer: false,
                is_win: false
            }
        ];

        // 设置测试特殊牌
        this.gameState.lazy_tile = 5;  // 6万是赖子
        this.gameState.skin_tiles = [4, 6];  // 5万和7万是皮子
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

        // 清空玩家组
        for (let i = 0; i < 4; i++) {
            if (this.handGroups[i]) this.handGroups[i].clear(true, true);
            if (this.riverGroups[i]) this.riverGroups[i].clear(true, true);
            if (this.meldGroups[i]) this.meldGroups[i].clear(true, true);
        }

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
        const player = this.gameState.players[playerId];
        if (!player) return;

        const relativePos = getRelativePosition(playerId, this.selfPlayerIdx);
        const scale = window.GLOBAL_SCALE_RATE;

        // 渲染手牌
        this.renderHandTiles(playerId, player.hand_tiles, relativePos);

        // 渲染弃牌河
        this.renderRiver(playerId, player.discard_tiles, relativePos);

        // 渲染副露区
        this.renderMelds(playerId, player.melds, relativePos);
    }

    /**
     * 渲染手牌
     */
    renderHandTiles(playerId, handTiles, relativePos) {
        const scale = window.GLOBAL_SCALE_RATE;
        const canvasWidth = this.cameras.main.width;
        const canvasHeight = this.cameras.main.height;

        switch (relativePos) {
            case 0:  // 自己 - 底部，显示正面
                this.renderSelfHand(handTiles, scale);
                break;
            case 1:  // 下家 - 右侧，显示背面
                this.renderRightHand(handTiles.length, scale);
                break;
            case 2:  // 对家 - 顶部，显示背面
                this.renderOppoHand(handTiles.length, scale);
                break;
            case 3:  // 上家 - 左侧，显示背面
                this.renderLeftHand(handTiles.length, scale);
                break;
        }
    }

    /**
     * 渲染自己的手牌（正面）
     */
    renderSelfHand(handTiles, scale) {
        let x = 40 * scale;
        const y = this.cameras.main.height - 60 * scale;
        const tileScale = scale * 0.45;

        // 先排序手牌
        const sortedTiles = [...handTiles].sort((a, b) => a - b);

        for (let i = 0; i < sortedTiles.length; i++) {
            x += (117 + 2) * tileScale;
            const tileId = sortedTiles[i];
            const frameIndex = getTileFrameIndex(tileId);

            const tile = this.add.image(x, y, 'tiles4', frameIndex)
                .setScale(tileScale)
                .setDepth(1000);

            // 添加特殊牌效果
            this.addSpecialTileEffects(tile, tileId, x, y, tileScale);

            // 添加交互（带打牌动画）
            this.setHandTileInteractivity(tile, i, tileId, sortedTiles);

            this.handGroups[0].add(tile);
        }
    }

    /**
     * 渲染右侧手牌（背面）
     */
    renderRightHand(length, scale) {
        const tileScale = scale * 0.5;
        const bottom = this.cameras.main.height - 450 * tileScale;
        const imageWidth = 57 * tileScale;
        const imageHeight = 132 * tileScale;

        let x = this.cameras.main.width - imageWidth;
        let y = bottom - (imageHeight - 60 * tileScale) * length;

        for (let i = 0; i < length; i++) {
            y += (imageHeight - 60 * tileScale);
            const tile = this.add.image(x, y, 'right_tile')
                .setScale(tileScale)
                .setDepth(i + 1);
            this.handGroups[1].add(tile);
        }
    }

    /**
     * 渲染对家手牌（背面）
     */
    renderOppoHand(length, scale) {
        const tileScale = scale * 0.5;
        const imageWidth = 72 * tileScale;

        let x = this.cameras.main.width - 350 * tileScale;
        const y = 50 * tileScale;

        for (let i = 0; i < length; i++) {
            x -= imageWidth;
            const tile = this.add.image(x, y, 'tiles2', 30)
                .setScale(tileScale)
                .setDepth(10);
            this.handGroups[2].add(tile);
        }
    }

    /**
     * 渲染左侧手牌（背面）
     */
    renderLeftHand(length, scale) {
        const tileScale = scale * 0.5;
        const imageWidth = 57 * tileScale;
        const imageHeight = 132 * tileScale;

        const x = imageWidth;
        let y = 350 * tileScale;

        for (let i = 0; i < length; i++) {
            const tile = this.add.image(x, y, 'left_tile')
                .setScale(tileScale)
                .setDepth(i + 1);
            this.handGroups[3].add(tile);
            y += (imageHeight - 60 * tileScale);
        }
    }

    /**
     * 渲染弃牌河
     */
    renderRiver(playerId, discardTiles, relativePos) {
        if (!discardTiles || discardTiles.length === 0) return;

        const scale = window.GLOBAL_SCALE_RATE;
        const canvasWidth = this.cameras.main.width;
        const canvasHeight = this.cameras.main.height;

        switch (relativePos) {
            case 0:  // 自己的弃牌河 - 右下区域
                this.renderSelfRiver(discardTiles, scale);
                break;
            case 1:  // 下家的弃牌河 - 右下区域
                this.renderRightRiver(discardTiles, scale);
                break;
            case 2:  // 对家的弃牌河 - 左上区域
                this.renderOppoRiver(discardTiles, scale);
                break;
            case 3:  // 上家的弃牌河 - 左上区域
                this.renderLeftRiver(discardTiles, scale);
                break;
        }
    }

    /**
     * 渲染自己的弃牌河
     */
    renderSelfRiver(discardTiles, scale) {
        const tileScale = scale * 0.35;
        const startX = this.cameras.main.width - 400 * scale;
        const startY = this.cameras.main.height - 180 * scale;
        const tileWidth = 72 * tileScale;
        const tileHeight = 109 * tileScale;
        const gap = 2 * scale;

        // 分成多行显示（每行6张）
        const rowSize = 6;
        for (let i = 0; i < discardTiles.length; i++) {
            const row = Math.floor(i / rowSize);
            const col = i % rowSize;

            const x = startX + col * (tileWidth + gap);
            const y = startY - row * (tileHeight + gap);

            const tileId = discardTiles[i];
            const frameIndex = getTileFrameIndex(tileId);

            const tile = this.add.image(x, y, 'tiles0', frameIndex)
                .setScale(tileScale)
                .setDepth(500);

            this.addSpecialTileEffects(tile, tileId, x, y, tileScale);
            this.riverGroups[0].add(tile);
        }
    }

    /**
     * 渲染右侧弃牌河
     */
    renderRightRiver(discardTiles, scale) {
        const tileScale = scale * 0.35;
        const startX = this.cameras.main.width - 80 * scale;
        const startY = this.cameras.main.height - 400 * scale;
        const tileWidth = 97 * tileScale;
        const tileHeight = 89 * tileScale;
        const gap = 2 * scale;

        const rowSize = 6;
        for (let i = 0; i < discardTiles.length; i++) {
            const row = Math.floor(i / rowSize);
            const col = i % rowSize;

            const x = startX - row * (tileHeight + gap);
            const y = startY + col * (tileWidth + gap);

            const tileId = discardTiles[i];
            const frameIndex = getTileFrameIndex(tileId);

            const tile = this.add.image(x, y, 'tiles1', frameIndex)
                .setScale(tileScale)
                .setDepth(500)
                .setRotation(Math.PI / 2);

            this.addSpecialTileEffects(tile, tileId, x, y, tileScale);
            this.riverGroups[1].add(tile);
        }
    }

    /**
     * 渲染对家弃牌河
     */
    renderOppoRiver(discardTiles, scale) {
        const tileScale = scale * 0.35;
        const startX = 400 * scale;
        const startY = 180 * scale;
        const tileWidth = 72 * tileScale;
        const tileHeight = 109 * tileScale;
        const gap = 2 * scale;

        const rowSize = 6;
        for (let i = 0; i < discardTiles.length; i++) {
            const row = Math.floor(i / rowSize);
            const col = (rowSize - 1) - (i % rowSize);

            const x = startX - col * (tileWidth + gap);
            const y = startY + row * (tileHeight + gap);

            const tileId = discardTiles[i];
            const frameIndex = getTileFrameIndex(tileId);

            const tile = this.add.image(x, y, 'tiles0', frameIndex)
                .setScale(tileScale)
                .setDepth(500);

            this.addSpecialTileEffects(tile, tileId, x, y, tileScale);
            this.riverGroups[2].add(tile);
        }
    }

    /**
     * 渲染左侧弃牌河
     */
    renderLeftRiver(discardTiles, scale) {
        const tileScale = scale * 0.35;
        const startX = 80 * scale;
        const startY = 400 * scale;
        const tileWidth = 97 * tileScale;
        const tileHeight = 89 * tileScale;
        const gap = 2 * scale;

        const rowSize = 6;
        for (let i = 0; i < discardTiles.length; i++) {
            const row = Math.floor(i / rowSize);
            const col = (rowSize - 1) - (i % rowSize);

            const x = startX + row * (tileHeight + gap);
            const y = startY - col * (tileWidth + gap);

            const tileId = discardTiles[i];
            const frameIndex = getTileFrameIndex(tileId);

            const tile = this.add.image(x, y, 'tiles1', frameIndex)
                .setScale(tileScale)
                .setDepth(500)
                .setRotation(Math.PI / 2);

            this.addSpecialTileEffects(tile, tileId, x, y, tileScale);
            this.riverGroups[3].add(tile);
        }
    }

    /**
     * 渲染副露区
     */
    renderMelds(playerId, melds, relativePos) {
        if (!melds || melds.length === 0) return;

        const scale = window.GLOBAL_SCALE_RATE;

        switch (relativePos) {
            case 0:  // 自己的副露区 - 左侧
                this.renderSelfMelds(melds, scale);
                break;
            case 1:  // 下家的副露区 - 下侧
                this.renderRightMelds(melds, scale);
                break;
            case 2:  // 对家的副露区 - 右侧
                this.renderOppoMelds(melds, scale);
                break;
            case 3:  // 上家的副露区 - 上侧
                this.renderLeftMelds(melds, scale);
                break;
        }
    }

    /**
     * 渲染自己的副露区
     */
    renderSelfMelds(melds, scale) {
        const tileScale = scale * 0.4;
        const startX = 100 * scale;
        const startY = this.cameras.main.height - 140 * scale;
        const tileWidth = 117 * tileScale;
        const tileHeight = 177 * tileScale;
        const meldGap = 20 * scale;
        const tileGap = 5 * scale;

        for (let m = 0; m < melds.length; m++) {
            const meld = melds[m];
            const meldX = startX + m * (tileWidth * 3 + tileGap * 2 + meldGap);

            for (let i = 0; i < meld.tiles.length; i++) {
                const tileId = meld.tiles[i];
                const frameIndex = getTileFrameIndex(tileId);
                const tileX = meldX + i * (tileWidth + tileGap);

                const tile = this.add.image(tileX, startY, 'tiles4', frameIndex)
                    .setScale(tileScale)
                    .setDepth(800);

                this.addSpecialTileEffects(tile, tileId, tileX, startY, tileScale);
                this.meldGroups[0].add(tile);
            }
        }
    }

    /**
     * 渲染右侧副露区
     */
    renderRightMelds(melds, scale) {
        const tileScale = scale * 0.35;
        const startX = this.cameras.main.width - 60 * scale;
        const startY = this.cameras.main.height - 350 * scale;
        const tileWidth = 97 * tileScale;
        const tileHeight = 89 * tileScale;
        const meldGap = 10 * scale;
        const tileGap = 2 * scale;

        for (let m = 0; m < melds.length; m++) {
            const meld = melds[m];
            const meldY = startY + m * (tileHeight * 3 + tileGap * 2 + meldGap);

            for (let i = 0; i < meld.tiles.length; i++) {
                const tileId = meld.tiles[i];
                const frameIndex = getTileFrameIndex(tileId);
                const tileY = meldY + i * (tileHeight + tileGap);

                const tile = this.add.image(startX, tileY, 'tiles1', frameIndex)
                    .setScale(tileScale)
                    .setDepth(800)
                    .setRotation(Math.PI / 2);

                this.addSpecialTileEffects(tile, tileId, startX, tileY, tileScale);
                this.meldGroups[1].add(tile);
            }
        }
    }

    /**
     * 渲染对家副露区
     */
    renderOppoMelds(melds, scale) {
        const tileScale = scale * 0.35;
        const startX = this.cameras.main.width - 350 * scale;
        const startY = 130 * scale;
        const tileWidth = 72 * tileScale;
        const tileHeight = 109 * tileScale;
        const meldGap = 15 * scale;
        const tileGap = 3 * scale;

        for (let m = melds.length - 1; m >= 0; m--) {
            const meld = melds[m];
            const meldX = startX - (melds.length - 1 - m) * (tileWidth * 3 + tileGap * 2 + meldGap);

            for (let i = 0; i < meld.tiles.length; i++) {
                const tileId = meld.tiles[i];
                const frameIndex = getTileFrameIndex(tileId);
                const tileX = meldX - i * (tileWidth + tileGap);

                const tile = this.add.image(tileX, startY, 'tiles0', frameIndex)
                    .setScale(tileScale)
                    .setDepth(800);

                this.addSpecialTileEffects(tile, tileId, tileX, startY, tileScale);
                this.meldGroups[2].add(tile);
            }
        }
    }

    /**
     * 渲染左侧副露区
     */
    renderLeftMelds(melds, scale) {
        const tileScale = scale * 0.35;
        const startX = 60 * scale;
        const startY = 350 * scale;
        const tileWidth = 97 * tileScale;
        const tileHeight = 89 * tileScale;
        const meldGap = 10 * scale;
        const tileGap = 2 * scale;

        for (let m = melds.length - 1; m >= 0; m--) {
            const meld = melds[m];
            const meldY = startY + (melds.length - 1 - m) * (tileHeight * 3 + tileGap * 2 + meldGap);

            for (let i = 0; i < meld.tiles.length; i++) {
                const tileId = meld.tiles[i];
                const frameIndex = getTileFrameIndex(tileId);
                const tileY = meldY + i * (tileHeight + tileGap);

                const tile = this.add.image(startX, tileY, 'tiles1', frameIndex)
                    .setScale(tileScale)
                    .setDepth(800)
                    .setRotation(Math.PI / 2);

                this.addSpecialTileEffects(tile, tileId, startX, tileY, tileScale);
                this.meldGroups[3].add(tile);
            }
        }
    }

    /**
     * 添加特殊牌视觉效果
     */
    addSpecialTileEffects(tile, tileId, x, y, scale) {
        const { lazy_tile, skin_tiles } = this.gameState;

        // 赖子效果 - 金色边框
        if (lazy_tile !== null && isLazyTile(tileId, lazy_tile)) {
            const graphics = this.add.graphics();
            graphics.lineStyle(3, COLORS.LAZY_BORDER, 1);
            const width = 117 * scale;
            const height = 177 * scale;
            graphics.strokeRect(x - width / 2, y - height / 2, width, height);
            graphics.setDepth(1001);
            this.layers.effects.add(graphics);
        }

        // 皮子效果 - 银色边框
        if (skin_tiles.length > 0 && isSkinTile(tileId, skin_tiles)) {
            const graphics = this.add.graphics();
            graphics.lineStyle(2, COLORS.SKIN_BORDER, 1);
            const width = 117 * scale;
            const height = 177 * scale;
            graphics.strokeRect(x - width / 2 - 2, y - height / 2 - 2, width + 4, height + 4);
            graphics.setDepth(1001);
            this.layers.effects.add(graphics);
        }

        // 红中效果 - 红色光晕
        if (isRedDragon(tileId)) {
            tile.setTint(0xFFCCCC);
        }
    }

    /**
     * 设置手牌交互
     */
    setHandTileInteractivity(tile, index, tileId, sortedTiles) {
        tile.setInteractive();

        tile.on('pointerover', () => {
            tile.y -= 15 * window.GLOBAL_SCALE_RATE;
        });

        tile.on('pointerout', () => {
            tile.y += 15 * window.GLOBAL_SCALE_RATE;
        });

        tile.on('pointerdown', () => {
            console.log(`Tile ${index} (${tileId}) clicked`);
            this.playDiscardAnimation(tile, tileId, index, sortedTiles);
        });
    }

    /**
     * 播放打牌动画
     */
    playDiscardAnimation(tile, tileId, index, sortedTiles) {
        const scale = window.GLOBAL_SCALE_RATE;
        const canvasWidth = this.cameras.main.width;
        const canvasHeight = this.cameras.main.height;

        // 计算弃牌河的目标位置
        const player = this.gameState.players[0];
        const newDiscardIndex = player.discard_tiles.length;
        const tileScale = scale * 0.35;
        const startX = canvasWidth - 400 * scale;
        const startY = canvasHeight - 180 * scale;
        const tileWidth = 72 * tileScale;
        const tileHeight = 109 * tileScale;
        const gap = 2 * scale;
        const rowSize = 6;

        const row = Math.floor(newDiscardIndex / rowSize);
        const col = newDiscardIndex % rowSize;

        const targetX = startX + col * (tileWidth + gap);
        const targetY = startY - row * (tileHeight + gap);

        // 播放动画
        this.tweens.add({
            targets: tile,
            x: targetX,
            y: targetY,
            scaleX: tileScale,
            scaleY: tileScale,
            duration: 300,
            ease: 'Power2',
            onComplete: () => {
                // 动画完成后，更新游戏状态
                this.updateAfterDiscard(tileId, index, sortedTiles);
            }
        });
    }

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

    /**
     * 渲染中央区域
     */
    renderCenterArea() {
        const centerX = this.cameras.main.width / 2;
        const centerY = this.cameras.main.height / 2;
        const scale = window.GLOBAL_SCALE_RATE;

        // 显示赖子指示器
        if (this.gameState.lazy_tile !== null) {
            this.renderLazyIndicator(centerX, centerY, scale);
        }

        // 显示皮子指示器
        if (this.gameState.skin_tiles.length > 0) {
            this.renderSkinIndicator(centerX, centerY, scale);
        }

        // 显示剩余牌数
        this.renderWallCount(centerX, centerY, scale);

        // 显示当前玩家指示器
        this.renderCurrentPlayerIndicator(centerX, centerY, scale);

        // 显示游戏状态
        this.renderGameState(centerX, centerY, scale);
    }

    /**
     * 渲染当前玩家指示器
     */
    renderCurrentPlayerIndicator(centerX, centerY, scale) {
        const currentPlayer = this.gameState.current_player_idx;
        const relativePos = getRelativePosition(currentPlayer, this.selfPlayerIdx);

        // 根据相对位置确定指示器位置
        const positions = [
            { x: centerX, y: centerY + 120 * scale },      // 自己 - 下方
            { x: centerX + 120 * scale, y: centerY },      // 下家 - 右方
            { x: centerX, y: centerY - 120 * scale },      // 对家 - 上方
            { x: centerX - 120 * scale, y: centerY }       // 上家 - 左方
        ];

        const pos = positions[relativePos];

        // 绘制指示器圆圈
        const graphics = this.add.graphics();
        graphics.fillStyle(0x00FF00, 0.8);
        graphics.fillCircle(pos.x, pos.y, 15 * scale);
        graphics.setDepth(700);

        this.layers.ui.add(graphics);

        // 添加闪烁动画
        this.tweens.add({
            targets: graphics,
            alpha: 0.3,
            duration: 500,
            yoyo: true,
            repeat: -1
        });
    }

    /**
     * 渲染游戏状态
     */
    renderGameState(centerX, centerY, scale) {
        const state = this.gameState.current_state;

        const style = {
            fontFamily: 'Microsoft YaHei',
            fontSize: 18 * scale + 'px',
            color: '#00FF00',
            fontStyle: 'bold',
            backgroundColor: '#000000',
            padding: { x: 10 * scale, y: 5 * scale }
        };

        // 状态名称映射
        const stateNames = {
            'INITIAL': '初始化',
            'DRAWING': '摸牌中',
            'PLAYER_DECISION': '请出牌',
            'DISCARDING': '出牌中',
            'WAITING_RESPONSE': '等待响应',
            'GONG': '杠牌',
            'WIN': '和牌！',
            'FLOW_DRAW': '流局'
        };

        const stateText = stateNames[state] || state;

        const text = this.add.text(
            centerX,
            centerY - 80 * scale,
            stateText,
            style
        ).setOrigin(0.5);

        this.layers.ui.add(text);
    }

    /**
     * 渲染赖子指示器
     */
    renderLazyIndicator(centerX, centerY, scale) {
        const lazyTileId = this.gameState.lazy_tile;
        const frameIndex = getTileFrameIndex(lazyTileId);

        // 在中央显示赖子牌
        const tile = this.add.image(
            centerX - 80 * scale,
            centerY,
            'tiles0',
            frameIndex
        ).setScale(scale * 0.4).setDepth(600);

        // 添加赖子标签
        const style = {
            fontFamily: 'Microsoft YaHei',
            fontSize: 16 * scale + 'px',
            color: '#FFD700',
            fontStyle: 'bold'
        };

        const label = this.add.text(
            centerX - 80 * scale,
            centerY + 50 * scale,
            '赖子',
            style
        ).setOrigin(0.5);

        this.layers.ui.add(tile);
        this.layers.ui.add(label);
    }

    /**
     * 渲染皮子指示器
     */
    renderSkinIndicator(centerX, centerY, scale) {
        const skinTiles = this.gameState.skin_tiles;

        // 显示皮子牌
        for (let i = 0; i < skinTiles.length; i++) {
            const frameIndex = getTileFrameIndex(skinTiles[i]);
            const tile = this.add.image(
                centerX + 80 * scale,
                centerY - 20 * scale + i * 40 * scale,
                'tiles0',
                frameIndex
            ).setScale(scale * 0.3).setDepth(600);

            this.layers.ui.add(tile);
        }

        // 添加皮子标签
        const style = {
            fontFamily: 'Microsoft YaHei',
            fontSize: 14 * scale + 'px',
            color: '#C0C0C0',
            fontStyle: 'bold'
        };

        const label = this.add.text(
            centerX + 80 * scale,
            centerY + 50 * scale,
            '皮子',
            style
        ).setOrigin(0.5);

        this.layers.ui.add(label);
    }

    /**
     * 渲染剩余牌数
     */
    renderWallCount(centerX, centerY, scale) {
        const style = {
            fontFamily: 'Arial',
            fontSize: 20 * scale + 'px',
            color: '#ffffff'
        };

        const text = this.add.text(
            centerX,
            centerY + 80 * scale,
            `剩余: ${this.gameState.wall_count}`,
            style
        ).setOrigin(0.5);

        this.layers.ui.add(text);
    }

    /**
     * 初始化WebSocket连接
     */
    initWebSocket() {
        // 使用localhost以便本地测试
        const wsUrl = `ws://localhost:8011/ws`;

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
                // 显示动作按钮
                this.showActionButtons(message.action_mask);
                break;

            case 'game_over':
                this.showGameOverScreen(message.winner_ids || []);
                break;

            default:
                console.log('未知消息类型:', message.type);
        }
    }

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
}
