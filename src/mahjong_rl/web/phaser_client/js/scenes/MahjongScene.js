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
