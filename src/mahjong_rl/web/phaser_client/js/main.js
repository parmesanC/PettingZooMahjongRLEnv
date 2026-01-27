/**
 * 武汉麻将网页客户端 - 主入口
 * 初始化 Phaser 游戏实例
 */

import { DEFAULT_DIMENSION } from './utils/Constants.js';
import MahjongScene from './scenes/MahjongScene.js';

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
