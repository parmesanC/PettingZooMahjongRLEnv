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

// 导出函数 (ES6 modules for browser)
export {
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
