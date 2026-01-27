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

// 导出常量 (ES6 modules for browser)
export {
    TILE_TYPES,
    TILE_RANGES,
    WIND_TILES,
    DRAGON_TILES,
    ACTION_TYPES,
    GAME_STATES,
    DEFAULT_DIMENSION,
    TILE_SIZES,
    COLORS,
    LAYOUT
};
