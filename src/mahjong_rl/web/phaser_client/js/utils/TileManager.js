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
