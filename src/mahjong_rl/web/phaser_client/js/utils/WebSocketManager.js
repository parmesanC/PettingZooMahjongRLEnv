/**
 * WebSocket通信管理器
 * 处理与后端的WebSocket连接和消息
 */

export class WebSocketManager {
    constructor(url, onMessageCallback) {
        this.url = url;
        this.onMessageCallback = onMessageCallback;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }

    /**
     * 连接WebSocket
     */
    connect(playerId = 0) {
        const wsUrl = `${this.url}/${playerId}`;
        console.log(`正在连接WebSocket: ${wsUrl}`);

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('✓ WebSocket连接成功');
            this.reconnectAttempts = 0;
        };

        this.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                console.log('收到消息:', message.type);

                if (this.onMessageCallback) {
                    this.onMessageCallback(message);
                }
            } catch (e) {
                console.error('解析消息失败:', e, event.data);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket错误:', error);
        };

        this.ws.onclose = () => {
            console.log('WebSocket连接关闭');
            this.attemptReconnect(playerId);
        };
    }

    /**
     * 尝试重连
     */
    attemptReconnect(playerId) {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * this.reconnectAttempts;

            console.log(`${delay}ms后尝试重连 (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

            setTimeout(() => {
                this.connect(playerId);
            }, delay);
        } else {
            console.error('达到最大重连次数，放弃重连');
        }
    }

    /**
     * 发送动作
     */
    sendAction(actionType, parameter = 0) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const message = {
                type: 'action',
                action_type: actionType,
                parameter: parameter
            };

            this.ws.send(JSON.stringify(message));
            console.log('发送动作:', message);
        } else {
            console.error('WebSocket未连接，无法发送动作');
        }
    }

    /**
     * 请求当前状态
     */
    requestState() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const message = { type: 'get_state' };
            this.ws.send(JSON.stringify(message));
        }
    }

    /**
     * 断开连接
     */
    disconnect() {
        if (this.ws) {
            this.reconnectAttempts = this.maxReconnectAttempts; // 防止重连
            this.ws.close();
        }
    }
}
