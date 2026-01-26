# 武汉麻将网页客户端设计文档

**创建日期**: 2026-01-27
**设计目标**: 基于 Phaser.js 实现武汉麻将网页客户端，支持本地对战、局域网对战、观战和回放功能

---

## 一、系统架构概览

### 1.1 整体架构

采用三层设计：

1. **前端层**
   - Phaser.js 游戏引擎渲染麻将界面
   - 复用 Mahjong-AI 的麻将牌 sprite sheet 和布局
   - 实现武汉麻将特色元素：赖子/皮/红中的视觉标识

2. **后端层**
   - FastAPI 提供 WebSocket 实时通信
   - 集成现有的 WuhanMahjongEnv 环境
   - 房间管理器：处理玩家/AI配置、观战者管理
   - AI策略接口：支持加载训练模型

3. **数据流**
   - 状态机事件 → WebSocket → 前端渲染
   - 玩家操作 → WebSocket → 状态机更新
   - 回放系统：记录所有事件序列，支持时序回放

### 1.2 MVP 功能定位

**MVP 阶段功能：**
- 单人模式：1人类玩家 + 3简单AI
- 本地热座模式：4人类玩家轮流操作
- 明牌模式：查看所有隐藏信息（牌墙、对手手牌等）
- 基础游戏界面渲染
- 武汉麻将特殊牌显示

**迭代阶段功能：**
- 局域网对战（单房间，无匹配系统）
- AI 策略选择（随机AI/训练模型）
- 观战模式
- 回放系统（暂停/快进/慢放）
- 动态房间设置

---

## 二、界面布局设计

### 2.1 四方位布局

```
┌─────────────────────────────────────────────┐
│        [对家手牌-背面]                       │
│        [对家副露区] [对家特殊杠]             │
│        [对家弃牌河]                          │ ← 北方玩家
│        [玩家信息、分数]                      │
├──────────────┬──────────────┬───────────────┤
│              │              │               │
│  西家手牌-背面│  赖子指示器   │  东家手牌-背面│
│  西家副露区   │  皮指示器     │  东家副露区   │
│  西家特殊杠   │  红中状态     │  东家特殊杠   │
│  西家弃牌河   │  牌墙剩余数   │  东家弃牌河   │
│  西家玩家信息 │  東風1局      │  东家玩家信息 │
│              │              │               │
├──────────────┴──────────────┴───────────────┤
│  [我的手牌-正面+可点击+特殊杠]                │ ← 南方玩家（我）
│  [我的副露区] [我的特殊杠]                    │
│  [我的弃牌河]                                │
│  [我的玩家信息、分数]                        │
│  [操作按钮：吃/碰/杠/胡/过]                  │
└─────────────────────────────────────────────┘
```

### 2.2 武汉麻将特有元素（替换日麻元素）

**新增元素：**
- **赖子指示器**：显示当前赖子牌（翻牌+1）
- **皮指示器**：显示两张皮牌（翻牌-1）
- **红中状态**：红中是否被杠的标记
- **特殊杠展示**：赖子杠、皮子杠、红中杠（与普通杠区分显示）

**移除的日麻元素：**
- 立直棒、立直宣言牌
- 振听标记
- 宝牌显示

---

## 三、特殊牌视觉设计

### 3.1 赖子（万能牌）标识
- 麻将牌上叠加金色边框或光晕效果
- 中央显示"赖"字标记
- 在赖子指示器区域显示当前赖子牌
- 手牌中的赖子使用高亮显示

### 3.2 皮标识
- 麻将牌上叠加银色边框
- 在皮指示器区域显示两张皮
- 弃牌河中的皮牌使用特殊标记

### 3.3 红中标识
- 红中牌本身保持红色
- 红中杠时使用特殊动画（红色光效）
- 界面右上角显示红中状态

### 3.4 特殊杠展示（与普通杠区分）
- **赖子杠**：金色"赖"字图标
- **皮子杠**：银色"皮"字图标
- **红中杠**：红色"中"字图标
- **普通杠**（明杠/暗杠/补杠）：保持常规样式

---

## 四、明牌模式设计

### 4.1 功能说明
- 在玩家设置中添加"明牌模式"开关
- 开启后可以看到所有隐藏信息
- 适用于单机练习、调试、观战学习

### 4.2 明牌模式显示的额外信息
- 对手手牌（正面显示）
- 牌墙展开：点击牌墙区域，展开显示所有剩余牌
- 按照实际顺序排列

---

## 五、前后端通信协议设计

### 5.1 基于现有数据结构

**使用的核心数据结构：**
- `ActionType` 枚举（constants.py）
- `GameStateType` 枚举（constants.py）
- `MahjongAction` 类（mahjong_action.py）
- `GameContext` 类（GameData.py）
- `PlayerData` 类（PlayerData.py）

### 5.2 客户端 → 服务器消息

```json
// 加入房间
{
  "event": "join",
  "username": "玩家名",
  "mode": "play",           // play/observe
  "ai_config": [false, true, true, false]  // false=人类, true=AI
}

// 玩家动作
{
  "event": "action",
  "action": {
    "action_type": 0,       // ActionType枚举值
    "parameter": 34         // 牌ID或吃牌类型
  },
  "player_id": 0
}

// 切换明牌模式
{
  "event": "toggle_debug"
}

// 选择AI策略
{
  "event": "select_ai",
  "ai_config": {
    "player_id": 1,
    "model_name": "random"  // 或训练模型路径
  }
}
```

### 5.3 服务器 → 客户端消息

```json
// 游戏状态更新
{
  "event": "game_state",
  "context": {
    "current_state": 2,     // GameStateType枚举值
    "current_player_idx": 0,
    "dealer_idx": 0,
    "lazy_tile": 16,        // 赖子牌ID
    "skin_tile": [15, 17],  // 皮子牌ID
    "wall_count": 67,
    "players": [
      {
        "player_id": 0,
        "hand_tiles": [1, 2, 3, ...],   // 仅发给当前玩家
        "melds": [...],
        "discard_tiles": [...],
        "special_gangs": [0, 0, 0],     // [皮子杠, 赖子杠, 红中杠]
        "is_dealer": true,
        "is_win": false
      }
    ]
  }
}

// 明牌模式响应
{
  "event": "debug_mode",
  "enabled": true,
  "wall_tiles": [0, 1, 2, ...],
  "opponent_hands": [[...], [...]]
}

// 结算
{
  "event": "settlement",
  "winner_ids": [0],
  "win_way": 0,            // WinWay枚举值
  "win_type": 0,           // WinType枚举值
  "scores": [100, -50, -25, -25]
}
```

---

## 六、MVP 实现阶段划分

### 6.1 阶段 1：基础渲染（1-2周）
- 搭建 Phaser.js 基础项目结构
- 复用 Mahjong-AI 的麻将牌 sprite sheet
- 实现四方位手牌显示（背面/正面）
- 实现弃牌河和副露区渲染
- 实现武汉麻将特殊牌标识（赖子/皮/红中）
- 明牌模式切换功能

### 6.2 阶段 2：本地游戏逻辑（2-3周）
- 集成现有的 WuhanMahjongEnv
- 实现单人模式（1人类 + 3简单AI）
- 实现本地热座模式（4人类轮流）
- 操作按钮交互（吃/碰/杠/胡/过）
- 游戏状态实时更新
- 基础结算界面

### 6.3 阶段 3：回放系统（2-3周）
- 游戏录制（记录 GameContext 和动作序列）
- 回放界面（暂停/快进/慢放/步进）
- 回放文件保存和加载
- 回放时的明牌模式展示
- 导出回放数据

### 6.4 阶段 4：局域网对战（3-4周）
- FastAPI WebSocket 服务器搭建
- 房间管理（单房间）
- 支持任意数量人类+AI配置
- AI策略选择界面（随机AI/训练模型）
- 观战模式
- 明牌模式同步

---

## 七、关键技术实现细节

### 7.1 麻将牌渲染（复用 Mahjong-AI）
- 使用 Mahjong-AI 的 `vieww000072.png` 等 sprite sheet
- 牌ID映射：0-33 → sprite frame index
- 特殊牌叠加效果：赖子（金色边框+"赖"字）、皮子（银色边框）、红中（红色光效）

### 7.2 FastAPI 后端架构
```python
# src/mahjong_rl/web/game_server.py
class GameRoom:
    def __init__(self):
        self.env: WuhanMahjongEnv
        self.ai_config: List[Optional[AIAgent]]
        self.observers: List[WebSocket]
        self.debug_mode: bool = False

    async def handle_action(self, action: MahjongAction)
    async def broadcast_state(self)
    async def add_observer(self, ws: WebSocket)
```

### 7.3 回放数据格式
```json
{
  "game_id": "uuid",
  "timestamp": "2026-01-27",
  "players": ["玩家1", "AI1", "AI2", "AI3"],
  "ai_config": ["random", "random", "random"],
  "actions": [
    {
      "step": 0,
      "state": "PLAYER_DECISION",
      "action": {"action_type": 0, "parameter": 5},
      "player_id": 0
    }
  ]
}
```

### 7.4 AI 策略接口
```python
# 支持现有 RandomAgent 和训练模型
class AIAgent:
    def get_action(self, observation, action_mask) -> MahjongAction
```

---

## 八、技术栈总结

| 层级 | 技术选型 | 说明 |
|------|----------|------|
| 前端渲染 | Phaser.js | 参考 Mahjong-AI，复用 sprite sheet |
| 前端通信 | WebSocket | 实时游戏状态同步 |
| 后端框架 | FastAPI | WebSocket + HTTP API |
| 游戏逻辑 | WuhanMahjongEnv | 复用现有环境 |
| AI 接口 | AIAgent | 支持随机AI和训练模型 |

---

## 九、参考资源

- Mahjong-AI 项目：`D:\DATA\Python_Project\Code\PettingZooRLENVMahjong\Mahjong-AI`
- Phaser.js 文档：https://photonstorm.github.io/phaser-ce/
- FastAPI WebSocket 文档：https://fastapi.tiangolo.com/advanced/websockets/
- 武汉麻将规则：`src/mahjong_rl/rules/wuhan_mahjong_rule_engine/wuhan_mahjong_rules.md`
