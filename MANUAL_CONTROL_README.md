# 麻将Manual Control 使用说明

## 概述

本系统提供了符合PettingZoo AEC标准的手动控制功能，支持命令行(CLI)和网页(Web)两种可视化方式。

## 文件结构

```
src/mahjong_rl/
├── manual_control/
│   ├── __init__.py
│   ├── base.py                      # Manual Control基类
│   ├── cli_controller.py             # CLI控制器（元组输入）
│   └── web_controller.py             # Web控制器（FastAPI + WebSocket）
│
├── agents/
│   ├── __init__.py
│   ├── base.py                     # 策略基类
│   ├── human/
│   │   ├── __init__.py
│   │   └── manual_strategy.py       # 人类策略
│   └── ai/
│       ├── __init__.py
│       └── random_strategy.py         # 随机AI
│
├── visualization/
│   ├── __init__.py
│   ├── base.py                    # 可视化基类（已存在）
│   ├── cli_renderer.py             # CLI渲染器（含special_gangs显示）
│   └── web_renderer.py            # Web渲染器（HTML）
│
└── web/
    ├── __init__.py
    ├── fastapi_server.py            # FastAPI服务器
    ├── websocket_manager.py         # WebSocket管理器
    └── static/
        └── game.html                # 麻将游戏HTML页面

根目录:
├── play_mahjong.py                    # 主程序入口
├── requirements.txt                  # 依赖项
└── test_env.py                       # 环境测试脚本
```

## 安装依赖

### 全部安装
```bash
pip install -r requirements.txt
```

### CLI模式（不需要额外依赖）
CLI模式使用标准库，无需额外安装。

### Web模式（需要FastAPI）
```bash
pip install fastapi uvicorn websockets
```

## 使用方法

### 命令行模式 (CLI)

```bash
# 人 vs 3AI
python play_mahjong.py --renderer cli --mode human_vs_ai --human-player 0

# 4人热座
python play_mahjong.py --renderer cli --mode four_human

# 观察模式
python play_mahjong.py --renderer cli --mode observation
```

### 网页模式 (Web - FastAPI)

```bash
# 启动网页服务器
python play_mahjong.py --renderer web --mode human_vs_ai --human-player 0 --port 8000

# 然后在浏览器打开
# http://localhost:8000
# API文档: http://localhost:8000/docs
```

## 参数说明

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|----------|--------|
| --mode | 游戏模式 | human_vs_ai, four_human, observation | human_vs_ai |
| --renderer | 可视化方式 | cli(命令行), web(网页) | cli |
| --human-player | 人类玩家位置 | 0, 1, 2, 3 | 0 |
| --port | 网页服务器端口 | 8000-65535 | 8000 |
| --episodes | 回合数 | 1+ | 1 |
| --seed | 随机种子 | 整数 | None |

## CLI输入格式

CLI模式使用元组形式输入动作：

### 打牌
```
(0, 5)  # 打出5号牌（2万）
```

### 其他动作
```
(2, -1)  # 碰牌
(9, -1)  # 胡牌
(10, -1) # 过牌
```

### 牌ID对照表
```
万: 0=1万, 1=2万, 2=3万, 3=4万, 4=5万, 5=6万, 6=7万, 7=8万, 8=9万
条: 9=1条, 10=2条, 11=3条, 12=4条, 13=5条, 14=6条, 15=7条, 16=8条, 17=9条
筒: 18=1筒, 19=2筒, 20=3筒, 21=4筒, 22=5筒, 23=6筒, 24=7筒, 25=8筒, 26=9筒
字: 27=东风, 28=南风, 29=西风, 30=北风, 31=红中, 32=发财, 33=白板
```

## Web界面功能

### FastAPI特性

✅ **WebSocket原生支持** - 真正的实时双向通信  
✅ **自动静态文件服务** - `/static` 自动挂载  
✅ **异步高性能** - 基于Starlette异步框架  
✅ **完整类型提示** - Pydantic支持  
✅ **Swagger API文档** - 访问 `/docs` 查看  
✅ **CORS支持** - 内置CORS中间件  
✅ **连接状态显示** - 右上角显示WebSocket连接状态

### Web界面功能

- 🎮 实时游戏状态更新
- 🀄 麻将牌汉字显示
- 🖱️ 点击按钮选择动作
- 🔄 自动重连（最多5次）
- ✅ 连接状态指示器（右上角）

### API端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 游戏主页面 |
| `/ws` | WebSocket | WebSocket连接 |
| `/docs` | GET | Swagger API文档 |
| `/static/*` | GET | 静态文件 |

## 设计原则遵循

- ✅ **SRP**: 单一职责，每个类只负责一件事
- ✅ **OCP**: 开放封闭，通过基类扩展新策略
- ✅ **LSP**: 里氏替换，策略接口可互换
- ✅ **ISP**: 接口隔离，最小化接口
- ✅ **DIP**: 依赖倒置，依赖抽象而非具体实现
- ✅ **LKP**: 最少知识，模块间低耦合

## 特殊杠显示

武汉麻将规则中的特殊杠独立显示：

- **红中杠**: 计入 `special_gangs[2]`，**不计入 melds**
- **皮子杠**: 计入 `special_gangs[0]`
- **赖子杠**: 计入 `special_gangs[1]`

### CLI显示示例
```
你的手牌 (玩家0):
  万: 1万 2万 3万 ...
  条: 2条 3条 4条 ...
  筒: 1筒 1筒 2筒
  字: 东风
  副露: 
  特殊杠:
    红中杠: 2次
```

### Web显示示例

网页界面在"特殊杠"区域独立显示红中杠、皮子杠、赖子杠的次数，与普通副露分开。

## PettingZoo兼容性

- ✅ 使用 `env.agent_iter()` 循环
- ✅ 使用 `env.last()` 获取观测/奖励
- ✅ 使用 `env.step(action)` 执行动作
- ✅ 支持 Action Masking (`observation['action_mask']`)
- ✅ 返回标准格式: `(observation, reward, terminated, truncated, info)`

## 故障排除

### 端口占用
```
错误: [Errno 98] Address already in use
解决: 使用 --port 参数指定其他端口，如: --port 8001
```

### 导入错误
```
错误: 无法导入必要模块
解决: 确保位于项目根目录，运行: pip install -r requirements.txt
```

### WebSocket连接失败
```
错误: WebSocket连接失败
解决: 
1. 确保服务器正常启动
2. 检查浏览器控制台错误信息
3. 检查防火墙设置
```

## 技术栈对比

### CLI模式
| 特性 | 实现 |
|------|------|
| 界面 | 命令行 |
| 依赖 | Python标准库 |
| 性能 | 高 |
| 适用场景 | 本地测试 |

### Web模式 (FastAPI)
| 特性 | 实现 |
|------|------|
| 界面 | HTML5 + CSS3 |
| 后端 | FastAPI + Uvicorn |
| 通信 | WebSocket |
| 静态文件 | 自动挂载 |
| API文档 | Swagger |
| 性能 | 异步高性能 |
| 适用场景 | 远程访问、多人在线 |

## 注意事项

1. Web模式确保端口未被占用
2. 确保您的环境支持中文显示
3. 游戏逻辑依赖 `example_mahjong_env.py` 中的实现
4. WebSocket会在断开后自动重连（最多5次）
5. API文档访问: `http://localhost:8000/docs`

## 扩展建议

### 添加新的AI策略

1. 继承 `PlayerStrategy` 类
2. 实现 `choose_action()` 方法
3. 在 `play_mahjong.py` 中注册新策略

### 自定义Web界面

1. 修改 `src/mahjong_rl/web/static/game.html`
2. 确保WebSocket消息格式一致
3. 参考现有HTML结构

## API文档

访问 `http://localhost:8000/docs` 查看完整的API文档（仅Web模式）。
