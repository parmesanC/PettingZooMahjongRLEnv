# FastAPI版本快速开始指南

## 安装步骤

### 1. 安装依赖

```bash
# 方式1：安装所有依赖（推荐）
pip install -r requirements.txt

# 方式2：仅安装Web模式所需
pip install fastapi uvicorn websockets
```

### 2. 验证安装

```bash
python -c "import fastapi, uvicorn, websockets; print('✓ 所有依赖已安装')"
```

## 运行方式

### 方式1：测试服务器启动（无游戏）

```bash
python test_fastapi.py
```

访问：
- 游戏页面: http://localhost:8000
- API文档: http://localhost:8000/docs

### 方式2：完整游戏

```bash
# 人 vs 3AI
python play_mahjong.py --renderer web --mode human_vs_ai --human-player 0 --port 8000

# 观察模式
python play_mahjong.py --renderer web --mode observation --port 8000

# 4人热座
python play_mahjong.py --renderer web --mode four_human --port 8000
```

### 方式3：自定义端口

```bash
python play_mahjong.py --renderer web --port 8001
```

## 功能特点

### FastAPI优势

✅ **原生WebSocket支持** - 真正的实时双向通信  
✅ **自动静态文件服务** - `/static` 自动挂载  
✅ **异步高性能** - 基于Starlette异步框架  
✅ **Swagger API文档** - 访问 `/docs` 查看  
✅ **CORS支持** - 内置CORS中间件  
✅ **类型安全** - 完整的Pydantic支持

### Web界面功能

- 实时游戏状态更新
- 麻将牌汉字显示
- 点击按钮选择动作
- WebSocket自动重连（最多5次）
- 连接状态指示器（右上角）

## 文件说明

```
src/mahjong_rl/web/
├── fastapi_server.py      # FastAPI主服务器
├── websocket_manager.py  # WebSocket连接管理
└── static/
    └── game.html         # 麻将游戏HTML页面
```

## API端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 游戏主页面 |
| `/ws` | WebSocket | WebSocket连接 |
| `/docs` | GET | Swagger API文档 |
| `/static/*` | GET | 静态文件 |

## 故障排除

### 问题：ImportError: No module named 'fastapi'

```bash
# 解决：安装FastAPI
pip install fastapi uvicorn websockets
```

### 问题：端口占用

```bash
# 解决：使用其他端口
python play_mahjong.py --renderer web --port 8001
```

### 问题：WebSocket连接失败

```bash
# 解决：检查防火墙设置，确保端口可访问
# 查看浏览器控制台错误信息
```

## 性能调优

### Uvicorn工作进程数

```bash
# 多进程模式（生产环境推荐）
uvicorn app:app --workers 4 --host 0.0.0.0 --port 8000
```

### 开发模式

```bash
# 自动重载（开发时使用）
uvicorn app:app --reload
```

## 生产部署建议

### 使用Gunicorn + Uvicorn

```bash
pip install gunicorn
gunicorn src.mahjong_rl.web.fastapi_server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Docker部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "src.mahjong_rl.web.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 技术栈

- **后端**: FastAPI + Uvicorn
- **WebSocket**: websockets
- **前端**: HTML5 + CSS3 + Vanilla JavaScript
- **类型检查**: Pydantic
- **文档**: Swagger UI

## 对比：旧版 vs 新版

| 特性 | 旧版 (http.server) | 新版 (FastAPI) |
|------|---------------------|----------------|
| WebSocket | ❌ 预留未实现 | ✅ 完整支持 |
| 静态文件 | ❌ 手动处理 | ✅ 自动挂载 |
| 性能 | ❌ 同步阻塞 | ✅ 异步高性能 |
| API文档 | ❌ 无 | ✅ Swagger `/docs` |
| 类型安全 | ❌ 无 | ✅ Pydantic |
| CORS | ❌ 手动实现 | ✅ 内置中间件 |
| 调试 | ❌ 难以调试 | ✅ 清晰错误信息 |

## 下一步

1. 安装依赖：`pip install -r requirements.txt`
2. 启动服务器：`python play_mahjong.py --renderer web`
3. 打开浏览器：http://localhost:8000
4. 开始游戏！

## 支持和反馈

遇到问题？查看详细的故障排除说明：`MANUAL_CONTROL_README.md`
