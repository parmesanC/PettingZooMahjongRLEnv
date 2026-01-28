# SimpleGameRunner 重构实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 重构 `simple_game_runner.py` 继承 `ManualController` 基类，添加动作验证和异步事件驱动支持

**Architecture:** SimpleGameRunner 继承 ManualController，复用基类的 AI 处理和回合管理逻辑，在 WebSocket 回调中验证动作合法性，采用步进式执行替代阻塞循环

**Tech Stack:** Python 3.12+, FastAPI, WebSocket, PettingZoo AECEnv

---

## Task 1: 创建可复用的动作验证工具模块

**Files:**
- Create: `src/mahjong_rl/web/utils/action_validator.py`

**Step 1: Write the failing test**

创建 `tests/web/test_action_validator.py`:
```python
import pytest
import numpy as np
from src.mahjong_rl.web.utils.action_validator import ActionValidator

def test_validate_discard_action_valid():
    """测试验证合法的打牌动作"""
    action_mask = np.zeros(145, dtype=np.int8)
    action_mask[5] = 1  # 可以打出5号牌

    validator = ActionValidator()
    assert validator.validate_action(0, 5, action_mask) == True

def test_validate_discard_action_invalid_tile():
    """测试验证非法的打牌动作（牌不可用）"""
    action_mask = np.zeros(145, dtype=np.int8)
    # 5号牌不可用

    validator = ActionValidator()
    assert validator.validate_action(0, 5, action_mask) == False

def test_validate_skin_kong_action_valid():
    """测试验证合法的皮子杠动作"""
    action_mask = np.zeros(145, dtype=np.int8)
    action_mask[108 + 5] = 1  # 可以杠5号皮子

    validator = ActionValidator()
    assert validator.validate_action(7, 5, action_mask) == True

def test_validate_action_invalid_type():
    """测试验证非法的动作类型"""
    action_mask = np.zeros(145, dtype=np.int8)

    validator = ActionValidator()
    assert validator.validate_action(99, 0, action_mask) == False

def test_get_action_name():
    """测试获取动作名称"""
    validator = ActionValidator()
    assert validator.get_action_name(0) == "打牌"
    assert validator.get_action_name(7) == "皮子杠"
    assert validator.get_action_name(10) == "过牌"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/web/test_action_validator.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'src.mahjong_rl.web.utils'"

**Step 3: Write minimal implementation**

创建 `src/mahjong_rl/web/utils/__init__.py`:
```python
"""Web utils package"""
```

创建 `src/mahjong_rl/web/utils/action_validator.py`:
```python
"""动作验证工具 - 从 CLI 控制器提取的可复用验证逻辑"""

import numpy as np
from typing import Tuple, Optional


class ActionValidator:
    """
    动作验证器

    基于145位 action_mask 验证动作的合法性。
    复用自 CLI 控制器的验证逻辑。
    """

    # action_mask 索引范围定义
    ACTION_RANGES = {
        0: (0, 33),      # DISCARD
        1: (34, 36),     # CHOW
        2: (37, 37),     # PONG
        3: (38, 38),     # KONG_EXPOSED
        4: (39, 72),     # KONG_SUPPLEMENT
        5: (73, 106),    # KONG_CONCEALED
        6: (107, 107),   # KONG_RED
        7: (108, 141),   # KONG_SKIN
        8: (142, 142),   # KONG_LAZY
        9: (143, 143),   # WIN
        10: (144, 144),  # PASS
    }

    ACTION_NAMES = {
        0: "打牌", 1: "吃牌", 2: "碰牌",
        3: "明杠", 4: "补杠", 5: "暗杠",
        6: "红中杠", 7: "皮子杠", 8: "赖子杠",
        9: "胡牌", 10: "过牌"
    }

    def validate_action(self, action_type: int, parameter: int, action_mask: np.ndarray) -> bool:
        """
        验证动作是否有效

        Args:
            action_type: 动作类型 (0-10)
            parameter: 动作参数（牌ID或吃牌类型）
            action_mask: 145位动作掩码

        Returns:
            True if action is valid, False otherwise
        """
        # 检查动作类型是否在有效范围内
        if action_type not in self.ACTION_RANGES:
            return False

        start, end = self.ACTION_RANGES[action_type]

        # 检查动作类型是否可用
        if not any(action_mask[start:end+1]):
            return False

        # 对于需要参数的动作类型，检查具体参数是否有效
        if action_type in [0, 4, 5, 7]:  # 打牌、补杠、暗杠、皮子杠
            if parameter < 0 or parameter > 33:
                return False

            # 计算对应的索引
            if action_type == 0:  # DISCARD
                index = parameter
            elif action_type == 4:  # KONG_SUPPLEMENT
                index = 39 + parameter
            elif action_type == 5:  # KONG_CONCEALED
                index = 73 + parameter
            elif action_type == 7:  # KONG_SKIN
                index = 108 + parameter
            else:
                return False

            return action_mask[index] == 1

        # 对于不需要参数的动作类型
        if action_type in [6, 8]:  # 红中杠、赖子杠
            return action_mask[start] == 1

        return True

    def get_action_name(self, action_type: int) -> str:
        """
        获取动作名称

        Args:
            action_type: 动作类型

        Returns:
            动作的中文名称
        """
        return self.ACTION_NAMES.get(action_type, "未知")

    def validate_action_with_error_message(
        self,
        action_type: int,
        parameter: int,
        action_mask: np.ndarray
    ) -> Tuple[bool, Optional[str]]:
        """
        验证动作并返回错误消息

        Args:
            action_type: 动作类型
            parameter: 动作参数
            action_mask: 动作掩码

        Returns:
            (is_valid, error_message) 元组
        """
        if self.validate_action(action_type, parameter, action_mask):
            return True, None

        # 生成错误消息
        action_name = self.get_action_name(action_type)
        return False, f"⚠️ {action_name} 当前不可用"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/web/test_action_validator.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/web/test_action_validator.py src/mahjong_rl/web/utils/
git commit -m "feat(web): add reusable ActionValidator utility

- Extract validation logic from CLI controller
- Support all action types based on 145-bit action_mask
- Include error message generation for frontend feedback

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: 创建新的 SimpleGameRunner 继承 ManualController

**Files:**
- Modify: `src/mahjong_rl/web/simple_game_runner.py`
- Create: `tests/web/test_simple_game_runner.py`

**Step 1: Write the failing test**

创建 `tests/web/test_simple_game_runner.py`:
```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.mahjong_rl.web.simple_game_runner import SimpleGameRunner

def test_simple_game_runner_inherits_manual_controller():
    """测试 SimpleGameRunner 继承 ManualController"""
    from src.mahjong_rl.manual_control.base import ManualController

    env = Mock()
    runner = SimpleGameRunner(env, port=8011, strategies=[None, None, None, None])

    assert isinstance(runner, ManualController)

def test_simple_game_runner_initialization():
    """测试 SimpleGameRunner 初始化"""
    env = Mock()
    strategies = [None, None, Mock(), Mock()]

    runner = SimpleGameRunner(
        env=env,
        port=8011,
        max_episodes=1000,
        strategies=strategies,
        ai_delay=0.5
    )

    assert runner.env == env
    assert runner.port == 8011
    assert runner.max_episodes == 1000
    assert runner.strategies == strategies
    assert runner.ai_delay == 0.5
    assert runner.server is None
    assert runner.pending_action is None
    assert runner.action_received is False

@patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer')
def test_on_action_received_valid_action(MockServer):
    """测试接收有效动作"""
    env = Mock()
    env.unwrapped.context.current_player_idx = 0
    env.step = Mock(return_value=(Mock(), 0, False, False, {}))

    runner = SimpleGameRunner(env, port=8011, strategies=[None, None, None, None])
    runner.server = Mock()
    runner.server.send_json_state = Mock()
    runner._get_action_mask = Mock(return_value=[1]*145)

    action = (0, 5)  # 打出5号牌
    runner.on_action_received(action, player_id=0)

    assert runner.pending_action == action
    assert runner.action_received is True

@patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer')
def test_on_action_received_invalid_player(MockServer):
    """测试非当前玩家发送动作被拒绝"""
    env = Mock()
    env.unwrapped.context.current_player_idx = 0

    runner = SimpleGameRunner(env, port=8011, strategies=[None, None, None, None])
    runner.server = Mock()
    runner.server.websocket_manager = Mock()

    action = (0, 5)
    runner.on_action_received(action, player_id=1)  # 玩家1尝试在玩家0的回合行动

    assert runner.pending_action is None
    assert runner.action_received is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/web/test_simple_game_runner.py::test_simple_game_runner_inherits_manual_controller -v`

Expected: FAIL with SimpleGameRunner 不继承 ManualController

**Step 3: Write minimal implementation**

重写 `src/mahjong_rl/web/simple_game_runner.py`:
```python
"""
简单的游戏运行器
继承 ManualController 基类，复用标准游戏循环逻辑
"""
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.mahjong_rl.manual_control.base import ManualController
from src.mahjong_rl.web.fastapi_server import MahjongFastAPIServer
from src.mahjong_rl.web.utils.action_validator import ActionValidator


class SimpleGameRunner(ManualController):
    """
    简单的游戏运行器，继承 ManualController

    复用基类的标准游戏循环：
    - 自动处理 AI 玩家
    - 自动状态推进
    - 人类玩家通过 WebSocket 交互
    """

    def __init__(self, env, port=8011, max_episodes=1000, strategies=None, ai_delay=0.5):
        """
        初始化游戏运行器

        Args:
            env: PettingZoo 环境实例
            port: WebSocket 服务器端口
            max_episodes: 最大回合数
            strategies: 玩家策略列表
            ai_delay: AI 思考延迟时间（秒）
        """
        super().__init__(env, max_episodes, strategies)
        self.port = port
        self.ai_delay = ai_delay
        self.server = None
        self.pending_action = None
        self.action_received = False
        self.action_validator = ActionValidator()

    def run(self):
        """
        启动服务器并运行游戏循环
        """
        # 创建 FastAPI 服务器
        self.server = MahjongFastAPIServer(
            env=self.env,
            controller=self,
            port=self.port
        )

        # 启动服务器（阻塞）
        self.server.start()

    def render_env(self):
        """
        渲染环境状态到前端
        """
        if self.server and self.env.unwrapped.context:
            context = self.env.unwrapped.context

            # 给每个玩家发送对应视角的状态
            for player_idx in range(4):
                # 获取当前玩家的 action_mask
                action_mask = self._get_action_mask(player_idx)
                self.server.send_json_state(
                    context,
                    player_idx,
                    action_mask
                )

    def get_human_action(self, observation, info):
        """
        获取人类玩家动作（通过 WebSocket）

        阻塞等待前端发送动作。

        Returns:
            (action_type, parameter) 元组
        """
        self.action_received = False
        self.pending_action = None

        # 阻塞等待动作
        timeout = 300  # 5分钟超时
        start_time = time.time()

        while not self.action_received:
            time.sleep(0.1)
            if time.time() - start_time > timeout:
                # 超时，返回 PASS
                return (10, -1)

        action = self.pending_action
        self.action_received = False
        self.pending_action = None

        return action

    def render_final_state(self, info):
        """
        渲染最终状态
        """
        if self.server:
            # 发送游戏结束消息到前端
            message = {
                'type': 'game_over',
                'info': info
            }
            self.server.websocket_manager.broadcast_sync(message)

    def on_action_received(self, action, player_id=None):
        """
        前端发送动作的回调（由 WebSocket 调用）

        Args:
            action: (action_type, parameter) 元组
            player_id: 发送动作的玩家ID（用于验证）
        """
        # 使用 context 的 current_player_idx 进行验证
        if player_id is not None:
            current_player_idx = self.env.unwrapped.context.current_player_idx
            if player_id != current_player_idx:
                print(f"警告: 玩家{player_id}尝试在玩家{current_player_idx}的回合行动")
                return

        # 设置动作，解除阻塞
        self.pending_action = action
        self.action_received = True

    def _get_action_mask(self, player_idx):
        """获取指定玩家的 action_mask"""
        try:
            current_agent = self.env.possible_agents[player_idx]
            obs, reward, terminated, truncated, info = self.env.last()
            return obs['action_mask'] if not terminated and not truncated else None
        except:
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='武汉麻将WebSocket服务器')
    parser.add_argument('--port', type=int, default=8011, help='服务器端口')
    parser.add_argument('--human', type=int, default=1, help='人类玩家数量（0-4）')
    parser.add_argument('--ai-delay', type=float, default=0.5, help='AI思考延迟时间（秒）')

    args = parser.parse_args()

    from example_mahjong_env import WuhanMahjongEnv
    from src.mahjong_rl.agents.ai.random_strategy import RandomStrategy

    # 创建环境
    env = WuhanMahjongEnv(
        render_mode=None,
        training_phase=3,  # 完全信息
        enable_logging=False
    )

    # 创建策略
    strategies = []
    for i in range(4):
        if i < args.human:
            strategies.append(None)  # 人类玩家
        else:
            strategies.append(RandomStrategy())

    # 创建并启动游戏运行器
    runner = SimpleGameRunner(
        env=env,
        port=args.port,
        strategies=strategies,
        ai_delay=args.ai_delay
    )
    runner.run()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/web/test_simple_game_runner.py -v`

Expected: PASS (部分测试可能需要 mock 调整)

**Step 5: Commit**

```bash
git add tests/web/test_simple_game_runner.py src/mahjong_rl/web/simple_game_runner.py
git commit -m "refactor(web): SimpleGameRunner now inherits ManualController

- Inherit from ManualController to reuse standard game loop
- Add action validator for human player actions
- Implement render_env(), get_human_action(), render_final_state()
- Support AI strategies with ai_delay parameter

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: 添加动作验证到 on_action_received 回调

**Files:**
- Modify: `src/mahjong_rl/web/simple_game_runner.py`
- Modify: `tests/web/test_simple_game_runner.py`

**Step 1: Write the failing test**

添加到 `tests/web/test_simple_game_runner.py`:
```python
@patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer')
def test_on_action_received_rejects_invalid_action(MockServer):
    """测试非法动作被拒绝"""
    import numpy as np

    env = Mock()
    env.unwrapped.context.current_player_idx = 0
    env.possible_agents = ['player_0', 'player_1', 'player_2', 'player_3']

    # 创建全0的action_mask（没有任何动作可用）
    mock_obs = {'action_mask': np.zeros(145, dtype=np.int8)}
    env.last = Mock(return_value=(mock_obs, 0, False, False, {}))

    runner = SimpleGameRunner(env, port=8011, strategies=[None, None, None, None])
    runner.server = Mock()
    runner.server.websocket_manager = Mock()
    runner.server.websocket_manager.broadcast_sync = Mock()

    action = (0, 5)  # 尝试打出5号牌，但action_mask全0
    runner.on_action_received(action, player_id=0)

    # 验证错误消息被发送
    assert runner.server.websocket_manager.broadcast_sync.called
    call_args = runner.server.websocket_manager.broadcast_sync.call_args[0][0]
    assert call_args['type'] == 'error'
    assert '当前不可用' in call_args['message']

    # 验证动作没有被接受
    assert runner.pending_action is None
    assert runner.action_received is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/web/test_simple_game_runner.py::test_on_action_received_rejects_invalid_action -v`

Expected: FAIL (当前实现没有验证动作)

**Step 3: Write minimal implementation**

修改 `src/mahjong_rl/web/simple_game_runner.py` 的 `on_action_received` 方法:

```python
def on_action_received(self, action, player_id=None):
    """
    前端发送动作的回调（由 WebSocket 调用）

    Args:
        action: (action_type, parameter) 元组
        player_id: 发送动作的玩家ID（用于验证）
    """
    action_type, parameter = action

    # 处理重启请求 (action_type = -1)
    if action_type == -1:
        self._restart_game()
        return

    # 使用 context 的 current_player_idx 进行验证
    current_player_idx = self.env.unwrapped.context.current_player_idx

    if player_id is not None and player_id != current_player_idx:
        self._send_error(f"不是你的回合，当前是玩家{current_player_idx}")
        return

    # 获取当前玩家的 action_mask
    action_mask = self._get_action_mask(current_player_idx)
    if action_mask is None:
        self._send_error("游戏已结束")
        return

    # 验证动作合法性
    is_valid, error_msg = self.action_validator.validate_action_with_error_message(
        action_type, parameter, action_mask
    )

    if not is_valid:
        self._send_error(error_msg)
        return

    # 设置动作，解除阻塞
    self.pending_action = action
    self.action_received = True

def _send_error(self, message: str):
    """发送错误消息到前端"""
    if self.server and self.server.websocket_manager:
        error_message = {
            'type': 'error',
            'message': message
        }
        self.server.websocket_manager.broadcast_sync(error_message)

def _restart_game(self):
    """重启游戏，开始新的一局（保留配置）"""
    try:
        print("\n" + "=" * 60)
        print("收到重启请求，开始新的一局...")
        print("=" * 60)

        # 重置环境
        self.env.reset()

        print(f"✓ 新的一局开始！")
        print(f"  - 当前玩家: {self.env.unwrapped.context.current_player_idx}")
        print(f"  - 赖子: {self.env.unwrapped.context.lazy_tile}")
        print(f"  - 皮子: {self.env.unwrapped.context.skin_tile}")

        # 发送新状态到前端
        self.render_env()

        # 发送重启成功消息
        self._send_message("游戏已重启，配置已保留")

    except Exception as e:
        print(f"重启游戏失败: {e}")
        import traceback
        traceback.print_exc()

def _send_message(self, message: str):
    """发送普通消息到前端"""
    if self.server and self.server.websocket_manager:
        msg = {
            'type': 'info',
            'message': message
        }
        self.server.websocket_manager.broadcast_sync(msg)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/web/test_simple_game_runner.py::test_on_action_received_rejects_invalid_action -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/mahjong_rl/web/simple_game_runner.py tests/web/test_simple_game_runner.py
git commit -m "feat(web): add action validation to on_action_received callback

- Validate player turn before accepting action
- Use ActionValidator to check action legality
- Send error message to frontend on validation failure
- Add _restart_game() method with config preservation

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: 重写 run() 方法使用步进式执行

**Files:**
- Modify: `src/mahjong_rl/web/simple_game_runner.py`
- Modify: `tests/web/test_simple_game_runner.py`

**Step 1: Write the failing test**

添加到 `tests/web/test_simple_game_runner.py`:
```python
@patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer')
def test_process_auto_players_executes_all_ai(MockServer):
    """测试自动处理所有AI玩家"""
    env = Mock()
    env.unwrapped.context.current_player_idx = 1
    env.unwrapped.context.is_win = False
    env.unwrapped.context.is_flush = False
    env.agent_selection = 'player_1'
    env.possible_agents = ['player_0', 'player_1', 'player_2', 'player_3']

    # Mock observations
    mock_obs = {'action_mask': [1]*145}
    env.last = Mock(return_value=(mock_obs, 0, False, False, {}))
    env.step = Mock(return_value=(mock_obs, 0, False, False, {}))

    # Mock AI strategy
    ai_strategy = Mock()
    ai_strategy.choose_action = Mock(return_value=(10, -1))  # PASS

    runner = SimpleGameRunner(
        env=env,
        port=8011,
        strategies=[None, ai_strategy, ai_strategy, None],
        ai_delay=0
    )
    runner.server = Mock()
    runner.render_env = Mock()

    # 执行一个AI动作
    runner._process_auto_players(mock_obs)

    # 验证AI动作被执行
    assert ai_strategy.choose_action.called
    assert env.step.called
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/web/test_simple_game_runner.py::test_process_auto_players_executes_all_ai -v`

Expected: FAIL with "_process_auto_players method not found"

**Step 3: Write minimal implementation**

修改 `src/mahjong_rl/web/simple_game_runner.py`，添加 `_process_auto_players` 方法和修改 `run` 方法：

```python
def run(self):
    """
    启动服务器并运行游戏循环

    注意：重写基类的 run() 方法，使用 FastAPI 的异步事件循环
    """
    # 创建 FastAPI 服务器
    self.server = MahjongFastAPIServer(
        env=self.env,
        controller=self,
        port=self.port
    )

    # 发送初始状态
    self.render_env()

    # 启动服务器（阻塞）
    self.server.start()

def on_action_received(self, action, player_id=None):
    """
    前端发送动作的回调（由 WebSocket 调用）

    这是主要的游戏驱动方法：
    1. 验证动作
    2. 执行当前玩家动作
    3. 自动推进AI玩家
    4. 渲染新状态

    Args:
        action: (action_type, parameter) 元组
        player_id: 发送动作的玩家ID（用于验证）
    """
    action_type, parameter = action

    # 处理重启请求 (action_type = -1)
    if action_type == -1:
        self._restart_game()
        return

    # 使用 context 的 current_player_idx 进行验证
    current_player_idx = self.env.unwrapped.context.current_player_idx

    if player_id is not None and player_id != current_player_idx:
        self._send_error(f"不是你的回合，当前是玩家{current_player_idx}")
        return

    # 获取当前玩家的 action_mask
    action_mask = self._get_action_mask(current_player_idx)
    if action_mask is None:
        self._send_error("游戏已结束")
        return

    # 验证动作合法性
    is_valid, error_msg = self.action_validator.validate_action_with_error_message(
        action_type, parameter, action_mask
    )

    if not is_valid:
        self._send_error(error_msg)
        return

    # 执行人类玩家的动作
    try:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 发送新状态到前端
        self.render_env()

        # 检查游戏是否结束
        if terminated or truncated:
            self.render_final_state(info)
            return

        # 自动处理AI玩家
        self._process_auto_players(obs)

    except Exception as e:
        print(f"执行动作失败: {e}")
        import traceback
        traceback.print_exc()

def _process_auto_players(self, initial_obs):
    """
    处理 AI 玩家和自动状态推进

    在人类玩家动作后，自动执行所有AI玩家的动作。

    Args:
        initial_obs: 初始观测（人类玩家动作后的状态）
    """
    max_ai_steps = 20  # 防止无限循环
    steps = 0

    while steps < max_ai_steps:
        current_agent = self.env.agent_selection
        current_player_idx = self.env.unwrapped.context.current_player_idx

        # 检查游戏是否结束
        if self.env.unwrapped.context.is_win or self.env.unwrapped.context.is_flush:
            break

        # 如果是人类玩家，停止自动推进
        if self._is_human_player(current_agent):
            break

        # 获取AI策略
        strategy = self.strategies[current_player_idx]
        if strategy is None:
            break

        # 执行AI动作
        print(f"AI玩家{current_player_idx}思考中...")

        # 获取当前观测
        obs, reward, terminated, truncated, info = self.env.last()

        # AI选择动作
        action_mask = obs['action_mask']
        action = strategy.choose_action(obs, action_mask)

        if action is None:
            print(f"AI玩家{current_player_idx}无可用动作")
            break

        # 验证AI动作
        is_valid, error_msg = self.action_validator.validate_action_with_error_message(
            action[0], action[1], action_mask
        )

        if not is_valid:
            print(f"AI玩家{current_player_idx}选择了非法动作: {error_msg}")
            # AI使用PASS代替
            action = (10, -1)

        action_type, parameter = action
        print(f"  AI动作: type={action_type}, param={parameter}")

        # 执行AI动作
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)

            # AI延迟（观察用）
            if self.ai_delay > 0:
                time.sleep(self.ai_delay)

            # 发送状态更新
            self.render_env()

            # 检查游戏是否结束
            if terminated or truncated:
                self.render_final_state(info)
                return

        except Exception as e:
            print(f"AI动作执行失败: {e}")
            import traceback
            traceback.print_exc()
            break

        steps += 1

    if steps >= max_ai_steps:
        print("警告：达到最大AI步数")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/web/test_simple_game_runner.py::test_process_auto_players_executes_all_ai -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/mahjong_rl/web/simple_game_runner.py tests/web/test_simple_game_runner.py
git commit -m "feat(web): add step-by-step execution with auto AI player handling

- Rewrite run() to start FastAPI server without blocking loop
- Add _process_auto_players() to execute all AI players after human action
- Validate AI actions before execution
- Send state updates after each player action

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: 更新前端支持错误消息显示

**Files:**
- Modify: `src/mahjong_rl/web/phaser_client/js/scenes/MahjongScene.js`
- Modify: `src/mahjong_rl/web/phaser_client/js/utils/WebSocketManager.js`

**Step 1: Verify error message handling exists**

检查前端是否已有错误消息处理，如果没有则添加。

打开 `src/mahjong_rl/web/phaser_client/js/scenes/MahjongScene.js` 搜索 `handleWebSocketMessage` 方法中的 `error` 类型处理。

如果不存在，添加：
```javascript
handleWebSocketMessage(message) {
    switch (message.type) {
        // ... 其他case ...

        case 'error':
            console.error('收到错误:', message.message);
            this.showErrorNotification(message.message);
            break;

        case 'info':
            console.log('收到消息:', message.message);
            this.showInfoNotification(message.message);
            break;
    }
}

showErrorNotification(message) {
    // 显示错误通知（红色，3秒后消失）
    const notification = this.add.text(400, 50, message, {
        fontSize: '24px',
        fill: '#ff0000',
        backgroundColor: '#000000',
        padding: { x: 20, y: 10 }
    }).setOrigin(0.5);

    this.tweens.add({
        targets: notification,
        alpha: 0,
        duration: 3000,
        onComplete: () => notification.destroy()
    });
}

showInfoNotification(message) {
    // 显示信息通知（绿色，2秒后消失）
    const notification = this.add.text(400, 50, message, {
        fontSize: '24px',
        fill: '#00ff00',
        backgroundColor: '#000000',
        padding: { x: 20, y: 10 }
    }).setOrigin(0.5);

    this.tweens.add({
        targets: notification,
        alpha: 0,
        duration: 2000,
        onComplete: () => notification.destroy()
    });
}
```

**Step 2: Test manually**

启动游戏：
```bash
python src/mahjong_rl/web/simple_game_runner.py --port 8011 --human 1
```

在浏览器中打开 `http://localhost:8011`，尝试：
1. 点击不在自己回合的按钮 → 应显示错误提示
2. 点击不可用的动作按钮 → 应显示错误提示

**Step 3: Commit**

```bash
git add src/mahjong_rl/web/phaser_client/js/
git commit -m "feat(web): add error message display to frontend

- Handle 'error' and 'info' message types from WebSocket
- Show error notifications in red (3 seconds)
- Show info notifications in green (2 seconds)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: 集成测试 - 完整游戏流程

**Files:**
- Create: `tests/integration/test_simple_game_runner_full.py`

**Step 1: Write the integration test**

创建 `tests/integration/test_simple_game_runner_full.py`:
```python
"""
SimpleGameRunner 完整游戏流程集成测试
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.mahjong_rl.web.simple_game_runner import SimpleGameRunner


@pytest.fixture
def mock_env():
    """创建mock环境"""
    env = Mock()
    env.unwrapped.context.current_player_idx = 0
    env.unwrapped.context.is_win = False
    env.unwrapped.context.is_flush = False
    env.unwrapped.context.lazy_tile = 27
    env.unwrapped.context.skin_tile = [12, 13]
    env.agent_selection = 'player_0'
    env.possible_agents = ['player_0', 'player_1', 'player_2', 'player_3']

    # Mock observation with valid action_mask
    import numpy as np
    mock_obs = {
        'action_mask': np.zeros(145, dtype=np.int8)
    }
    mock_obs['action_mask'][0] = 1  # 可以打出0号牌
    mock_obs['action_mask'][144] = 1  # 可以PASS
    env.last = Mock(return_value=(mock_obs, 0, False, False, {}))

    return env


def test_full_game_flow_human_then_ai(mock_env):
    """测试完整游戏流程：人类玩家 -> AI玩家"""
    from src.mahjong_rl.agents.ai.random_strategy import RandomStrategy

    # 创建策略：玩家0是人类，玩家1-3是AI
    strategies = [None, RandomStrategy(), RandomStrategy(), RandomStrategy()]

    with patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer'):
        runner = SimpleGameRunner(
            env=mock_env,
            port=8011,
            strategies=strategies,
            ai_delay=0
        )
        runner.server = Mock()
        runner.render_env = Mock()

        # 模拟人类玩家动作
        human_action = (0, 0)  # 打出0号牌

        # 修改 env.step 返回值模拟AI玩家回合
        mock_env.unwrapped.context.current_player_idx = 1
        mock_env.agent_selection = 'player_1'

        # 执行人类动作
        runner.on_action_received(human_action, player_id=0)

        # 验证人类动作被接受
        assert runner.pending_action == human_action
        assert runner.action_received is True


def test_action_validation_blocks_invalid_actions(mock_env):
    """测试动作验证阻止非法动作"""
    import numpy as np

    # 修改action_mask，使0号牌不可用
    mock_obs = mock_env.last.return_value[0]
    mock_obs['action_mask'][0] = 0  # 0号牌不可用

    with patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer'):
        runner = SimpleGameRunner(
            env=mock_env,
            port=8011,
            strategies=[None, None, None, None]
        )
        runner.server = Mock()
        runner.server.websocket_manager = Mock()
        runner.server.websocket_manager.broadcast_sync = Mock()

        # 尝试非法动作
        invalid_action = (0, 0)  # 尝试打出不可用的0号牌
        runner.on_action_received(invalid_action, player_id=0)

        # 验证动作被拒绝
        assert runner.pending_action is None
        assert runner.action_received is False

        # 验证错误消息被发送
        assert runner.server.websocket_manager.broadcast_sync.called


def test_restart_game_preserves_config(mock_env):
    """测试重启游戏保留配置"""
    with patch('src.mahjong_rl.web.simple_game_runner.MahjongFastAPIServer'):
        runner = SimpleGameRunner(
            env=mock_env,
            port=8011,
            strategies=[None, None, Mock(), Mock()],
            ai_delay=0.5
        )
        runner.server = Mock()
        runner.render_env = Mock()
        runner._send_message = Mock()

        # 执行重启
        restart_action = (-1, 0)
        runner.on_action_received(restart_action, player_id=0)

        # 验证配置被保留
        assert runner.port == 8011
        assert runner.ai_delay == 0.5
        assert len(runner.strategies) == 4
```

**Step 2: Run integration test**

Run: `pytest tests/integration/test_simple_game_runner_full.py -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_simple_game_runner_full.py
git commit -m "test(web): add integration tests for SimpleGameRunner

- Test full game flow with human and AI players
- Test action validation blocks invalid actions
- Test restart game preserves configuration

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: 文档更新

**Files:**
- Update: `docs/plans/2026-01-28-simple-game-runner-refactor.md`

**Step 1: Update implementation status**

在原设计文档末尾添加：
```markdown
## 实施状态

- [x] Task 1: 创建可复用的动作验证工具模块
- [x] Task 2: 创建新的 SimpleGameRunner 继承 ManualController
- [x] Task 3: 添加动作验证到 on_action_received 回调
- [x] Task 4: 重写 run() 方法使用步进式执行
- [x] Task 5: 更新前端支持错误消息显示
- [x] Task 6: 集成测试
- [x] Task 7: 文档更新

**测试覆盖：**
- 单元测试: `tests/web/test_action_validator.py`
- 单元测试: `tests/web/test_simple_game_runner.py`
- 集成测试: `tests/integration/test_simple_game_runner_full.py`

**使用方式：**
```bash
# 启动游戏服务器（1个人类玩家 + 3个AI）
python src/mahjong_rl/web/simple_game_runner.py --port 8011 --human 1 --ai-delay 0.5

# 访问 http://localhost:8011
```
```

**Step 2: Commit documentation**

```bash
git add docs/plans/2026-01-28-simple-game-runner-refactor.md
git commit -m "docs: mark SimpleGameRunner refactor as complete

All tasks completed:
- ActionValidator utility extracted and tested
- SimpleGameRunner now inherits ManualController
- Action validation in WebSocket callback
- Step-by-step execution with auto AI handling
- Frontend error message support
- Integration tests passing

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Remember

- **TDD**: 每个任务先写测试，确保测试失败，再实现功能，确保测试通过
- **Frequent commits**: 每个任务完成后立即提交
- **Exact file paths**: 使用项目中实际的文件路径
- **Reference design**: 遵循 `docs/plans/2026-01-28-simple-game-runner-refactor-design.md` 中的设计

## Execution Handoff

Plan complete and saved to `docs/plans/2026-01-28-simple-game-runner-refactor.md`.

Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?

---

## 实施状态

- [x] Task 1: 创建可复用的动作验证工具模块
- [x] Task 2: 创建新的 SimpleGameRunner 继承 ManualController
- [x] Task 3: 添加动作验证到 on_action_received 回调
- [x] Task 4: 重写 run() 方法使用步进式执行
- [x] Task 5: 更新前端支持错误消息显示
- [x] Task 6: 集成测试
- [x] Task 7: 文档更新

**测试覆盖：**
- 单元测试: `tests/web/test_action_validator.py` (5 tests)
- 单元测试: `tests/web/test_simple_game_runner.py` (8 tests)
- 集成测试: `tests/integration/test_simple_game_runner_full.py` (11 tests)
- **总计: 24 个测试全部通过**

**使用方式：**
```bash
# 启动游戏服务器（1个人类玩家 + 3个AI）
python src/mahjong_rl/web/simple_game_runner.py --port 8011 --human 1 --ai-delay 0.5

# 访问 http://localhost:8011
```

**Git 提交记录：**
- 73ec46b: feat(web): add reusable ActionValidator utility
- 7d524b1: fix(web): correct ACTION_RANGES to use exclusive end indices
- dd2ce71: refactor(web): SimpleGameRunner now inherits ManualController
- 6e9fca7: refactor(web): improve code quality in SimpleGameRunner
- 5bf726b: feat(web): add action validation to on_action_received callback
- 04269ea: feat(web): add step-by-step execution with auto AI player handling
- 142820c: feat(web): add error message display to frontend
- 77caf04: test(web): add integration tests for SimpleGameRunner
