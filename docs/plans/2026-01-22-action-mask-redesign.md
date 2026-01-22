# 设计文档：重构 action_mask 结构

## 文档信息

| 项目 | 内容 |
|------|------|
| **目标** | 重构 action_mask 结构，解决参数混淆问题 |
| **日期** | 2026-01-22 |
| **问题** | 当前 action_mask 的 params 是统一掩码，不同动作类型的参数含义混淆 |
| **方案** | 改为扁平化二进制数组，每个动作类型有独立的参数段 |

---

## 1. 问题分析

### 1.1 当前问题

当前 `action_mask` 的结构：
```python
action_mask = {
    'types': [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 11维布尔数组
    'params': [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, ...]  # 35维布尔数组
}
```

**存在的问题**：
1. `params` 是统一的掩码，但不同动作类型的参数含义完全不同
2. 无法表示"红中杠的参数固定为31"这种约束
3. 策略网络难以理解哪个参数对应哪个动作类型
4. RandomStrategy 只处理了少数动作类型，特殊杠没有正确处理

### 1.2 示例问题

```python
# 当前错误的表示
action_mask = {
    'types': [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # DISCARD 和 KONG_RED 可用
    'params': [0, 1, 0, ..., 1, 0, ...]  # 哪些位是 DISCARD 的？哪些是 KONG_RED 的？
}
# 无法区分：红中杠(6)的参数应该是31，但 params 中其他位置的1是什么意思？
```

---

## 2. 新的 action_mask 设计

### 2.1 扁平化二进制掩码结构

action_mask 改为固定长度的一维二进制数组：

| 动作类型 | 起始索引 | 长度 | 含义 |
|---------|---------|------|------|
| DISCARD (0) | 0 | 34 | 34种牌是否可打出 |
| CHOW (1) | 34 | 3 | 左吃/中吃/右吃是否可用 |
| PONG (2) | 37 | 1 | 是否可碰 |
| KONG_EXPOSED (3) | 38 | 34 | 哪种牌可明杠 |
| KONG_SUPPLEMENT (4) | 72 | 34 | 哪种牌可补杠 |
| KONG_CONCEALED (5) | 106 | 34 | 哪种牌可暗杠 |
| KONG_RED (6) | 140 | 34 | 红中杠（仅红中位置为1） |
| KONG_LAZY (7) | 174 | 34 | 赖子杠（仅赖子位置为1） |
| KONG_SKIN (8) | 208 | 68 | 两个皮子杠（各34位） |
| WIN (9) | 276 | 1 | 是否可胡 |
| PASS (10) | 277 | 1 | 是否可过 |

**总长度：278 位**

### 2.2 观测空间定义

```python
# example_mahjong_env.py

def _create_observation_spaces(self) -> Dict[str, spaces.Dict]:
    """创建观测空间"""
    observation_spaces = {}

    for agent in self.possible_agents:
        observation_spaces[agent] = spaces.Dict({
            'global_hand': spaces.MultiDiscrete([6] * (4 * 34)),
            'private_hand': spaces.MultiDiscrete([6] * 34),
            'discard_pool_total': spaces.MultiDiscrete([6] * 34),
            'wall': spaces.MultiDiscrete([35] * 82),
            'melds': spaces.Dict({
                'action_types': spaces.MultiDiscrete([11] * 16),
                'tiles': spaces.MultiDiscrete([35] * 256),
                'group_indices': spaces.MultiDiscrete([4] * 32),
            }),
            'action_history': spaces.Dict({
                'types': spaces.MultiDiscrete([11] * 80),
                'params': spaces.MultiDiscrete([35] * 80),
                'players': spaces.MultiDiscrete([4] * 80),
            }),
            'special_gangs': spaces.MultiDiscrete([8, 4, 5] * 4),
            'current_player': spaces.MultiDiscrete([4]),
            'fan_counts': spaces.MultiDiscrete([600] * 4),
            'special_indicators': spaces.MultiDiscrete([34, 34]),
            'remaining_tiles': spaces.Discrete(137),
            'dealer': spaces.MultiDiscrete([4]),
            'current_phase': spaces.Discrete(8),
            'action_mask': spaces.MultiBinary(278),  # 新增
        })

    return observation_spaces
```

---

## 3. Wuhan7P4LObservationBuilder.build_action_mask() 重写

### 3.1 索引常量定义

```python
# src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py

# 在文件开头添加索引常量
ACTION_MASK_RANGES = {
    'DISCARD': (0, 34),
    'CHOW': (34, 37),
    'PONG': (37, 38),
    'KONG_EXPOSED': (38, 72),
    'KONG_SUPPLEMENT': (72, 106),
    'KONG_CONCEALED': (106, 140),
    'KONG_RED': (140, 174),
    'KONG_LAZY': (174, 208),
    'KONG_SKIN': (208, 276),
    'WIN': (276, 277),
    'PASS': (277, 278),
}
```

### 3.2 完整实现

```python
def build_action_mask(self, player_id: int, context: GameContext) -> np.ndarray:
    """
    构建动作掩码 - 返回扁平化的278位二进制数组

    Returns:
        np.ndarray: 形状为 (278,) 的二进制数组
    """
    mask = np.zeros(278, dtype=np.int8)

    current_state = context.current_state
    player = context.players[player_id]

    # 检查手牌是否已正确初始化
    if len(player.hand_tiles) < 13:
        return mask

    if current_state in [GameStateType.PLAYER_DECISION, GameStateType.DRAWING]:
        mask = self._build_draw_mask(player, context, mask)

    elif current_state in [GameStateType.WAITING_RESPONSE, GameStateType.RESPONSES,
                           GameStateType.RESPONSES_AFTER_GONG]:
        mask = self._build_response_mask(player, context, mask)

    return mask

def _build_draw_mask(self, player, context, mask):
    """构建摸牌后的动作掩码"""
    last_tile = context.last_drawn_tile
    if last_tile is None and len(player.hand_tiles) > 0:
        last_tile = player.hand_tiles[-1]

    if last_tile is not None:
        actions = ActionValidator(context).detect_available_actions_after_draw(
            player, last_tile
        )

        for action in actions:
            action_type = action.action_type.value

            if action_type == ActionType.DISCARD.value:
                # DISCARD: 标记手牌中所有可打出的牌 (索引 0-33)
                hand_counts = self._get_hand_counts(player.hand_tiles)
                for tile_id in range(34):
                    if hand_counts[tile_id] > 0:
                        mask[tile_id] = 1

            elif action_type == ActionType.CHOW.value:
                # CHOW: 标记具体吃法 (索引 34-36)
                chow_param = action.parameter  # 0=左, 1=中, 2=右
                mask[34 + chow_param] = 1

            elif action_type == ActionType.PONG.value:
                mask[37] = 1  # PONG 位

            elif action_type == ActionType.KONG_EXPOSED.value:
                mask[38 + action.parameter] = 1

            elif action_type == ActionType.KONG_SUPPLEMENT.value:
                mask[72 + action.parameter] = 1

            elif action_type == ActionType.KONG_CONCEALED.value:
                mask[106 + action.parameter] = 1

            elif action_type == ActionType.KONG_RED.value:
                # 红中杠：固定在红中位置 (31)
                mask[140 + 31] = 1

            elif action_type == ActionType.KONG_LAZY.value:
                # 赖子杠：赖子位置
                lazy_tile = context.lazy_tile
                if lazy_tile is not None:
                    mask[174 + lazy_tile] = 1

            elif action_type == ActionType.KONG_SKIN.value:
                # 皮子杠：两个皮子位置 (索引 208-275)
                for i, skin_tile in enumerate(context.skin_tile):
                    if skin_tile != -1:
                        mask[208 + i * 34 + skin_tile] = 1

            elif action_type == ActionType.WIN.value:
                mask[276] = 1  # WIN 位

    return mask

def _build_response_mask(self, player, context, mask):
    """构建响应状态的动作掩码"""
    discard_tile = context.last_discarded_tile
    discard_player = context.discard_player

    if discard_tile is not None and discard_player is not None:
        actions = ActionValidator(context).detect_available_actions_after_discard(
            player, discard_tile, discard_player
        )

        for action in actions:
            action_type = action.action_type.value

            if action_type == ActionType.CHOW.value:
                mask[34 + action.parameter] = 1

            elif action_type == ActionType.PONG.value:
                mask[37] = 1

            elif action_type == ActionType.KONG_EXPOSED.value:
                mask[38 + action.parameter] = 1

            elif action_type == ActionType.WIN.value:
                mask[276] = 1

        # PASS 在响应状态总是可用
        mask[277] = 1

    return mask

def _get_hand_counts(self, hand_tiles: List[int]) -> np.ndarray:
    """获取手牌中每种牌的数量"""
    counts = np.zeros(34, dtype=np.int8)
    for tile in hand_tiles:
        if 0 <= tile < 34:
            counts[tile] = min(counts[tile] + 1, 4)
    return counts
```

---

## 4. RandomStrategy 修改

### 4.1 新的 choose_action 逻辑

```python
# src/mahjong_rl/agents/ai/random_strategy.py

def choose_action(self, observation: Dict, action_mask: np.ndarray) -> Tuple[int, int]:
    """
    根据扁平化 action_mask 随机选择动作

    Args:
        observation: 观测字典（action_mask 现在是 278 位的一维数组）
        action_mask: 扁平化的 278 位动作掩码

    Returns:
        (action_type, parameter) 元组
    """
    # 定义索引范围
    RANGES = {
        'DISCARD': (0, 34),
        'CHOW': (34, 37),
        'PONG': (37, 38),
        'KONG_EXPOSED': (38, 72),
        'KONG_SUPPLEMENT': (72, 106),
        'KONG_CONCEALED': (106, 140),
        'KONG_RED': (140, 174),
        'KONG_LAZY': (174, 208),
        'KONG_SKIN': (208, 276),
        'WIN': (276, 277),
        'PASS': (277, 278),
    }

    # 收集所有可用的动作类型
    available_actions = []

    # 检查每个动作类型
    for action_type, (start, end) in RANGES.items():
        segment = action_mask[start:end]

        if np.any(segment > 0):
            # 该动作类型可用
            action_type_value = ActionType[action_type].value

            if action_type in ['DISCARD', 'KONG_EXPOSED', 'KONG_SUPPLEMENT',
                              'KONG_CONCEALED']:
                # 需要参数：从可用的牌ID中随机选择
                valid_params = np.where(segment > 0)[0]
                if len(valid_params) > 0:
                    param = int(np.random.choice(valid_params))
                    available_actions.append((action_type_value, param))

            elif action_type == 'CHOW':
                # 吃法：0=左, 1=中, 2=右
                valid_chows = np.where(segment > 0)[0]
                if len(valid_chows) > 0:
                    param = int(np.random.choice(valid_chows))
                    available_actions.append((action_type_value, param))

            elif action_type in ['KONG_RED', 'KONG_LAZY']:
                # 特殊杠：从掩码中找出具体牌ID
                valid_tiles = np.where(segment > 0)[0]
                if len(valid_tiles) > 0:
                    param = int(np.random.choice(valid_tiles))
                    available_actions.append((action_type_value, param))

            elif action_type == 'KONG_SKIN':
                # 皮子杠：两个皮子，各34位
                for i in range(2):
                    skin_segment = segment[i * 34:(i + 1) * 34]
                    valid_tiles = np.where(skin_segment > 0)[0]
                    if len(valid_tiles) > 0:
                        param = int(np.random.choice(valid_tiles))
                        available_actions.append((action_type_value, param))
                        break  # 只选一个皮子杠

            elif action_type in ['PONG', 'WIN', 'PASS']:
                # 无参数动作
                available_actions.append((action_type_value, 0))

    # 如果没有可用动作，返回默认
    if len(available_actions) == 0:
        return (ActionType.PASS.value, 0)

    # 随机选择一个动作
    return tuple(available_actions[np.random.choice(len(available_actions))])
```

---

## 5. 文件修改清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `example_mahjong_env.py` | 修改观测空间定义 | 添加 `action_mask: spaces.MultiBinary(278)` |
| `src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py` | 重写 `build_action_mask()` | 返回 278 位扁平化数组 |
| `src/mahjong_rl/agents/ai/random_strategy.py` | 重写 `choose_action()` | 适配新的 action_mask 格式 |

---

## 6. 测试计划

### 6.1 单元测试

```python
# test_action_mask.py

def test_action_mask_shape():
    """测试 action_mask 的形状"""
    env = WuhanMahjongEnv()
    obs, info = env.reset()

    assert obs['action_mask'].shape == (278,), \
        f"Expected shape (278,), got {obs['action_mask'].shape}"

def test_action_mask_discard():
    """测试 DISCARD 掩码"""
    env = WuhanMahjongEnv()
    obs, info = env.reset()

    # 检查 DISCARD 段 (0-33)
    discard_segment = obs['action_mask'][0:34]
    # 应该至少有一些牌可打出
    assert np.any(discard_segment > 0), "No tiles available to discard"

def test_random_strategy_with_new_mask():
    """测试 RandomStrategy 适配新格式"""
    strategy = RandomStrategy()
    env = WuhanMahjongEnv()
    obs, info = env.reset()

    action = strategy.choose_action(obs, obs['action_mask'])
    action_type, param = action

    assert 0 <= action_type < 11
    if action_type in [0, 3, 4, 5]:  # 需要参数的动作
        assert 0 <= param < 34
```

### 6.2 集成测试

```python
def test_full_game_with_new_action_mask():
    """测试完整游戏流程"""
    env = WuhanMahjongEnv()
    obs, info = env.reset()

    step_count = 0
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()

        # 验证 action_mask 形状
        assert obs['action_mask'].shape == (278,)

        # 验证至少有一个动作可用（非终端状态）
        if not terminated and not truncated:
            assert np.any(obs['action_mask'] > 0)

        # 使用 RandomStrategy
        strategy = RandomStrategy()
        action = strategy.choose_action(obs, obs['action_mask'])

        if not terminated:
            env.step(action)

        step_count += 1
        if step_count > 200:
            break
```

---

## 7. 迁移步骤

1. **修改观测空间定义** - 在 `example_mahjong_env.py` 中添加 action_mask
2. **重写 build_action_mask()** - 返回 278 位扁平化数组
3. **更新 RandomStrategy** - 适配新的 action_mask 格式
4. **运行测试验证** - 确保所有功能正常
5. **更新文档** - 记录新的 action_mask 格式

---

## 8. 设计优势

| 优势 | 说明 |
|------|------|
| **清晰分离** | 每个动作类型的参数有独立的段，不会混淆 |
| **神经网络友好** | 固定长度的一维数组，便于处理 |
| **易于扩展** | 添加新动作类型只需扩展数组长度 |
| **约束明确** | 红中杠等特殊杠的参数约束清晰可见 |
