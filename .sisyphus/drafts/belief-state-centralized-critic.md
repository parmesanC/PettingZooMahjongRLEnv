# Draft: Belief State & Centralized Critic Implementation

## User's Goal
实现完整的信念状态（Belief State）和完整的Centralized Critic，用于优化麻将MADRL系统。

## Current System Architecture

### Existing Network Components
**File**: `src/drl/network.py`

1. **ObservationEncoder** - Encodes observations into hidden representations
2. **HandEncoder** - Encodes hand tiles (34-dim per player)
3. **DiscardEncoder** - Encodes discard pool (34-dim)
4. **MeldEncoder** - Encodes melds (16 groups × 9 features)
5. **TransformerHistoryEncoder** - Encodes action history (80 steps)
6. **ActorCriticNetwork** - Current actor + critic network
   - Input: Single agent's observation
   - Output: action_type_logits, action_param_logits, value
   - Critic head: Linear(hidden_dim → 128 → 1)

### Current MAPPO Implementation
**File**: `src/drl/mappo.py`

- **Actor**: Parameter sharing (4 agents share same network)
- **Critic**: Currently only sees single agent's observation ( decentralized)
- Training: Uses GAE, clipped PPO objective
- Missing: True centralized critic that sees all agents' info

### Observation Space Structure
**File**: `src/mahjong_rl/observation/wuhan_7p4l_observation_builder.py`

```python
observation = {
    'global_hand': [4, 34],        # All 4 players' hands (masked in phase 3)
    'private_hand': [34],             # Current player's hand only
    'discard_pool_total': [34],         # Discarded tiles
    'wall': [82],                     # Remaining tiles in wall
    'melds': {                        # 16 meld groups
        'action_types': [16],
        'tiles': [256],                # 16 groups × 16 tiles × 34 one-hot
        'group_indices': [32]
    },
    'action_history': {                 # Last 80 actions
        'types': [80],
        'params': [80],
        'players': [80]
    },
    'special_gangs': [12],             # Special gang counts
    'current_player': [1],
    'fan_counts': [4],
    'special_indicators': [2],          # [lazy_tile, skin_tile]
    'remaining_tiles': scalar,
    'dealer': [1],
    'current_phase': int,
    'action_mask': [145]               # Valid actions
}
```

### Visibility Masking (Curriculum Learning)
**File**: `example_mahjong_env.py:683-771`

```python
def _apply_visibility_mask(self, observation, agent_id):
    """Three-phase curriculum learning"""
    if self.training_phase == 1:
        # Phase 1: Full observation (all info visible)
        pass
    elif self.training_phase == 2:
        # Phase 2: Progressive masking
        mask_prob = self._get_masking_probability()
        if np.random.random() < mask_prob:
            # Mask opponent hands, wall, concealed kongs
    elif self.training_phase == 3:
        # Phase 3: Real imperfect information
        # Mask all opponent hands and wall
```

### Available Global Information (GameContext)
**File**: `src/mahjong_rl/core/GameData.py`

```python
@dataclass
class GameContext:
    # Complete game state (not exposed to networks yet)
    wall: deque[int]              # Full wall (136 tiles total)
    discard_pile: List[int]          # All discarded tiles
    players: List[PlayerData]        # All 4 players' data
    current_player_idx: int
    lazy_tile: int                   # Wild card
    skin_tile: List[int]             # Skin tiles
    red_dragon: int
    action_history: List[ActionRecord]
    # ... more fields
```

## Key Challenges Identified

1. **Training-Execution Gap**:
   - Training: Centralized critic sees all 4 players' hands
   - Execution: Only local observation available
   - Solution: Curriculum learning (already exists), dual critics

2. **Imperfect Information**:
   - Opponent hands are hidden (34 unknown tiles × 3 opponents = 102 unknown tiles)
   - Belief state needs to estimate probability distribution
   - Requires probabilistic reasoning, not deterministic

3. **Information Asymmetry**:
   - Public info: discard pile, melds, action history
   - Private info: own hand, concealed kongs
   - Hidden info: opponent hands, wall ordering

## User Design Decisions (Confirmed)

1. **信念状态表示**: 蒙特卡罗采样
   - 从概率分布采样N个可能手牌
   - 平均处理采样结果
   - 计算密集但最精确

2. **Centralized Critic输入**: 完整全局状态（激进方案）
   - 所有4玩家完整手牌
   - 牌墙完整信息
   - 公共信息（弃牌、副露、动作历史）

3. **训练策略**: 两者结合（最佳）
   - Phase 1（全知）: Centralized Critic
   - Phase 2（渐进）: Centralized Critic
   - Phase 3（真实）: Decentralized Critic

4. **信念集成**: 仅Actor输入
   - 信念状态用于动作决策
   - Critic不使用信念（直接用真实状态）

## Architecture Design Summary

```
训练时 (Phase 1-2):
┌─────────────────────────────────────────────────────────────┐
│  公共信息 (弃牌、副露、历史)                  │
│           ↓                                      │
│  信念网络: [公共信息] → [对手1-3概率]         │
│                              ↓                    │
│  蒙特卡罗采样器: 采样N个可能手牌              │
│                              ↓                    │
│  Actor: [私有手牌, 采样对手, 公共信息] → 动作   │
│                              ↓                    │
│  Centralized Critic: [完整全局状态] → V值         │
└─────────────────────────────────────────────────────────────┘

执行时 (Phase 3):
┌─────────────────────────────────────────────────────────────┐
│  信念网络: [公共信息] → [对手1-3概率]         │
│                              ↓                    │
│  蒙特卡罗采样器: 采样N个可能手牌              │
│                              ↓                    │
│  Actor: [私有手牌, 采样对手, 公共信息] → 动作   │
│                              ↓                    │
│  Decentralized Critic: [局部观测] → V值          │
└─────────────────────────────────────────────────────────────┘
```

## Key Components to Implement

1. **BeliefNetwork** (新建)
   - 输入: 公共信息 (discard_pool, melds, action_history)
   - 输出: 3个对手的概率分布 (各34维)
   - 架构: Transformer + Linear heads

2. **MonteCarloSampler** (新建)
   - 输入: 对手概率分布
   - 输出: N个采样的完整状态
   - 参数: sample_count (默认5-10)

3. **CentralizedCriticNetwork** (新建)
   - 输入: 完整全局状态 (4玩家手牌 + 牌墙 + 公共信息)
   - 输出: 标量V值
   - 架构: Large Transformer + Linear

4. **Modified ActorNetwork** (修改现有)
   - 输入: [私有手牌, 采样对手手牌, 公共信息]
   - 改动: 集成蒙特卡罗采样
   - 输出: action_type, action_param

5. **DualCriticTraining** (修改MAPPO)
   - Phase 1-2: 使用CentralizedCritic
   - Phase 3: 使用现有DecentralizedCritic
   - 切换逻辑: 基于training_phase

## Integration Points

1. **Wuhan7P4LObservationBuilder**:
   - 添加构建完整全局状态的方法
   - 确保Phase 1-2返回完整信息

2. **MAPPO训练循环**:
   - 修改update()方法支持dual-critic
   - 根据phase选择critic
   - 处理不同critic的价值输出

3. **WuhanMahjongEnv**:
   - 保持现有3阶段课程学习
   - Phase 1-2: 提供完整全局状态给critic
   - Phase 3: 仅提供局部观测

## Estimated Complexity

- **新增代码行数**: ~2000-3000行
- **网络参数量**:
  - BeliefNetwork: ~2M
  - MonteCarloSampler: 0 (无参数)
  - CentralizedCritic: ~5M
  - Modified Actor: 不变 (~3M)
- **训练时间增加**: ~3-5x (蒙特卡罗采样 + dual-critic)
- **内存需求**: ~2-3x (存储多个采样状态)

## Additional Design Considerations

###信念状态更新机制
- **贝叶斯更新**: 每次公开信息（弃牌、副露）后更新对手手牌概率
- **序列建模**: 使用LSTM/GRU/Transformer跟踪信念状态随时间的演变
- **对抗推理**: 考虑对手策略类型（激进/保守）调整信念

### Centralized Critic架构选择
- **MAPPO风格**: Critic输入(concatenated observations) → 输出(4个Q值，每个agent一个)
- **QMIX风格**: Value decomposition - 分层神经网络（hypernetwork + agent nets）
- **MADDPG风格**: Centralized Q-function - 输入(state, actions_1..4) → 标量Q值

### 信息流设计
```
训练时:
- 信念网络: 公共信息 → 对手手牌概率
- Actor: [私有手牌, 信念, 公共信息] → 动作
- Centralized Critic: [完整全局状态] → V值

执行时:
- 信念网络: 公共信息 → 对手手牌概率
- Actor: [私有手牌, 信念, 公共信息] → 动作
- Decentralized Critic: [私有观测] → V值 (或使用pre-trained centralized)
```

### 关键挑战
1. **状态空间爆炸**: 完整全局状态 > 1000维 vs 局部观测 ~500维
2. **信念准确性**: 初期信念估计非常粗略，随游戏进行逐渐精确
3. **训练稳定性**: Centralized critic可能导致策略过度依赖全局信息
4. **计算开销**: 信念估计和centralized critic都增加显著计算量

## Questions for User

**Critical Design Decisions (需确认)**:

1. **信念状态表示方式**
   - 选项A: 概率分布（每个对手34维，总和=1）- 简单但信息有限
   - 选项B: 多模态分布（每个牌ID：[概率, 置信度]）- 更丰富
   - 选项C: 采样表示（蒙特卡罗采样N个可能手牌）- 计算密集

2. **Centralized Critic输入**
   - 选项A: 完整全局状态（4玩家手牌 + 牌墙 + 弃牌）- 最强但有训练-执行差距
   - 选项B: 聚合表示（每个玩家编码后concatenate）- 更平衡
   - 选项C: 分层Critic（local critic + global critic混合）- 最佳实践

3. **训练策略**
   - 选项A: 保留3阶段课程学习（Phase 1全知 → Phase 2渐进 → Phase 3真实）- 已有基础
   - 选项B: Dual-Critic（训练时用centralized，执行时用decentralized）- 新增工作
   - 选项C: 结合A和B - 最全面但最复杂

4. **信念状态集成点**
   - 选项A: 仅Actor输入（帮助决策）
   - 选项B: 仅Critic输入（帮助价值估计）
   - 选项C: 两者都集成 - 推荐
1. Belief representation: Should belief state be probability distribution (34-dim per opponent) or more complex representation?
2. Centralized critic input: Should critic see full observation (all 4 hands) or aggregated representations?
3. Training strategy: Keep 3-phase curriculum or add dual-critic (centralized training, decentralized execution)?
4. Integration: Should belief state be part of actor input, critic input, or both?
