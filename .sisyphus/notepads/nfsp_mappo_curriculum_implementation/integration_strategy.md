# CentralizedCritic 集成修复 - 精确实施指南

**日期**: 2025-02-09
**问题**: MAPPO 的 CentralizedCritic 未被实际使用
**目标**: 修复该问题，实现 phase-aware dual-critic 训练

---

## 📋 修改清单（按优先级）

### 🔴 P0: 核心问题修复

#### 1. src/drl/agent.py - NFSPAgentPool 添加全局观测收集

**文件**: `D:\DATA\Python_Project\Code\PettingZooRLENVMahjong\src\drl\agent.py`

**当前代码**（约第180-220行）：
```python
class NFSPAgentPool:
    def store_transition(self, obs, action_type, action_param, ...):
        # ... 现有代码存储单个 agent 的观测
```

**需要的修改**：
在 `store_transition()` 方法后添加新方法：

```python
def store_global_observation(self, all_agents_observations, episode_info):
    """
    存储所有智能体的全局观测
    
    Args:
        all_agents_observations: Dict[str, Dict] - agent_name -> observation
        episode_info: Dict - 当前回合信息
    """
    # 暂存到 buffer 或新字典
    self._global_observations[episode_info['episode_num']] = all_agents_observations
```

**验证方法**：
```python
def get_global_observations(self, episode_num):
    """获取指定回合的所有智能体观测"""
    return self._global_observations.get(episode_num, {})
```

---

#### 2. src/drl/agent.py - NFSPAgent 在选择动作时收集全局观测

**修改位置**: `NFSPAgent.select_action()` 方法

**添加到方法中**：
```python
# 在返回 action 之前
if hasattr(self.agent_pool, 'get_global_observations'):
    # 获取当前回合的全局观测（从上一个动作时存储）
    current_global_obs = self.agent_pool.get_global_observations(self.episode_num)
    if current_global_obs:
        # 将全局观测附加到 obs 字典中
        obs['all_agents_observations'] = current_global_obs
```

---

#### 3. src/drl/buffer.py - CentralizedRolloutBuffer 完善 get_centralized_batch

**文件**: `D:\DATA\Python_Project\Code\PettingZooRLENVMahjong\src\drl\buffer.py`

**当前代码**（第567-667行）：
```python
def get_centralized_batch(self, batch_size: int, device: str = 'cuda'):
    # ... 现有基本实现，但需要确保数据格式正确
```

**检查点**：
1. 确保 `all_observations` 存储为 List[List[Dict]] 格式
2. 每个 observation 应该包含完整的 agent 信息

---

#### 4. src/drl/mappo.py - MAPPO 添加 centralized_critic 参数

**文件**: `D:\DATA\Python_Project\Code\PettingZooRLENVMahjong\src\drl\mappo.py`

**修改位置 1**: `__init__` 方法参数列表（第25-38行）

**当前代码**：
```python
def __init__(
    self,
    network,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    ppo_epochs: int = 4,
    device: str = 'cuda'
):
```

**需要修改为**：
```python
def __init__(
    self,
    network,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    ppo_epochs: int = 4,
    device: str = 'cuda',
    centralized_critic=None  # NEW: 添加 centralized_critic 支持
):
```

**修改位置 2**: `__init__` 方法体中（第55-60行）

**当前代码**：
```python
self.network = network
self.device = device

# 超参数
self.gamma = gamma
...
```

**需要修改为**：
```python
self.network = network
self.device = device
self.centralized_critic = centralized_critic  # NEW: 添加 centralized_critic 支持

# 超参数
self.gamma = gamma
...
```

---

#### 5. src/drl/mappo.py - MAPPO 添加 update_centralized() 方法

**添加位置**: `get_training_stats()` 方法之后（第319行左右）

**新方法**：
```python
def update_centralized(
    self,
    all_observations,  # List[List[Dict]] - 所有智能体的完整观测序列
    all_actions_type,
    all_actions_param,
    all_rewards,
    training_phase  # 1=全知，2=渐进，3=真实
):
    """
    使用 centralized critic 进行训练（Phase 1-2: 使用完整全局状态）
    
    Returns:
        训练统计字典
    """
    # Phase 3 或没有 centralized_critic，使用现有的 decentralized 方法
    if training_phase not in [1, 2] or self.centralized_critic is None:
        return self.update(buffer)
    
    # Phase 1-2: 使用 centralized critic
    with torch.no_grad():
        # 准备批次数据
        # 这里的实现可以简化，重点在于调用 centralized_critic
        
        # 调用 centralized_critic
        # 假设 all_observations 格式正确
        values = self.centralized_critic(all_observations)
        
        # 计算优势和损失（可以复用现有逻辑）
        # ...
    
    return {
        'loss': avg_loss,
        'training_step': self.training_step,
        'used_centralized': True
    }
```

**简化建议**：
- 如果一开始实现完整版太复杂，可以先实现一个简化版本
- 简化版本只在 `update()` 中添加 phase-aware 切换
- 简化版本不使用 `update_centralized()`，直接在 `update()` 中根据 phase 选择 critic

---

#### 6. src/drl/trainer.py - NFSPTrainer 传递全局观测

**文件**: `D:\DATA\Python_Project\Code\PettingZooRLENVMahjong\src\drl\trainer.py`

**修改位置**: `_run_episode()` 方法，全局观测收集后（第244行左右）

**当前代码**：
```python
# [临时] 收集全局观测（用于诊断）
all_agents_observations = {}

for agent_name in self.env.agent_iter():
    obs, reward, terminated, truncated, info = self.env.last()
    agent_idx = int(agent_name.split('_')[1])
    
    # [临时] 收集全局观测（用于诊断）
    all_agents_observations[agent_name] = obs
```

**需要的修改**：
```python
# 在 episode 结束前
episode_stats['all_agents_observations'] = all_agents_observations

# 传递给 agent_pool 存储
self.agent_pool.store_global_observation(
    all_agents_observations=all_agents_observations,
    episode_info={'episode_num': self.episode_count}
)
```

**另一个修改位置**: `train_agent_pool()` 方法

**需要在调用 `agent_pool.train_all()` 时添加**：
```python
# 当前实现
train_stats = self.agent_pool.train_all(
    training_phase=self.current_phase
)

# 需要添加参数
train_stats = self.agent_pool.train_all(
    training_phase=self.current_phase,
    global_observations=all_agents_observations  # NEW
)
```

---

## 🚀 实施策略（避免技术问题）

### 策略 A：渐进式修改（推荐）

由于之前的编辑和委托遇到技术问题，建议采用渐进式修改：

**第 1 步**：修改 `agent.py` 添加全局观测存储
- 只修改 `agent.py`
- 不修改其他文件
- 验证修改后运行简单测试

**第 2 步**：修改 `mappo.py` 添加 centralized_critic 参数
- 只添加参数，不改变训练逻辑
- 验证 MAPPO 可以正常初始化

**第 3 步**：修改 `mappo.py` 添加简化的 phase-aware 切换
- 在 `update()` 中根据 training_phase 选择使用哪个 critic
- 不立即实现完整的 `update_centralized()` 方法

**第 4 步**：修改 `trainer.py` 传递全局观测
- 修改 `agent_pool.train_all()` 调用

**第 5 步**：完整集成测试
- 运行完整的训练流程验证

---

### 策略 B：简化实现（备选）

如果渐进式修改仍然遇到问题，可以采用更简化的实现：

**简化方案**：只在 `mappo.update()` 中添加 phase-aware 逻辑

```python
def update(self, buffer, next_obs=None, next_action_mask=None, training_phase=1):
    """
    根据 training_phase 选择使用哪个 critic
    """
    # 保存原始行为（Phase 3）
    original_update = super().update(buffer, next_obs, next_action_mask)
    
    # Phase 1-2: 使用 centralized critic
    if training_phase in [1, 2] and self.centralized_critic is not None:
        # 这里实现 centralized critic 训练逻辑
        # 可以简化为直接调用 self.centralized_critic(all_observations)
        pass
    
    # Phase 3: 使用 decentralized critic
    return original_update
```

---

## ✅ 验证步骤

每完成一个修改后，运行以下验证：

### 验证 1: 修改 agent.py 后
```bash
cd /d/DATA/Python_Project/Code/PettingZooRLENVMahjong
"D:\DATA\Development\Anaconda\condabin\conda.bat" activate PettingZooRLMahjong
python -c "import sys; sys.path.insert(0, 'src/drl'); from agent import NFSPAgentPool; print('NFSPAgentPool loaded successfully')"
```

### 验证 2: 修改 mappo.py 后
```bash
cd /d/DATA/Python_Project/Code/PettingZooRLENVMahjong
"D:\DATA\Development\Anaconda\condabin\conda.bat" activate PettingZooRLMahjong
python -c "from mappo import MAPPO; from network import CentralizedCriticNetwork; print('Modules imported successfully')"
```

### 验证 3: 修改 trainer.py 后
```bash
cd /d/DATA/Python_Project/Code/PettingZooRLENVMahjong
"D:\DATA\Development\Anaconda\condabin\conda.bat" activate PettingZooRLMahjong
python -c "from trainer import NFSPTrainer; print('Trainer module structure check')"
```

---

## 📝 记录到 notepad

在修改过程中记录发现和遇到的问题。

---

## 🎯 总结

**核心问题**：CentralizedCritic 已存在但未在训练中使用

**解决方案**：6 个具体修改点，按优先级执行

**推荐策略**：渐进式修改，每步验证后继续

汪呜呜，这是一个清晰的、可执行的计划。你可以按照这个指南逐步实施，遇到任何问题随时告诉我！
