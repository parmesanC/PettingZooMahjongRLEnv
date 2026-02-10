# NFSP/MAPPO 训练器实现 - 进度总结

## 完成的工作

### 1. 训练器修复 (`src/drl/trainer.py`)

#### 1.1 重写 `_run_episode` 方法
- 实现了 PettingZoo 标准模式
- 使用 `agent_iter()` 和 `last()` 正确循环
- 支持对手切换（前期随机 → 后期历史策略）
- 返回正确的回合统计信息

#### 1.2 添加 `_evaluate` 方法
- 评估 NFSP agent vs 随机对手
- 使用 PettingZoo 标准模式
- Agent 0 使用 NFSP，agents 1-3 使用随机对手
- 返回 win_rate, avg_reward, wins, games

#### 1.3 更新训练循环
- 使用 `actual_total_episodes` 而不是 `total_episodes`
- 使用 `actual_save_interval` 而不是 `save_interval`
- 添加详细的训练信息输出

#### 1.4 集成课程学习调度器
- 在 `__init__` 中创建 `CurriculumScheduler` 实例
- 在训练循环中获取当前阶段和进度
- 更新环境的 `training_phase` 和 `training_progress`
- 在 `_run_episode` 返回值中包含课程学习信息
- 在 `_print_progress` 中显示当前课程学习阶段
- 在 `_save_checkpoint` 中保存课程学习元数据

### 2. 配置系统更新 (`src/drl/config.py`)

#### 2.1 添加训练模式支持
- `quick_test`: 快速测试模式（10,000 局）
- `full_training`: 完整训练模式（20,000,000 局）

#### 2.2 检查点保存配置
- 快速测试：每 100 局保存一次
- 完整训练：每 1,000 局保存一次（符合用户需求）
- 评估间隔：每 1,000 局评估一次

#### 2.3 新增属性
- `actual_total_episodes`: 根据模式返回总局数
- `actual_save_interval`: 根据模式返回保存间隔

#### 2.4 更新 `get_quick_test_config()` 函数
- 设置模式为 `quick_test`
- 更新 `quick_test_episodes` 为 10,000
- 更新 `save_interval_quick_test` 为 100

### 3. 课程学习调度器修复 (`src/drl/curriculum.py`)

#### 3.1 修复进度计算
- 阶段1（全知）：progress 固定为 0.0（无掩码）
- 阶段2（渐进）：progress 从 0.0 到 1.0 变化
- 阶段3（真实）：progress 固定为 1.0（完全掩码）

### 4. 训练脚本更新 (`train_nfsp.py`)

#### 4.1 更新命令行参数
- 添加 `--quick-episodes` 参数
- 添加 `--full-episodes` 参数
- 移除过时的 `--episodes` 参数

#### 4.2 更新配置输出
- 显示训练模式
- 显示实际总局数和保存间隔
- 显示课程学习信息

## 验证结果

### ✅ 语法检查
```bash
python -m py_compile src/drl/trainer.py
python -m py_compile src/drl/config.py
python -m py_compile src/drl/curriculum.py
python -m py_compile train_nfsp.py
```
- 所有文件语法正确

### ✅ 配置测试
```bash
from src.drl.config import get_quick_test_config, get_default_config

# Quick test config
Mode: quick_test
Quick test episodes: 10,000
Actual total episodes: 10,000
Save interval: 100
Eval interval: 100

# Default config
Mode: full_training
Full training episodes: 20,000,000
Actual total episodes: 20,000,000
Save interval: 1,000
Eval interval: 1,000
```

### ✅ 课程学习测试
```bash
# Quick Test (10k episodes)
Phase 1: 0-3,330 episodes (progress: 0.00%)
Phase 2: 3,330-6,660 episodes (progress: 0.0% → 100%)
Phase 3: 6,660-10,000 episodes (progress: 100%)

# Full Training (20M episodes)
Phase 1: 0-6,660,000 episodes (progress: 0.00%)
Phase 2: 6,660,000-13,320,000 episodes (progress: 0.0% → 100%)
Phase 3: 13,320,000-20,000,000 episodes (progress: 100%)
```

### ✅ 导入测试
```bash
from src.drl.trainer import NFSPTrainer
from src.drl.config import get_quick_test_config
from src.drl.curriculum import CurriculumScheduler
```
- 所有模块成功导入

### ✅ 命令行脚本测试
```bash
python train_nfsp.py --help
```
- 所有命令行参数正确显示
- 帮助信息格式正确

## 待完成任务

### 集成测试
- [ ] 运行完整训练循环测试（小规模）
- [ ] 验证环境创建和初始化
- [ ] 验证检查点保存和加载
- [ ] 验证评估功能

### 性能测试
- [ ] 测试训练速度
- [ ] 监控内存使用
- [ ] 验证 GPU 使用情况

## 技术细节

### PettingZoo 标准模式
```python
for agent_name in self.env.agent_iter():
    obs, reward, terminated, truncated, info = self.env.last()
    agent_idx = int(agent_name.split('_')[1])
    action_mask = obs['action_mask']

    # 选择动作
    if agent_idx == 0:
        agent = self.agent_pool.get_agent(0)
        action_type, action_param = agent.choose_action(obs, action_mask)
    else:
        action_type, action_param = self.random_opponent.choose_action(
            obs, action_mask
        )

    self.env.step((action_type, action_param))
```

### 课程学习集成
```python
# 获取当前阶段和进度
phase, progress = self.curriculum.get_phase(self.episode_count)
self.current_phase = phase
self.current_progress = progress

# 更新环境训练阶段
self.env.training_phase = phase
self.env.training_progress = progress

# 返回回合统计
episode_stats = {
    'rewards': episode_rewards,
    'steps': episode_steps,
    'winner': winner,
    'curriculum_phase': self.current_phase,
    'curriculum_progress': self.current_progress
}
```

### 检查点格式
```python
checkpoint.pth: 模型权重
checkpoint_metadata.json: 元数据
{
    'episode': 1000,
    'phase': 1,
    'progress': 0.0,
    'mode': 'quick_test',
    'total_episodes': 10000,
    'timestamp': 1234567890.0
}
```

### 配置文件结构
```python
@dataclass
class TrainingConfig:
    mode: str = 'full_training'
    quick_test_episodes: int = 100_000
    full_training_episodes: int = 20_000_000
    switch_point: int = 1_000_000
    eval_interval: int = 1000
    eval_games: int = 100
    save_interval_quick_test: int = 100
    save_interval_full_training: int = 1000
    # ...

    @property
    def actual_total_episodes(self) -> int:
        if self.mode == 'quick_test':
            return self.quick_test_episodes
        else:
            return self.full_training_episodes

    @property
    def actual_save_interval(self) -> int:
        if self.mode == 'quick_test':
            return self.save_interval_quick_test
        else:
            return self.save_interval_full_training
```

## 已知问题

### LSP 错误（不影响运行）
1. Import "torch" could not be resolved - LSP 配置问题
2. "split" is not a known attribute of "None" - PettingZoo 类型推断问题
3. Type "None" is not assignable to declared type - dataclass 类型注解问题

这些是 LSP 的静态类型检查错误，不影响 Python 代码的实际运行。

## 下一步建议

1. 创建小规模测试脚本验证完整训练流程
2. 测试检查点保存和加载功能
3. 运行快速测试模式验证所有功能
4. 准备完整的训练运行（如果测试通过）

## 使用说明

### 快速测试（10,000 局）
```bash
python train_nfsp.py --quick-test
```

### 完整训练（20,000,000 局）
```bash
python train_nfsp.py
```

### 自定义配置
```bash
# 自定义快速测试局数
python train_nfsp.py --quick-test --quick-episodes 50000

# 自定义完整训练局数
python train_nfsp.py --full-episodes 10000000

# 使用 CPU
python train_nfsp.py --device cpu

# 自定义网络结构
python train_nfsp.py --hidden-dim 128 --transformer-layers 2
```

### 检查点保存位置
- 模型文件：`checkpoints/checkpoint_{episode}.pth`
- 元数据文件：`checkpoints/checkpoint_{episode}_metadata.json`
- 最终模型：`checkpoints/final_model.pth`
- 日志文件：`logs/training_log.jsonl`

## 课程学习阶段说明

### 阶段1：全知视角（Phase 1）
- **Episode范围**：0 - 33.3% 总局数
- **Progress**：固定 0.0%
- **含义**：可以看到所有玩家手牌、牌墙信息
- **目的**：让智能体快速学习游戏规则和基本策略

### 阶段2：渐进式掩码（Phase 2）
- **Episode范围**：33.3% - 66.6% 总局数
- **Progress**：0.0% → 100%（线性增长）
- **含义**：逐渐隐藏对手手牌信息
- **目的**：平滑过渡到真实环境

### 阶段3：真实环境（Phase 3）
- **Episode范围**：66.6% - 100% 总局数
- **Progress**：固定 100%
- **含义**：只能看到自己的手牌和公共信息
- **目的**：在真实环境中提高策略鲁棒性
