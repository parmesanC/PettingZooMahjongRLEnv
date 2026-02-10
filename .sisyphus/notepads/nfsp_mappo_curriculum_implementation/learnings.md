## [2025-02-09] CentralizedCriticNetwork实现完成

### 问题描述
任务31的第一个子任务：实现CentralizedCriticNetwork类，使其能够看到所有4个agents的观测并输出每个agent的价值估计。

### 实现方案

**网络架构**：
```python
class CentralizedCriticNetwork(nn.Module):
    - 4个独立的ObservationEncoder（每个agent一个，hidden_dim=256）
    - 融合层：Linear(1024 -> 1024 -> 512)
    - 4个独立的价值头（每个agent一个，输出[batch, 1]）
    - 最终输出：stack + squeeze -> [batch, 4]
```

### 关键设计决策

1. **Agent Encoders**: 使用4个独立的ObservationEncoder
   - 决策：独立编码器（更多参数，更灵活）
   - 理由：每个agent可以有独特的编码，捕获agent-specific模式

2. **Fusion Strategy**: Concatenate所有agent特征后通过融合层
   - 架构：先stack [batch, 4, 256] -> view [batch, 1024]
   - 融合：Linear(1024 -> 1024 -> 512)
   - 理由：处理更高维的输入，捕获跨agent信息

3. **Value Heads**: 4个独立的value head
   - 决策：每个agent有自己的价值头
   - 理由：PPO需要每个agent的独立价值估计
   - 输出形状：[batch, 4]

4. **网络尺寸**: hidden_dim=512（比decentralized的256更大）
   - 决策：需要处理4x更多的数据
   - 理由：centralized critic的输入维度更高

### 遇到的问题和解决方案

1. **Import错误**: `List` 未从typing导入
   - 解决方案：添加`List`到typing导入

2. **形状不匹配**: RuntimeError "mat1 and mat2 shapes cannot be multiplied"
   - 原因：stacking后形状为[batch, 4, 256]但view后变成[batch, 1024]
   - 解决方案：先stack到[batch, 4, 256]再view到[batch, 1024]

3. **输出形状错误**: 输出[4, 2]而不是[2, 4]
   - 原因：`torch.stack(values, dim=0)`在第一个维度stack，batch维度移动
   - 第一次尝试：transpose(0, 1) -> 结果[2, 4, 1]仍有多余维度
   - 解决方案：`torch.stack(values, dim=1).squeeze(-1)` -> stack在value维度，squeeze移除多余维度

### 验证结果

```bash
[OK] Forward pass successful, output shape: torch.Size([2, 4])
[OK] All tests passed
```

### 修改的文件

- `src/drl/network.py`:
  - 添加`List`到typing导入
  - 添加`CentralizedCriticNetwork`类（约70行代码）
  - 类位于AveragePolicyNetwork之后，init_weights函数之前

### 网络参数量估计

- 每个 ObservationEncoder（hidden_dim=256）: ~2M 参数
- 4个独立编码器: ~8M 参数
- 融合层: ~0.5M 参数
- 4个value head: ~0.3M 参数
- **总计**: ~8.8M 参数（比decentralized critic的~3M大很多）

### 下一步

- 子任务31-2: 创建CentralizedRolloutBuffer类（src/drl/buffer.py）
- 需要能够存储所有agents的观测、动作、奖励
