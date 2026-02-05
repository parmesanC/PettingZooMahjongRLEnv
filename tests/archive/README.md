# 归档测试文件

此目录包含已归档的临时测试、调试脚本和特定任务验证文件。

## 归档原因

这些文件被归档是因为它们：
- 是用于特定bug修复的临时测试
- 是诊断脚本，用于调试特定问题
- 是任务验证脚本，用于验证特定功能的实现
- 已被更完整的测试替代

## 注意

**不建议直接运行这些测试**，因为它们可能：
- 包含过时的代码
- 引用已修改的API
- 产生误导性的结果

如需使用，请先检查代码是否仍然有效。

## 文件列表

| 文件名 | 归档原因 |
|--------|----------|
| test_fix.py | 临时测试脚本 - 用于验证状态机修复 |
| test_simple_3.py | 简单的Hello World测试 |
| test_reset_fix.py | 验证初始化重构的测试脚本 |
| test_kong_lazy_debug.py | 重现 KONG_LAZY 错误的调试脚本 |
| test_obs_builder_validator.py | 检查 ObservationBuilder 初始化的诊断脚本 |
| test_action_validator_init.py | 检查 ActionValidator 初始化时机的诊断脚本 |
| test_tiles_output.py | 诊断 Tiles 输出问题的脚本 |
| test_server_simple.py | 快速测试FastAPI服务器的简化版 |
| simple_test.py | 简单的导入测试 |
| quick_test.py | agent_iter() 的快速测试 |
| test_task3_implementation.py | Task 3 实现验证 |
| test_task3_logic.py | Task 3 逻辑验证 |

## 归档日期

2025年1月27日
