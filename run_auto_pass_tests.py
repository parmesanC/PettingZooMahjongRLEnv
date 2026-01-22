"""
运行 Auto-Pass 优化测试的简单脚本
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 导入测试模块
from tests.test_auto_pass_optimization import run_all_tests

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
