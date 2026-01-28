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
