"""
场景测试框架 - 数据结构定义

包含测试场景、步骤配置和测试结果的数据结构。
"""

from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any
from src.mahjong_rl.core.constants import GameStateType, ActionType


@dataclass
class StepConfig:
    """单个测试步骤配置"""
    step_number: int
    description: str

    # 动作配置
    is_action: bool = True
    player: Optional[int] = None
    action_type: Optional[ActionType] = None
    parameter: int = -1
    is_auto: bool = False

    # 验证配置
    expect_state: Optional[GameStateType] = None
    expect_action_mask_contains: Optional[List[ActionType]] = None
    validators: List[Callable] = field(default_factory=list)

    # 快捷验证
    verify_hand_tiles: Optional[Dict[int, List[int]]] = None
    verify_wall_count: Optional[int] = None
    verify_discard_pile_contains: Optional[List[int]] = None


@dataclass
class ScenarioContext:
    """测试场景上下文"""
    name: str
    description: str = ""

    # 游戏初始配置
    wall: List[int] = field(default_factory=list)
    special_tiles: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None

    # 自定义初始状态配置（绕过 InitialState）
    initial_config: Optional[Dict[str, Any]] = None

    # 步骤配置
    steps: List[StepConfig] = field(default_factory=list)

    # 终止验证（游戏结束时）
    final_validators: List[Callable] = field(default_factory=list)
    expect_winner: Optional[List[int]] = None


@dataclass
class TestResult:
    """测试执行结果"""
    scenario_name: str
    success: bool
    failed_step: Optional[int] = None
    failure_message: Optional[str] = None

    # 执行信息
    total_steps: int = 0
    executed_steps: int = 0

    # 最终状态快照（用于调试）
    final_state: Optional[GameStateType] = None
    final_context_snapshot: Optional[Dict] = None
