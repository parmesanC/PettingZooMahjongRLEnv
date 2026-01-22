from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from mahjong_rl.core.constants import ActionType


class WuhanMahjongEnv(AECEnv):
    """武汉麻将七皮四赖子环境 - 基于实际规则优化"""

    metadata = {'render_modes': ['human', 'ansi'], 'name': 'wuhan_mahjong_v0.1'}

    # 牌编码：0-26: 万/筒/条 1-9（0=1万, 8=9万, 9=1筒...）
    # 27-33: 东南西北中发白（27=东, 28=南, 29=西, 30=北, 31=红中, 32=发财, 33=白板）
    TILE_COUNT = 34
    HAND_SIZE = 13
    # 使用动作掩码对无效动作进行遮蔽
    # 动作空间设计为参数化动作，
    ACTION_SPACE = len(ActionType)

    def __init__(self, render_mode=None, training_phase=3):
        super().__init__()

        self.training_phase = training_phase
        self.num_players = 4
        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        self.agents = self.possible_agents[:]
        self.agents_name_mapping = {name: i for i, name in enumerate(self.agents)}
        self.observation_spaces = self._create_observation_spaces()
        self.action_spaces = {agent: spaces.Discrete(self.ACTION_SPACE) for agent in self.possible_agents}
        # 定义动作类型常量
        self.ACTION_TYPES = {
            'DISCARD': 0,  # 0-33: 打牌
            'LEFT_CHOW': 34,  # 左吃
            'MID_CHOW': 35,  # 中吃
            'RIGHT_CHOW': 36,  # 右吃
            'PONG': 37,  # 碰
            'DIRECT_KONG': 38,  # 冲杠
            'RESERVE_KONG': 39,  # 蓄杠
            'CONCEALED_KONG': 40,  # 暗杠
            'SPECIAL_KONG': 41,  # 特殊杠基值，41-43: 红中杠/赖子杠/皮子杠
        }

        # 初始化状态变量
        self.hands = {agent: [0] * self.TILE_COUNT for agent in self.possible_agents}
        self.melds = {agent: [] for agent in self.possible_agents}
        self.discard_pool = []
        self.wall = []
        self.current_player = None
        self.dealer = None
        self.special_tiles = {'lai_tile': None, 'pi_tile': None}
        self.special_gangs = {agent: {'pi_gang': 0, 'lai_gang': 0, 'zhong_gang': 0} for agent in self.possible_agents}
        self.fan_counts = {agent: 0 for agent in self.possible_agents}
        self.action_history = []
        self.valid_actions = []

        self.render_mode = render_mode

    def _create_observation_spaces(self) -> Dict[str, spaces.Dict]:
        """为每个玩家创建观测空间 - 基于实际规则优化"""
        observation_spaces = {}

        for agent in self.possible_agents:
            observation_spaces[agent] = spaces.Dict({
                # 手牌信息 - 使用MultiDiscrete（每人手牌中每张牌有几张）
                'global_hand': spaces.MultiDiscrete([6] * (4 * 34)),  # 4玩家×34种牌，每种牌有6个状态：0-4表示牌数，5表示不可见
                'private_hand': spaces.MultiDiscrete([6] * 34),  # 私有手牌

                # 牌河总量 - 使用MultiDiscrete（每种牌0-4张）
                'discard_pool_total': spaces.MultiDiscrete([6] * 34),

                # 牌墙信息 - 使用MultiDiscrete（每种牌0-4张）
                'wall': spaces.MultiDiscrete([35] * 82),

                # 副露信息 - 按玩家分组
                'melds': spaces.Dict({
                    'action_types': spaces.MultiDiscrete([11] * 4 * 4),  # 4玩家×4副露
                    'tiles': spaces.MultiDiscrete([35] * 4 * 4 * 4),  # 4玩家×4副露×4张牌×34种牌
                    # 添加分组索引以便网络关联
                    'group_indices': spaces.MultiDiscrete([4, 4] * 16)  # [玩家ID, 副露ID] × 16
                }),

                # 动作历史 - 类似结构
                'action_history': spaces.Dict({
                    'types': spaces.MultiDiscrete([11] * 80),
                    'params': spaces.MultiDiscrete([35] * 80),
                    'players': spaces.MultiDiscrete([4] * 80)
                }),

                # 特殊杠状态 - 使用MultiDiscrete（基于实际牌张数限制）
                'special_gangs': spaces.MultiDiscrete([8, 4, 5] * 4),  # 4玩家×[0-7张皮子;0-3张赖子;0-4张红中]

                # 当前玩家 - 使用MultiBinary（one-hot）
                'current_player': spaces.MultiDiscrete([4]),

                # 可见番数计数 - 使用MultiDiscrete（1-600番）
                'fan_counts': spaces.MultiDiscrete([600] * 4),  # 1-600

                # 特殊牌指示 - 使用MultiDiscrete
                'special_indicators': spaces.MultiDiscrete([34, 34]),  # [赖子ID, 皮子ID]

                # 剩余牌数 - 使用Discrete（0-136张）
                'remaining_tiles': spaces.Discrete(137),

                # 庄家信息 - 使用MultiBinary（one-hot）与current_player对齐
                'dealer': spaces.MultiDiscrete([4]),

                # 动作掩码 - 使用MultiBinary（0无效，1有效）
                'action_mask': spaces.Dict({
                    'types': spaces.MultiBinary(11),
                    'params': spaces.MultiBinary(35),
                }),

                # 当前阶段上下文（Agent决策关键）
                'current_phase': spaces.Discrete(8),  # 0:Draw, 1:Play... 映射8个主状态

            })

        return observation_spaces

    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        """为指定玩家生成观测数据"""
        observation = {}
        agent_idx = self.possible_agents.index(agent)

        # 1. 手牌信息
        observation['global_hand'] = self._get_global_hand_observation(agent_idx)
        observation['private_hand'] = self._get_private_hand_observation(agent_idx)

        # 2. 牌河总量
        observation['discard_pool_total'] = self._get_discard_pool_observation()

        # 3. 牌墙信息
        observation['wall'] = self._get_wall_observation()

        # 4. 副露信息
        observation['melds'] = self._get_melds_observation()

        # 5. 动作历史
        observation['action_history'] = self._get_action_history_observation()

        # 6. 特殊杠状态
        observation['special_gangs'] = self._get_special_gangs_observation()

        # 7. 当前玩家
        observation['current_player'] = self._get_current_player_observation()

        # 8. 可见番数计数
        observation['fan_counts'] = self._get_fan_counts_observation()

        # 9. 特殊牌指示
        observation['special_indicators'] = self._get_special_indicators_observation()

        # 10. 剩余牌数
        observation['remaining_tiles'] = self._get_remaining_tiles_observation()

        # 11. 庄家信息
        observation['dealer'] = self._get_dealer_observation()

        # 12. 动作掩码
        observation['action_mask'] = self._get_action_mask_observation(agent_idx)
        
        # 13. 当前阶段上下文
        observation['current_phase'] = self._get_current_phase_observation()

        return observation


    def _apply_phase_mask(self, observation: Dict, agent_id: int) -> Dict:
        """根据训练阶段应用信息屏蔽"""
        if self.training_phase == 1:
            # 阶段1：只保留私有手牌和基本公共信息
            observation['global_hand'] = np.zeros(4 * 34, dtype=np.int8)
            # 清空动作历史和副露信息
            observation['action_history'] = (
                np.full(100, 44, dtype=np.int8),  # 全部设为"未进行"
                np.full(100, 34, dtype=np.int8),  # 全部设为"无牌"
                np.full(100, 0, dtype=np.int8)  # 玩家ID设为0
            )
            observation['melds'] = self._get_empty_melds()

        elif self.training_phase == 3:
            # 阶段3：屏蔽对手手牌信息
            global_hand = observation['global_hand'].copy()
            for i in range(4):
                if i != agent_id:
                    # 屏蔽其他玩家手牌（设为0张）
                    global_hand[i * 34:(i + 1) * 34] = 0
            observation['global_hand'] = global_hand

        return observation



    # 环境核心方法
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        # 实现重置逻辑
        # 初始化游戏状态、发牌、确定庄家等
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """执行一步动作"""
        # 实现游戏逻辑
        # 处理动作、更新游戏状态、检查胡牌等
        pass

    def render(self):
        """渲染环境"""
        pass

    def close(self):
        """关闭环境"""
        pass

    def _get_action_mask(self, agent_id):
        pass

    def _get_current_phase_observation(self):
        pass


# 辅助函数：将观测数据转换为神经网络友好的格式
def flatten_observation(observation: Dict) -> np.ndarray:
    """将观测字典展平为一维数组，便于神经网络处理"""
    flattened = []

    # 展平全局手牌 (136维)
    flattened.extend(observation['global_hand'])

    # 展平私有手牌 (34维)
    flattened.extend(observation['private_hand'])

    # 展平牌河总量 (34维)
    flattened.extend(observation['discard_pool_total'])

    # 展平动作历史 (300维: 100×3)
    action_types, tile_ids, player_ids = observation['action_history']
    flattened.extend(action_types)
    flattened.extend(tile_ids)
    flattened.extend(player_ids)

    # 展平副露状态 (4×4×5=80维: 4玩家×4组×(1动作类型+4牌ID))
    for player_melds in observation['melds']:
        for meld in player_melds:
            action_type, tiles = meld
            flattened.append(action_type)
            flattened.extend(tiles)

    # 展平其他组件
    flattened.extend(observation['special_gangs'])  # 12维
    flattened.extend(observation['current_player'])  # 4维
    flattened.extend(observation['fan_counts'])  # 4维
    flattened.extend(observation['special_indicators'])  # 2维
    flattened.append(observation['remaining_tiles'])  # 1维
    flattened.append(observation['dealer'])  # 1维
    flattened.extend(observation['phase_mask'])  # 8维

    return np.array(flattened, dtype=np.float32)


def _calculate_space_dims(space):
    """计算空间的维度"""
    if isinstance(space, spaces.MultiDiscrete):
        return space.nvec.size
    elif isinstance(space, spaces.MultiBinary):
        return space.n
    elif isinstance(space, spaces.Discrete):
        return 1
    elif isinstance(space, spaces.Tuple):
        return sum(_calculate_space_dims(subspace) for subspace in space.spaces)
    elif isinstance(space, spaces.Box):
        return np.prod(space.shape)
    else:
        return 0

# 使用示例
if __name__ == "__main__":
    env = WuhanMahjongEnv(training_phase=3)

    print("基于实际规则的观测空间结构:")
    for agent, obs_space in env.observation_spaces.items():
        print(f"\n{agent}观测空间:")
        total_dims = 0
        for key, space in obs_space.spaces.items():
            dims = _calculate_space_dims(space)
            total_dims += dims
            print(f"  {key}: {dims}维 - {type(space).__name__}")
        print(f"  总维度: {total_dims}维")
        break

