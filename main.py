from gymnasium import spaces
import numpy as np
from gymnasium.spaces import Box
from pettingzoo import AECEnv
from typing import Dict, List, Optional, Tuple

class MahjongEnv(AECEnv):
    metadata = {
        'render.modes': ['human', 'ansi'],
        'name': 'mahjong_v0',
        'num_agents': 4,
    }

    # 牌编码：0-26: 万/筒/条 1-9（0=1万, 8=9万, 9=1筒, 17=9筒, 18=1条, 25=9条）
    # 27-33: 东南西北中发白（27=东, 28=南, 29=西, 30=北, 31=红中, 32=发财, 33=白板）
    # 动作空间设计
    # 0-33: 打牌; 34: 左吃; 35: 中吃; 36: 右吃; 37: 碰; 38: 冲杠（直接开杠）;
    # 39: 蓄杠（碰后开杠）; 40: 暗杠; 41: 赖子杠; 42: 红中杠; 43: 皮子杠; 44: 胡; 45: 过

    TILE_COUNT = 34
    HAND_SIZE = 13
    ACTION_COUNT = 46


    def __init__(self, render_mode=None, num_agents=4):
        super().__init__()
        self.render_mode = render_mode
        self.num_agents = num_agents
        self.agents = [f'agent_{i}' for i in range(num_agents)]
        self.possible_agents = self.agents[:]
        self.agents_name_mapping = {agent: i for i, agent in enumerate(self.agents)}

        self.action_spaces = {agent: spaces.Discrete(self.ACTION_COUNT) for agent in self.agents}

        # 观察空间：手牌34 + 牌河34 + 皮子标记1 + 赖子标记1 + 红中标记1 + 当前玩家4 + 开口次数4 + 是否听牌4
        # 总维度 = 34+34+1+1+1+4+4+4 = 83
        self._observation_spaces = {
            agent: Box(low=0, high=4, shape=(83,), dtype=np.int8)
            for agent in self.agents
        }

        self.reset()


    def reset(self, seed=None, options=None):
        pass

    def step(self, action):
        pass

    def render(self):
        if self.render_mode == 'human':
            print('render human')
        elif self.render_mode == 'ansi':
            print('render ansi')
        else:
            raise ValueError(f'Unknown render mode {self.render_mode}')

    def observe(self, agent):
        return np.random.rand(10)

    def close(self):
        pass


def _create_observation_spaces(self) -> Dict[str, spaces.Dict]:
    """为每个玩家创建观测空间 - 扁平化版本"""
    observation_spaces = {}

    for agent in self.possible_agents:
        observation_spaces[agent] = spaces.Dict({
            # 手牌信息
            'global_hand': spaces.MultiDiscrete([5] * (4 * 34)),  # 4玩家×34种牌
            'private_hand': spaces.MultiDiscrete([5] * 34),  # 私有手牌

            # 牌河总量
            'discard_pool_total': spaces.MultiDiscrete([5] * 34),

            # 动作历史 - 扁平化: 100个动作 × 3个元素
            'action_history': spaces.MultiDiscrete([45, 35, 4] * 100),
            # 解释: 每个动作 = (action_type, tile_id, player_id)
            # action_type: 0-43有效动作, 44表示未进行
            # tile_id: 0-33有效牌, 34表示无牌
            # player_id: 0-3玩家ID

            # 副露状态 - 扁平化: 4玩家 × 4组 × 5个元素
            'melds_flat': spaces.MultiDiscrete([44, 35, 35, 35, 35] * 4 * 4),
            # 解释: 每组副露 = (action_type, tile1, tile2, tile3, tile4)
            # action_type: 0-43有效动作, 44表示无效副露
            # tile1-tile4: 0-33有效牌, 34表示无牌

            # 特殊杠状态
            'special_gangs': spaces.MultiDiscrete([7, 3, 4] * 4),

            # 游戏状态
            'current_player': spaces.MultiBinary(4),
            'fan_counts': spaces.MultiDiscrete([600] * 4),
            'special_indicators': spaces.MultiDiscrete([34, 34]),
            'remaining_tiles': spaces.Discrete(137),
            'dealer': spaces.Discrete(4),
        })

    return observation_spaces


def observe(self, agent: str) -> Dict:
    """获取指定agent的观测 - 扁平化版本"""
    agent_id = int(agent.split('_')[1])

    # 获取原始观测数据
    raw_observation = {
        'global_hand': self._get_global_hand(),
        'private_hand': self._get_private_hand(agent_id),
        'discard_pool_total': self._get_discard_pool_total(),
        'action_history': self._get_action_history_flat(),
        'melds_flat': self._get_melds_flat(),
        'special_gangs': self._get_special_gangs_state(),
        'current_player': self._get_current_player_onehot(),
        'fan_counts': self._get_fan_counts(),
        'special_indicators': self._get_special_indicators(),
        'remaining_tiles': self._get_remaining_tiles(),
        'dealer': self._get_dealer_info(),
    }

    # 根据训练阶段应用信息屏蔽
    return self._apply_phase_masking(raw_observation, agent_id)


def _get_action_history_flat(self) -> np.ndarray:
    """获取扁平化的动作历史"""
    # 假设我们有方法获取原始动作历史数据
    action_types, tile_ids, player_ids = self._get_raw_action_history()

    # 扁平化: 将三个数组合并为一个交替的数组
    flat_history = np.empty(300, dtype=np.int8)  # 100个动作 × 3个元素
    for i in range(100):
        flat_history[i * 3] = action_types[i]  # 动作类型
        flat_history[i * 3 + 1] = tile_ids[i]  # 牌ID
        flat_history[i * 3 + 2] = player_ids[i]  # 玩家ID

    return flat_history


def _get_melds_flat(self) -> np.ndarray:
    """获取扁平化的副露状态"""
    # 假设我们有方法获取原始副露数据
    raw_melds = self._get_raw_melds()

    # 扁平化: 4玩家 × 4组 × 5个元素 = 80维
    flat_melds = np.full(80, 34, dtype=np.int8)  # 默认填充无牌

    idx = 0
    for player_melds in raw_melds:
        for meld in player_melds:
            action_type, tiles = meld
            flat_melds[idx] = action_type if action_type != -1 else 44  # -1转44
            idx += 1
            for tile in tiles:
                flat_melds[idx] = tile
                idx += 1

    return flat_melds


def _get_raw_action_history(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """获取原始动作历史数据（供内部使用）"""
    # 实现逻辑：返回三个数组 (action_types, tile_ids, player_ids)
    action_types = np.full(100, 44, dtype=np.int8)  # 44=未进行
    tile_ids = np.full(100, 34, dtype=np.int8)  # 34=无牌
    player_ids = np.full(100, 0, dtype=np.int8)

    # 填充实际数据...
    return action_types, tile_ids, player_ids


def _get_raw_melds(self) -> List[List[Tuple[int, np.ndarray]]]:
    """获取原始副露数据（供内部使用）"""
    # 实现逻辑：返回嵌套列表结构
    melds = []
    for _ in range(4):
        player_melds = []
        for _ in range(4):
            # 默认填充无效副露
            player_melds.append((-1, np.full(4, 34, dtype=np.int8)))
        melds.append(player_melds)

    # 填充实际数据...
    return melds


def _apply_phase_masking(self, observation: Dict, agent_id: int) -> Dict:
    """根据训练阶段应用信息屏蔽"""
    if self.training_phase == 1:
        # 阶段1：基础训练，只保留最基本的信息
        observation['global_hand'] = np.zeros(4 * 34, dtype=np.int8)
        observation['action_history'] = np.full(300, [44, 34, 0], dtype=np.int8).flatten()
        observation['melds_flat'] = np.full(80, [44, 34, 34, 34, 34], dtype=np.int8).flatten()
        observation['special_gangs'] = np.zeros(12, dtype=np.int8)

    elif self.training_phase == 3:
        # 阶段3：实战训练，屏蔽对手手牌但保留其他信息
        global_hand = observation['global_hand'].copy()
        for i in range(4):
            if i != agent_id:
                global_hand[i * 34:(i + 1) * 34] = 0
        observation['global_hand'] = global_hand

    return observation