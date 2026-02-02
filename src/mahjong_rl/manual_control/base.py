"""
PettingZoo标准Manual Control基类
遵循AEC API规范：使用env.agent_iter()和env.last()
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional


class ManualController(ABC):
    """
    手动控制基类（PettingZoo标准）

    标准用法：
    ```python
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            action = None
        else:
            action = self.get_human_action(obs, info)
        env.step(action)
    ```

    设计原则：
    - SRP: 只负责控制循环和输入获取
    - DIP: 依赖抽象，不依赖具体实现
    """

    def __init__(self, env, max_episodes: int = 1, strategies: Optional[List] = None):
        """
        初始化手动控制器

        Args:
            env: PettingZoo AEC环境实例
            max_episodes: 最大回合数
            strategies: 玩家策略列表（用于AI玩家）
        """
        self.env = env
        self.max_episodes = max_episodes
        self.current_episode = 0
        self.current_step = 0
        self.strategies = strategies or []

    def _is_human_player(self, agent_name: str) -> bool:
        """检查当前agent是否是人类玩家"""
        if not self.strategies:
            return True  # 没有strategies时，默认所有都是人类玩家

        agent_idx = int(agent_name.split('_')[1])
        from src.mahjong_rl.agents.human.manual_strategy import ManualPlayerStrategy
        return isinstance(self.strategies[agent_idx], ManualPlayerStrategy)

    def _get_ai_action(self, agent_name: str, observation: Dict, info: Dict) -> Tuple[int, int]:
        """获取AI玩家的动作"""
        agent_idx = int(agent_name.split('_')[1])
        strategy = self.strategies[agent_idx]
        action_mask = observation['action_mask']
        return strategy.choose_action(observation, action_mask)

    def run(self):
        """
        运行手动控制主循环（PettingZoo标准模式）
        """
        for episode in range(self.max_episodes):
            self.current_episode = episode + 1
            self.current_step = 0

            self.env.reset()
            self.render_env()

            for agent in self.env.agent_iter():
                self.current_step += 1

                obs, reward, terminated, truncated, info = self.env.last()

                if terminated or truncated:
                    action = None
                else:
                    if self._is_human_player(agent):
                        action = self.get_human_action(obs, info)
                    else:
                        # AI玩家自动执行动作
                        action = self._get_ai_action(agent, obs, info)
                        print(f"[AI] {agent} 执行动作: {action}")

                # 执行动作并获取新的结果
                obs, reward, terminated, truncated, info = self.env.step(action)

                # 如果是人类玩家，渲染更新后的状态
                if not terminated and not truncated:
                    current_agent = self.env.agent_selection
                    if self._is_human_player(current_agent):
                        self.render_env()

            self.render_final_state(info)

    @abstractmethod
    def render_env(self):
        """
        渲染环境状态
        """
        pass

    @abstractmethod
    def get_human_action(self, observation, info) -> Tuple[int, int]:
        """
        获取人类动作（元组形式）

        Returns:
            (action_type, parameter) 元组
            - action_type: 0-10 (DISCARD, CHOW, PONG, KONG_*, WIN, PASS)
            - parameter: 牌ID (0-33) 或 吃牌类型 (0-2)
        """
        pass

    @abstractmethod
    def render_final_state(self, info):
        """
        渲染最终状态
        """
        pass
