"""
场景测试框架 - 测试执行器

负责执行配置好的测试场景，验证状态转换和游戏状态。
增强版：支持详细日志输出、手牌格式化显示、动作合法性检测。
"""

from typing import List
from tests.scenario.context import ScenarioContext, StepConfig, TestResult
from src.mahjong_rl.core.constants import GameStateType, ActionType, Tiles
from src.mahjong_rl.core.mahjong_action import MahjongAction
from src.mahjong_rl.visualization.TileVisualization import TileTextVisualizer

# ANSI 颜色代码
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


class TestExecutor:
    """测试执行器

    执行测试场景，收集验证结果。
    """

    def __init__(self, scenario: ScenarioContext, verbose: bool = True, tile_format: str = "name"):
        """初始化执行器

        Args:
            scenario: 测试场景配置
            verbose: 是否打印详细信息
            tile_format: 牌显示格式，"name"（牌名）或 "number"（数字）
        """
        self.scenario = scenario
        self.env = None
        self.result = TestResult(scenario_name=scenario.name, success=False)
        self.verbose = verbose
        self.tile_format = tile_format
        self.visualizer = TileTextVisualizer()

    def run(self) -> TestResult:
        """执行测试场景

        Returns:
            测试结果
        """
        try:
            # 延迟导入避免循环依赖
            from example_mahjong_env import WuhanMahjongEnv

            # 创建环境
            self.env = WuhanMahjongEnv(
                render_mode=None,
                training_phase=3,  # 完整信息
                enable_logging=False  # 测试时关闭日志
            )

            # 先调用 reset() 初始化 context 和状态机
            # 即使使用自定义初始化，我们也需要先初始化基础结构
            self.env.reset(seed=self.scenario.seed)

            # 检查是否有自定义初始状态配置
            if self.scenario.initial_config is not None:
                # 使用自定义初始化，覆盖 reset() 的结果
                self._apply_custom_initialization()
            else:
                # 使用标准初始化流程
                # 配置牌墙（标准流程）
                if self.scenario.wall:
                    self.env.context.wall.clear()
                    self.env.context.wall.extend(self.scenario.wall)

                # 配置特殊牌（标准流程）
                if self.scenario.special_tiles:
                    if 'lazy' in self.scenario.special_tiles:
                        self.env.context.lazy_tile = self.scenario.special_tiles['lazy']
                    if 'skins' in self.scenario.special_tiles:
                        skins = self.scenario.special_tiles['skins']
                        if len(skins) >= 2:
                            self.env.context.skin_tile = [skins[0], skins[1]]

            self.result.total_steps = len(self.scenario.steps)

            # 执行每个步骤
            for step_config in self.scenario.steps:
                self._execute_step(step_config)
                self.result.executed_steps += 1

            # 执行最终验证
            if self.scenario.final_validators:
                for validator in self.scenario.final_validators:
                    if not validator(self.env.context):
                        raise AssertionError(f"最终验证失败: {validator.__name__}")

            # 验证获胜者
            if self.scenario.expect_winner is not None:
                if set(self.env.context.winner_ids) != set(self.scenario.expect_winner):
                    raise AssertionError(
                        f"获胜者验证失败: 预期 {self.scenario.expect_winner}, "
                        f"实际 {self.env.context.winner_ids}"
                    )

            self.result.success = True
            self.result.final_state = self.env.state_machine.current_state_type
            # 无论成功失败都创建快照
            self.result.final_context_snapshot = self._create_snapshot()

            # 显示游戏分数（如果游戏结束）
            if self.verbose:
                self._print_scores()

        except Exception as e:
            self.result.success = False
            self.result.failed_step = self.result.executed_steps + 1
            self.result.failure_message = str(e)

            # 保存快照用于调试
            if self.env and self.env.context:
                self.result.final_context_snapshot = self._create_snapshot()

        finally:
            # 确保环境资源被正确释放
            if self.env is not None:
                self.env.close()

        return self.result

    def _execute_step(self, step: StepConfig):
        """执行单个步骤

        Args:
            step: 步骤配置

        Raises:
            AssertionError: 验证失败
            Exception: 执行错误
        """
        # 1. 打印步骤执行前的状态
        if self.verbose:
            self._print_game_state(step, is_before=True)

        # 2. 执行步骤
        if step.is_auto:
            self._auto_advance(step)
        elif step.is_action:
            self._execute_action(step)

        # 3. 打印步骤执行后的状态
        if self.verbose:
            self._print_game_state(step, is_before=False)

        # 4. 执行验证
        self._run_validations(step)

    def _execute_action(self, step: StepConfig):
        """执行动作步骤

        Args:
            step: 步骤配置

        Raises:
            AssertionError: 验证失败
        """
        # 打印动作信息
        formatted_param = self._format_action_param(step.action_type, step.parameter)
        action_str = f"{step.action_type.name}({formatted_param})"
        print(f"  玩家 {step.player} 尝试: {action_str}")

        # 执行动作，根据异常和 info 判断合法性
        try:
            action = (step.action_type.value, step.parameter)
            obs, reward, terminated, truncated, info = self.env.step(action)

            # 检查 info 中的错误信息
            if 'error' in info:
                print(f"  {RED}→ 动作非法: {info['error']}{RESET}")
            else:
                print(f"  → 执行成功，转移到状态: {self.env.state_machine.current_state_type.name}")
        except ValueError as e:
            # 动作非法（格式错误等）
            print(f"  {RED}→ 动作非法: {str(e)}{RESET}")

    def _auto_advance(self, step: StepConfig):
        """自动推进步骤

        Args:
            step: 步骤配置

        Raises:
            AssertionError: 验证失败
        """
        # 自动状态已由 env.step() 内部处理
        # 这里只需要验证当前状态
        print(f"  自动推进到: {self.env.state_machine.current_state_type.name}")

    # ==================== 格式化方法 ====================

    def _format_hand(self, hand_tiles: List[int]) -> str:
        """格式化手牌显示

        示例输出: [1万, 5万, 8条, 1筒, 2筒, 东, 南, 西, 北]
        """
        if self.tile_format == "name":
            # 使用 TileTextVisualizer，不分组，得到 "1万 5万 8条..."
            formatted = self.visualizer.format_hand(hand_tiles, group_by_suit=False)
            # 替换空格为 ", " 并加上方括号
            return "[" + ", ".join(formatted.split()) + "]"
        else:
            # 数字格式: [1, 5, 8, 19, 20, 28, 29, 30, 31]
            return "[" + ", ".join(map(str, sorted(hand_tiles))) + "]"

    def _format_action_param(self, action_type: ActionType, param: int) -> str:
        """格式化动作参数，牌ID转为牌名"""
        # 需要格式化牌名的动作类型
        tile_actions = {
            ActionType.DISCARD,
            ActionType.KONG_SUPPLEMENT,
            ActionType.KONG_CONCEALED,
            ActionType.KONG_SKIN,
            ActionType.KONG_LAZY,
            ActionType.KONG_RED,
        }

        if action_type in tile_actions and param >= 0:
            if self.tile_format == "name":
                return self.visualizer.format_tile(Tiles(param))
            else:
                return str(param)
        return str(param)

    def _format_melds(self, player_idx: int) -> List[str]:
        """格式化玩家的牌组信息

        Args:
            player_idx: 玩家索引

        Returns:
            牌组描述列表，如 ["碰: 一万", "明杠: 五条"]
        """
        player = self.env.context.players[player_idx]
        if not player.melds:
            return []

        result = []
        for meld in player.melds:
            try:
                # 获取动作类型（action_type 已经是 ActionType 枚举）
                action = meld.action_type.action_type
                action_name = action.name

                # 获取第一张牌的名称（需要先转换为 Tiles 枚举）
                if meld.tiles and len(meld.tiles) > 0:
                    tile_name = self.visualizer.format_tile(Tiles(meld.tiles[0]))
                else:
                    tile_name = ""

                # 根据动作类型格式化
                if action_name == "CHOW":
                    result.append(f"吃: {tile_name}")
                elif action_name == "PONG":
                    result.append(f"碰: {tile_name}")
                elif action_name == "KONG_EXPOSED":
                    result.append(f"明杠: {tile_name}")
                elif action_name == "KONG_CONCEALED":
                    result.append("暗杠")
                elif action_name == "KONG_SUPPLEMENT":
                    result.append(f"补杠: {tile_name}")
                else:
                    result.append(f"{action_name}: {tile_name}")
            except Exception:
                # 如果格式化失败，返回一个通用描述
                result.append("(牌组)")

        return result

    def _get_win_way_name(self) -> str:
        """获取胡牌方式的中文名称

        Returns:
            胡牌方式名称，如 "自摸"、"点炮" 等
        """
        from src.mahjong_rl.core.constants import WinWay

        win_way = self.env.context.win_way
        if win_way is None:
            return "未知"

        way_map = {
            WinWay.SELF_DRAW.value: "自摸",
            WinWay.DISCARD.value: "点炮",
            WinWay.KONG_SELF_DRAW.value: "杠上开花",
            WinWay.ROB_KONG.value: "抢杠和",
        }
        return way_map.get(int(win_way), "未知")

    # ==================== 打印方法 ====================

    def _print_game_state(self, step: StepConfig, is_before: bool = True):
        """打印当前游戏状态

        Args:
            step: 步骤配置
            is_before: True表示步骤开始前，False表示步骤执行后
        """
        prefix = "步骤执行前" if is_before else "步骤执行后"
        print(f"\n{'='*60}")
        print(f"步骤 {step.step_number}: {step.description} [{prefix}]")
        print(f"{'='*60}")

        context = self.env.context

        # 当前状态和玩家
        print(f"当前状态: {context.current_state.name}")
        print(f"当前玩家: {context.current_player_idx}")

        # 手牌
        print(f"\n--- 手牌 ---")
        for i, player in enumerate(context.players):
            print(f"  玩家 {i}: {self._format_hand(player.hand_tiles)}")

        # 弃牌堆（显示最后10张）
        if context.discard_pile:
            print(f"\n--- 弃牌堆 (最近10张) ---")
            recent = context.discard_pile[-10:]
            formatted_recent = [self._format_action_param(ActionType.DISCARD, t) for t in recent]
            print(f"  [{', '.join(formatted_recent)}]")

        # 牌墙数量
        print(f"\n牌墙剩余: {len(context.wall)} 张")

    def _print_scores(self):
        """打印游戏结束时的玩家分数和明细

        只有在游戏结束（WIN 或 FLOW_DRAW）且有分数数据时才打印
        """
        context = self.env.context

        # 只有在游戏结束且有分数时才打印
        if not context.final_scores:
            return

        # 检查游戏是否结束
        current_state = self.env.state_machine.current_state_type
        if current_state not in [GameStateType.WIN, GameStateType.FLOW_DRAW]:
            return

        print(f"\n{'='*60}")
        print(f"游戏结束 - 最终分数")
        print(f"{'='*60}")

        # 1. 显示每个玩家的最终得分
        for i, score in enumerate(context.final_scores):
            if score > 0:
                print(f"  玩家 {i}: {GREEN}+{score}{RESET}")
            elif score < 0:
                print(f"  玩家 {i}: {RED}{score}{RESET}")
            else:
                print(f"  玩家 {i}: 0")

        print()  # 空行

        # 2. 显示获胜者或流局信息
        if context.is_flush:
            print(f"{YELLOW}流局{RESET}")
        elif context.winner_ids:
            winners = ", ".join(f"玩家 {w}" for w in context.winner_ids)
            print(f"获胜者: {GREEN}{winners}{RESET}")

            # 显示胡牌方式
            win_way = self._get_win_way_name()
            print(f"胡牌方式: {win_way}")

        # 3. 显示庄家
        if context.dealer_idx is not None:
            print(f"庄家: 玩家 {context.dealer_idx}")

        print()  # 空行

        # 4. 显示每个玩家的牌组
        for i, player in enumerate(context.players):
            print(f"玩家 {i} 牌组:")

            # 显示吃碰杠
            melds = self._format_melds(i)
            if melds:
                for meld in melds:
                    print(f"  - {meld}")
            else:
                print(f"  (无)")

            # 显示特殊杠
            pi_zi, lai_zi, hong_zhong = player.special_gangs
            special_gangs = []
            if pi_zi > 0:
                special_gangs.append(f"皮子杠 x{pi_zi}")
            if lai_zi > 0:
                special_gangs.append(f"赖子杠 x{lai_zi}")
            if hong_zhong > 0:
                special_gangs.append(f"红中杠 x{hong_zhong}")

            if special_gangs:
                print(f"  特殊杠: {', '.join(special_gangs)}")

        print(f"{'='*60}\n")

    # ==================== 动作验证方法 ====================

    def _run_validations(self, step: StepConfig):
        """运行所有验证

        Args:
            step: 步骤配置

        Raises:
            AssertionError: 验证失败
        """
        context = self.env.context

        # 验证状态
        if step.expect_state:
            actual = self.env.state_machine.current_state_type
            if actual != step.expect_state:
                raise AssertionError(
                    f"状态验证失败: 预期 {step.expect_state.name}, "
                    f"实际 {actual.name if actual else None}"
                )

        # 验证 action_mask
        if step.expect_action_mask_contains:
            mask = context.action_mask
            for action_type in step.expect_action_mask_contains:
                action = MahjongAction(action_type, -1)
                index = self.env._action_to_index(action)
                if index < 0 or index >= len(mask) or mask[index] != 1:
                    raise AssertionError(
                        f"action_mask 验证失败: {action_type.name} 不在可用动作中"
                    )

        # 执行自定义验证器
        for validator in step.validators:
            if not validator(context):
                raise AssertionError(f"验证器失败: {validator.__name__}")

        # 快捷验证：手牌
        if step.verify_hand_tiles:
            for player_id, tiles in step.verify_hand_tiles.items():
                validator = hand_contains(player_id, tiles)
                if not validator(context):
                    raise AssertionError(f"手牌验证失败: 玩家 {player_id}")

        # 快捷验证：牌墙数量
        if step.verify_wall_count is not None:
            validator = wall_count_equals(step.verify_wall_count)
            if not validator(context):
                raise AssertionError(f"牌墙数量验证失败")

        # 快捷验证：弃牌堆
        if step.verify_discard_pile_contains:
            for tile in step.verify_discard_pile_contains:
                validator = discard_pile_contains(tile)
                if not validator(context):
                    raise AssertionError(f"弃牌堆验证失败: 牌 {tile}")

    def _create_snapshot(self) -> dict:
        """创建上下文快照用于调试

        Returns:
            快照字典
        """
        context = self.env.context
        return {
            'current_state': context.current_state.name if context.current_state else None,
            'current_player': context.current_player_idx,
            'wall_count': len(context.wall),
            'discard_pile': context.discard_pile[-10:] if context.discard_pile else [],  # 最后10张
            'player_hand_counts': [len(p.hand_tiles) for p in context.players],
            'winner_ids': context.winner_ids if hasattr(context, 'winner_ids') else [],
            'final_scores': context.final_scores if hasattr(context, 'final_scores') else [],
        }

    def _apply_custom_initialization(self) -> None:
        """应用自定义初始状态配置

        根据 scenario.initial_config 配置游戏状态，
        完全绕过 InitialState 的自动初始化流程。

        Raises:
            ValueError: 如果配置缺少必需字段
        """
        config = self.scenario.initial_config
        if not config:
            return

        context = self.env.context

        # 1. 设置庄家
        if 'dealer_idx' in config:
            dealer_idx = config['dealer_idx']
            if not 0 <= dealer_idx <= 3:
                raise ValueError(f"dealer_idx 必须在 0-3 之间，得到 {dealer_idx}")
            context.dealer_idx = dealer_idx

        # 2. 设置当前玩家
        if 'current_player_idx' in config:
            current_player = config['current_player_idx']
            if not 0 <= current_player <= 3:
                raise ValueError(f"current_player_idx 必须在 0-3 之间，得到 {current_player}")
            context.current_player_idx = current_player

        # 3. 设置玩家手牌
        if 'hands' in config:
            hands = config['hands']
            for player_id, tiles in hands.items():
                if not 0 <= player_id <= 3:
                    raise ValueError(f"玩家ID必须在 0-3 之间，得到 {player_id}")
                context.players[player_id].hand_tiles = tiles.copy()
                # 设置 is_dealer 标志
                if 'dealer_idx' in config:
                    context.players[player_id].is_dealer = (player_id == config['dealer_idx'])

        # 4. 设置牌墙
        if 'wall' in config:
            context.wall.clear()
            context.wall.extend(config['wall'])

        # 5. 设置特殊牌
        if 'special_tiles' in config:
            special = config['special_tiles']
            if 'lazy' in special:
                context.lazy_tile = special['lazy']
            if 'skins' in special:
                skins = special['skins']
                if len(skins) >= 2:
                    context.skin_tile = [skins[0], skins[1]]
            # 更新 special_tiles 元组
            context.special_tiles = (
                context.lazy_tile,
                context.skin_tile[0] if context.skin_tile else -1,
                context.skin_tile[1] if context.skin_tile else -1,
                context.red_dragon
            )

        # 6. 设置 last_drawn_tile（庄家刚摸的牌）
        if 'last_drawn_tile' in config:
            context.last_drawn_tile = config['last_drawn_tile']

        # 7. 初始化其他必要字段
        context.current_state = GameStateType.PLAYER_DECISION
        context.observation = None
        context.action_mask = None

        # 重置响应相关状态
        context.discard_pile = []
        context.last_discarded_tile = None
        context.pending_responses = {}
        context.response_order = []
        context.current_responder_idx = 0
        context.selected_responder = None
        context.response_priorities = {}

        # 重置杠牌相关状态
        context.last_kong_action = None
        context.last_kong_player_idx = None

        # 重置游戏结果状态
        context.is_win = False
        context.is_flush = False
        context.winner_ids = []
        context.reward = 0.0

        # 初始化 special_gangs
        for i in range(4):
            context.players[i].special_gangs = [0, 0, 0]

        # 8. 同步 env.agent_selection（重要！env 依赖 agent_selection 来确定当前玩家）
        # agent_selection 格式为 "player_0", "player_1" 等
        self.env.agent_selection = f'player_{context.current_player_idx}'


# 导入验证器函数用于快捷验证
from tests.scenario.validators import hand_contains, wall_count_equals, discard_pile_contains
