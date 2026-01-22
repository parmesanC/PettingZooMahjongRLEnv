"""
简化版CLI渲染器
使用纯文本，麻将牌直接用汉字，包含special_gangs显示
"""

from typing import Dict
from ..core.GameData import GameContext
from ..core.constants import Tiles, ActionType


class SimpleCLIRenderer:
    """
    简化版CLI渲染器
    
    特点：
    - 纯文本布局（无Unicode框）
    - 麻将牌直接用汉字
    - 清晰的信息分区
    - 显示special_gangs（特殊杠）
    """
    
    def __init__(self):
        pass
    
    def render(self, context: GameContext, current_agent: str):
        """渲染完整游戏状态"""
        self._render_header(context)
        print()
        self._render_game_info(context)
        print()
        print("-" * 60)
        self._render_other_players(context, current_agent)
        print("-" * 60)
        print()
        self._render_current_player(context, current_agent)
        print()
        print("-" * 60)
        print()
        self._render_discard_pool(context)
        print()
    
    def render_action_help(self, action_mask):
        """渲染动作帮助（包含牌ID对照表）"""
        print("\n" + "=" * 60)
        print("可用动作 (请输入元组形式):")
        print("=" * 60)

        # 新的145位action_mask直接访问
        action_ranges = {
            0: (0, 33, "(0, 牌ID)    打牌 - 输入要打出的牌ID"),
            1: (34, 36, "(1, 0/1/2)   吃牌 - 0=左吃, 1=中吃, 2=右吃"),
            2: (37, 37, "(2, 0)       碰牌"),
            3: (38, 38, "(3, 0)       明杠"),
            4: (39, 72, "(4, 牌ID)    补杠 - 输入要补杠的牌ID"),
            5: (73, 106, "(5, 牌ID)    暗杠 - 输入要暗杠的牌ID"),
            6: (107, 107, "(6, 0)       红中杠"),
            7: (108, 141, "(7, 牌ID)    皮子杠 - 输入皮子的牌ID"),
            8: (142, 142, "(8, 0)       赖子杠"),
            9: (143, 143, "(9, -1)      胡牌"),
            10: (144, 144, "(10, -1)     过牌"),
        }

        for action_type, (start, end, desc) in action_ranges.items():
            # 检查该动作类型的位是否有任何一个为1
            if any(action_mask[start:end+1]):
                print(f"  {desc}")

        print("\n" + "-" * 60)
        print("牌ID对照表:")
        print("-" * 60)
        self._render_tile_id_table()
        print("=" * 60)
    
    def _render_header(self, context: GameContext):
        """渲染标题"""
        print("=" * 60)
        print(f"武汉麻将 - 第{context.round_info.total_rounds_played + 1}局    庄家: 玩家{context.dealer_idx}")
        print("=" * 60)
    
    def _render_game_info(self, context: GameContext):
        """渲染游戏信息"""
        lai_tile, skin1, skin2, red_dragon = context.special_tiles
        
        print(f"剩余牌墙: {len(context.wall)}张")
        print(f"赖子: {Tiles.get_tile_name(lai_tile)}")
        print(f"皮子: {Tiles.get_tile_name(skin1)}, {Tiles.get_tile_name(skin2)}")
        print(f"红中: {Tiles.get_tile_name(red_dragon)}")
    
    def _render_other_players(self, context: GameContext, current_agent: str):
        """渲染其他玩家"""
        current_idx = int(current_agent.split('_')[1])
        players = context.players
        
        print("其他玩家:")
        
        for offset, name in [(-1, "上家"), (-2, "对家"), (-3, "下家")]:
            idx = (current_idx + offset) % 4
            player = players[idx]
            
            print(f"  {name} (玩家{idx}): {len(player.hand_tiles)}张牌")
            
            if player.melds:
                meld_strs = [self._format_meld(m) for m in player.melds]
                print(f"    副露: " + " ".join(meld_strs))
            else:
                print(f"    副露:")
            
            pi_gang, lai_gang, zhong_gang = player.special_gangs
            if pi_gang > 0 or lai_gang > 0 or zhong_gang > 0:
                special_strs = []
                if pi_gang > 0:
                    special_strs.append(f"皮子杠×{pi_gang}")
                if lai_gang > 0:
                    special_strs.append(f"赖子杠×{lai_gang}")
                if zhong_gang > 0:
                    special_strs.append(f"红中杠×{zhong_gang}")
                print(f"    特殊杠: " + " ".join(special_strs))
            else:
                print(f"    特殊杠:")
    
    def _render_current_player(self, context: GameContext, current_agent: str):
        """渲染当前玩家手牌"""
        current_idx = int(current_agent.split('_')[1])
        player = context.players[current_idx]
        
        print(f"你的手牌 (玩家{current_idx}):")

        tiles_by_suit = {"万": [], "筒": [], "条": [], "字": []}
        for t in sorted(player.hand_tiles):
            tile_name = Tiles.get_tile_name(t)
            # 检查是否是万/筒/条，否则归类为字牌
            if "万" in tile_name:
                tiles_by_suit["万"].append(tile_name)
            elif "筒" in tile_name:
                tiles_by_suit["筒"].append(tile_name)
            elif "条" in tile_name:
                tiles_by_suit["条"].append(tile_name)
            else:
                # 字牌（风牌和箭牌）
                tiles_by_suit["字"].append(tile_name)
        
        for suit_name, tiles in tiles_by_suit.items():
            if tiles:
                tiles_str = " ".join(tiles)
                print(f"  {suit_name}: {tiles_str}")
        
        if player.melds:
            print(f"  副露:")
            meld_strs = [self._format_meld(m) for m in player.melds]
            print("    " + " ".join(meld_strs))
        else:
            print(f"  副露:")
        
        pi_gang, lai_gang, zhong_gang = player.special_gangs
        if pi_gang > 0 or lai_gang > 0 or zhong_gang > 0:
            print(f"  特殊杠:")
            if pi_gang > 0:
                print(f"    皮子杠: {pi_gang}次")
            if lai_gang > 0:
                print(f"    赖子杠: {lai_gang}次")
            if zhong_gang > 0:
                print(f"    红中杠: {zhong_gang}次")
        else:
            print(f"  特殊杠:")
    
    def _render_discard_pool(self, context: GameContext):
        """渲染牌河"""
        discard_pile = context.discard_pile
        
        print(f"牌河 (最近{min(24, len(discard_pile))}张):")
        
        if not discard_pile:
            print("  (空)")
        else:
            recent = discard_pile[-24:] if len(discard_pile) > 24 else discard_pile
            tile_names = [Tiles.get_tile_name(t) for t in recent]
            
            for i in range(0, len(tile_names), 8):
                row = " ".join(tile_names[i:i+8])
                print(f"  {row}")
    
    def _render_tile_id_table(self):
        """渲染牌ID对照表"""
        print("  万: 0=1万, 1=2万, 2=3万, 3=4万, 4=5万, 5=6万, 6=7万, 7=8万, 8=9万")
        print("  条: 9=1条, 10=2条, 11=3条, 12=4条, 13=5条, 14=6条, 15=7条, 16=8条, 17=9条")
        print("  筒: 18=1筒, 19=2筒, 20=3筒, 21=4筒, 22=5筒, 23=6筒, 24=7筒, 25=8筒, 26=9筒")
        print("  字: 27=东风, 28=南风, 29=西风, 30=北风, 31=红中, 32=发财, 33=白板")
    
    def _format_meld(self, meld):
        """格式化副露"""
        tile_name = Tiles.get_tile_name(meld.tiles[0])
        return f"{tile_name}×{len(meld.tiles)}"
