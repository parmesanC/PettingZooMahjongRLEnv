"""
网页渲染器
生成HTML，麻将牌直接用汉字
"""

from typing import Dict
from ..core.GameData import GameContext
from ..core.constants import Tiles
from .TileVisualization import TileTextVisualizer


class WebRenderer:
    """
    网页渲染器
    
    特点：
    - 生成HTML模板
    - 麻将牌用汉字显示（简化）
    - 响应式布局
    - 使用内联CSS（单文件）
    - 显示special_gangs（特殊杠）
    """
    
    def __init__(self):
        self.visualizer = TileTextVisualizer()
    
    def render(self, context: GameContext, current_agent: str, action_mask: Dict = None) -> str:
        """生成完整游戏状态的HTML"""
        # 如果没有提供action_mask，尝试从context中获取
        if action_mask is None and hasattr(context, 'observation') and context.observation:
            action_mask = context.observation.get('action_mask')

        # 渲染动作区域
        if action_mask:
            action_html = self._render_action_buttons(action_mask)
        else:
            action_html = self._render_action_panel()

        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>武汉麻将</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        {self._render_header(context)}
        <div class="game-info">
            {self._render_game_info(context)}
        </div>
        <div class="players-area">
            {self._render_other_players(context, current_agent)}
        </div>
        <div class="current-player-area">
            {self._render_current_player(context, current_agent)}
        </div>
        <div class="discard-area">
            {self._render_discard_pool(context)}
        </div>
        <div class="action-area">
            {action_html}
        </div>
    </div>
    <script>
        {self._get_javascript()}
    </script>
</body>
</html>
        """
        return html
    
    def render_action_prompt(self, action_mask: Dict) -> str:
        """生成动作提示HTML"""
        types = action_mask['types']
        params = action_mask['params']
        
        html = '<div class="action-prompt"><h3>请选择动作</h3><div class="action-buttons">'

        if types[0]:
            valid_tiles = [i for i in range(34) if params[i]]
            html += f'<div class="action-group"><h4>打牌</h4>'
            for tile_id in valid_tiles:
                tile_name = self.visualizer.format_tile(Tiles(tile_id))
                html += f'<button class="action-btn tile-btn" data-action-type="0" data-parameter="{tile_id}">{tile_name}</button>'
            html += '</div>'

        for i, (is_valid, action_name, needs_param) in enumerate([
            (types[1], "吃牌", True),
            (types[2], "碰牌", False),
            (types[3], "明杠", False),
            (types[4], "补杠", True),
            (types[5], "暗杠", True),
            (types[6], "红中杠", False),
            (types[7], "皮子杠", False),
            (types[8], "赖子杠", False),
            (types[9], "胡牌", False),
            (types[10], "过牌", False),
        ]):
            if is_valid:
                html += f'<button class="action-btn" data-action-type="{i}" data-parameter="-1">{action_name}</button>'

        html += '</div></div>'
        return html
    
    def render_game_over(self, info) -> str:
        """生成游戏结束HTML"""
        winner = info.get('winners', [])
        
        if winner:
            html = f'<div class="game-over winner"><h1>🏆 玩家{winner[0]} 获胜！</h1>'
            html += f'<p>胜利方式: {info.get("win_way", "unknown")}</p></div>'
        else:
            html = '<div class="game-over"><h1>荒牌流局</h1></div>'
        
        html += '<button class="restart-btn" onclick="location.reload()">再来一局</button>'
        return html
    
    def _get_css(self) -> str:
        """内联CSS样式"""
        return """
        body {
            font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
            background-color: #2c3e50;
            color: #ecf0f1;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            font-size: 28px;
            margin-bottom: 20px;
            color: #f1c40f;
        }
        .game-info {
            background-color: #34495e;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            text-align: center;
        }
        .info-item {
            padding: 10px;
            background-color: #394b59;
            border-radius: 5px;
        }
        .players-area {
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .player-card {
            background-color: #34495e;
            padding: 15px;
            border-radius: 10px;
        }
        .player-title {
            font-size: 18px;
            margin-bottom: 10px;
            color: #f1c40f;
        }
        .tile-group {
            margin: 10px 0;
        }
        .tile-group-title {
            font-size: 14px;
            color: #8e44ad;
            margin-bottom: 5px;
        }
        .tile-row {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }
        .tile {
            width: 50px;
            height: 70px;
            background-color: #f5deb3;
            border: 2px solid #8b4513;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            font-weight: bold;
            color: #333;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tile:hover {
            transform: scale(1.1);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        .meld-tile {
            background-color: #ffe4b5;
            border-color: #d35400;
        }
        .special-gang {
            color: #e74c3c;
            font-weight: bold;
        }
        .discard-area {
            background-color: #34495e;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .action-area {
            background-color: #34495e;
            padding: 20px;
            border-radius: 10px;
        }
        .action-buttons {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        .action-group {
            grid-column: span 4;
        }
        .action-btn {
            padding: 12px 20px;
            font-size: 16px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .action-btn:hover {
            background-color: #2980b9;
        }
        .tile-btn {
            width: 60px;
            height: 80px;
            font-size: 18px;
            background-color: #f39c12;
            border-color: #d68910;
        }
        .game-over {
            text-align: center;
            padding: 40px;
            background-color: #34495e;
            border-radius: 10px;
            margin: 20px;
        }
        .game-over.winner {
            background-color: #27ae60;
        }
        .game-over h1 {
            font-size: 36px;
            margin: 0 0 20px 0;
        }
        .restart-btn {
            margin-top: 20px;
            padding: 15px 40px;
            font-size: 18px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        """
    
    def _get_javascript(self) -> str:
        """JavaScript代码"""
        return """
        let ws = null;
        let wsConnected = false;

        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:8000/ws');

            ws.onopen = function() {
                wsConnected = true;
                console.log('WebSocket connected');
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'state') {
                    document.body.innerHTML = data.html;
                    attachActionListeners();
                } else if (data.type === 'action_prompt') {
                    const actionArea = document.querySelector('.action-area');
                    if (actionArea) {
                        actionArea.innerHTML = data.html;
                        attachActionListeners();
                    }
                } else if (data.type === 'game_over') {
                    document.body.innerHTML = data.html;
                }
            };

            ws.onclose = function() {
                wsConnected = false;
                console.log('WebSocket disconnected');
            };

            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }

        function sendAction(actionType, parameter) {
            if (!wsConnected) {
                alert('WebSocket未连接，请等待连接');
                return;
            }
            ws.send(JSON.stringify({
                type: 'action',
                actionType: actionType,
                parameter: parameter
            }));
        }

        function attachActionListeners() {
            const actionBtns = document.querySelectorAll('.action-btn');
            actionBtns.forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const actionType = parseInt(e.target.dataset.actionType);
                    const parameter = parseInt(e.target.dataset.parameter);
                    sendAction(actionType, parameter);
                });
            });
        }

        // 页面加载完成后连接WebSocket并附加事件监听器
        document.addEventListener('DOMContentLoaded', () => {
            connectWebSocket();
            attachActionListeners();
        });
        """
    
    def _render_header(self, context: GameContext) -> str:
        """渲染标题"""
        return f'<div class="header">武汉麻将 - 第{context.round_info.total_rounds_played + 1}局  庄家: 玩家{context.dealer_idx}</div>'
    
    def _render_game_info(self, context: GameContext) -> str:
        """渲染游戏信息"""
        lai_tile, skin1, skin2, red_dragon = context.special_tiles
        
        items = [
            f'剩余牌墙<br><strong>{len(context.wall)}张</strong>',
            f'赖子<br><strong>{self.visualizer.format_tile(Tiles(lai_tile))}</strong>',
            f'皮子<br><strong>{self.visualizer.format_tile(Tiles(skin1))}, {self.visualizer.format_tile(Tiles(skin2))}</strong>',
            f'红中<br><strong>{self.visualizer.format_tile(Tiles(red_dragon))}</strong>'
        ]
        
        return ''.join([f'<div class="info-item">{item}</div>' for item in items])
    
    def _render_other_players(self, context: GameContext, current_agent: str) -> str:
        """渲染其他玩家"""
        current_idx = int(current_agent.split('_')[1])
        players = context.players
        
        up_idx = (current_idx - 1) % 4
        down_idx = (current_idx - 3) % 4
        across_idx = (current_idx - 2) % 4
        
        html = ''
        
        html += f'<div class="player-card"><div class="player-title">上家 (玩家{up_idx})</div>'
        html += f'<p><strong>手牌数:</strong> {len(players[up_idx].hand_tiles)}张</p>'
        html += self._render_player_melds_and_gangs(players[up_idx])
        html += '</div>'
        
        html += '<div class="column-container">'
        
        html += f'<div class="player-card"><div class="player-title">下家 (玩家{down_idx})</div>'
        html += f'<p><strong>手牌数:</strong> {len(players[down_idx].hand_tiles)}张</p>'
        html += self._render_player_melds_and_gangs(players[down_idx])
        html += '</div>'
        
        html += f'<div class="player-card"><div class="player-title">对家 (玩家{across_idx})</div>'
        html += f'<p><strong>手牌数:</strong> {len(players[across_idx].hand_tiles)}张</p>'
        html += self._render_player_melds_and_gangs(players[across_idx])
        html += '</div>'
        
        html += '</div>'
        return html
    
    def _render_player_melds_and_gangs(self, player) -> str:
        """渲染玩家的副露和特殊杠"""
        html = ''
        
        if player.melds:
            html += '<div class="tile-group"><div class="tile-group-title">副露</div>'
            html += '<div class="tile-row">' + ''.join([f'<div class="tile meld-tile">{self.visualizer.format_tile(Tiles(m.tiles[0]))}×{len(m.tiles)}</div>' for m in player.melds]) + '</div></div>'
        
        pi_gang, lai_gang, zhong_gang = player.special_gangs
        if pi_gang > 0 or lai_gang > 0 or zhong_gang > 0:
            html += '<div class="tile-group"><div class="tile-group-title special-gang">特殊杠</div>'
            if pi_gang > 0:
                html += f'<p>皮子杠: {pi_gang}次</p>'
            if lai_gang > 0:
                html += f'<p>赖子杠: {lai_gang}次</p>'
            if zhong_gang > 0:
                html += f'<p>红中杠: {zhong_gang}次</p>'
            html += '</div>'
        
        return html
    
    def _render_current_player(self, context: GameContext, current_agent: str) -> str:
        """渲染当前玩家手牌"""
        current_idx = int(current_agent.split('_')[1])
        player = context.players[current_idx]
        
        html = f'<div class="player-card current-player"><div class="player-title">你的手牌 (玩家{current_idx})</div>'
        
        tiles_by_suit = {"万": [], "筒": [], "条": [], "字": []}
        for t in sorted(player.hand_tiles):
            tile_name = self.visualizer.format_tile(Tiles(t))
            categorized = False
            for suit in ["万", "筒", "条"]:
                if suit in tile_name:
                    tiles_by_suit[suit].append(tile_name)
                    categorized = True
                    break
            if not categorized:
                tiles_by_suit["字"].append(tile_name)
        
        for suit_name, tiles in tiles_by_suit.items():
            if tiles:
                html += f'<div class="tile-group"><div class="tile-group-title">{suit_name}</div>'
                html += '<div class="tile-row">' + ''.join([f'<div class="tile">{tile}</div>' for tile in tiles]) + '</div></div>'
        
        if player.melds:
            html += '<div class="tile-group"><div class="tile-group-title">副露</div>'
            html += '<div class="tile-row">' + ''.join([f'<div class="tile meld-tile">{self.visualizer.format_tile(Tiles(m.tiles[0]))}×{len(m.tiles)}</div>' for m in player.melds]) + '</div></div>'
        
        pi_gang, lai_gang, zhong_gang = player.special_gangs
        if pi_gang > 0 or lai_gang > 0 or zhong_gang > 0:
            html += '<div class="tile-group"><div class="tile-group-title special-gang">特殊杠</div>'
            if pi_gang > 0:
                html += f'<p>皮子杠: {pi_gang}次</p>'
            if lai_gang > 0:
                html += f'<p>赖子杠: {lai_gang}次</p>'
            if zhong_gang > 0:
                html += f'<p>红中杠: {zhong_gang}次</p>'
            html += '</div>'
        
        html += '</div>'
        return html
    
    def _render_discard_pool(self, context: GameContext) -> str:
        """渲染牌河"""
        discard_pile = context.discard_pile
        
        html = '<div class="discard-area"><h3>牌河</h3>'
        
        if not discard_pile:
            html += '<p>(空)</p>'
        else:
            recent = discard_pile[-24:] if len(discard_pile) > 24 else discard_pile
            tile_names = [self.visualizer.format_tile(Tiles(t)) for t in recent]
            html += '<div class="tile-row">'
            html += ''.join([f'<div class="tile">{tile}</div>' for tile in tile_names])
            html += '</div>'
        
        html += '</div>'
        return html
    
    def _render_action_buttons(self, action_mask: Dict) -> str:
        """渲染动作按钮（直接嵌入到HTML中）"""
        types = action_mask['types']
        params = action_mask['params']

        html = '<div class="action-prompt"><h3>请选择动作</h3><div class="action-buttons">'

        # 打牌动作 (action_type = 0)
        if types[0]:
            valid_tiles = [i for i in range(34) if params[i]]
            html += f'<div class="action-group"><h4>打牌</h4>'
            for tile_id in valid_tiles:
                tile_name = self.visualizer.format_tile(Tiles(tile_id))
                html += f'<button class="action-btn tile-btn" data-action-type="0" data-parameter="{tile_id}">{tile_name}</button>'
            html += '</div>'

        # 其他动作
        for i, (is_valid, action_name, needs_param) in enumerate([
            (types[1], "吃牌", True),
            (types[2], "碰牌", False),
            (types[3], "明杠", False),
            (types[4], "补杠", True),
            (types[5], "暗杠", True),
            (types[6], "红中杠", False),
            (types[7], "皮子杠", False),
            (types[8], "赖子杠", False),
            (types[9], "胡牌", False),
            (types[10], "过牌", False),
        ]):
            if is_valid:
                html += f'<button class="action-btn" data-action-type="{i}" data-parameter="-1">{action_name}</button>'

        html += '</div></div>'
        return html

    def _render_action_panel(self) -> str:
        """渲染动作面板（等待动作）"""
        return '<div class="action-panel"><p>等待游戏状态更新...</p></div>'
