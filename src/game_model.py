import random
import copy
from src.adj_map import DEFAULT_BOARD

# --- 1. 遊戲核心邏輯 (Model) ---
class MoonPhaseGame:
    def __init__(self, adj_map=None):
        self.adj_map = adj_map if adj_map is not None else DEFAULT_BOARD.adj_map
        self.nodes = {k: {'val': None, 'owner': None} for k in self.adj_map.keys()}
        self.scores = {'P1': 0, 'P2': 0}
        
        # 初始化獨立牌堆 (使用雙倍 Bag 系統)
        self.decks = {'P1': [], 'P2': []}
        for p in ['P1', 'P2']:
            self._refill_deck(p)  # 初始填充
            self._refill_deck(p)  # 緩衝
        
        self.hands = {
            'P1': [self._draw('P1') for _ in range(3)],
            'P2': [self._draw('P2') for _ in range(3)]
        }
        self.turn = 'P1'
        self.game_over = False
        self.highlighted_edges = []  # 只有月週期的邊
        self.edge_markers = {}  # 邊上的標記 {(u, v): 'pair'|'complement'}
        self.scored_nodes = set()  # 上一次移動中有得分的所有節點
        self.last_scoring_logs = {'P1': "無", 'P2': "無"}  # 分別記錄 P1 和 P2 上一次移動的加分邏輯

    def _refill_deck(self, player):
        """生成一個新的雙倍 Bag (0-7 各兩張) 並加入指定玩家的牌堆"""
        # 雙倍 Bag: 每個數字 (0-7) 各 2 張，共 16 張
        bag = list(range(8)) * 2
        random.shuffle(bag)
        self.decks[player].extend(bag)

    def _draw(self, player):
        if not self.decks[player]:
            self._refill_deck(player)
        return self.decks[player].pop(0)

    def get_chains(self, start_node, current_val, step, visited):
        """ DFS 尋找特定方向的連續路徑 """
        target_val = (current_val + step) % 8
        valid_paths = []
        found_extension = False
        
        for neighbor in self.adj_map[start_node]:
            n_data = self.nodes[neighbor]
            if n_data['val'] == target_val and neighbor not in visited:
                found_extension = True
                sub_paths = self.get_chains(neighbor, target_val, step, visited + [neighbor])
                for path in sub_paths:
                    valid_paths.append([neighbor] + path)
        
        if not found_extension: return [[]]
        return valid_paths

    def _calculate_pair_score(self, node_id, current_val):
        """計算同象與互補配對得分"""
        points = 0
        captured = set()
        details = []
        new_markers = {}
        
        for neighbor in self.adj_map[node_id]:
            n_val = self.nodes[neighbor]['val']
            if n_val is None: continue
            
            edge = tuple(sorted([node_id, neighbor]))
            if abs(current_val - n_val) == 4: # 互補
                points += 2
                captured.add(neighbor)
                new_markers[edge] = 'complement'
                details.append("互補 +2")
            elif current_val == n_val: # 同象
                points += 1
                captured.add(neighbor)
                new_markers[edge] = 'pair'
                details.append("同象 +1")
                
        return points, captured, details, new_markers

    def _calculate_chain_score(self, node_id, current_val):
        """計算月週期連線得分"""
        points = 0
        captured = set()
        details = []
        highlight_edges = []
        
        prev_chains = self.get_chains(node_id, current_val, -1, [node_id])
        next_chains = self.get_chains(node_id, current_val, 1, [node_id])
        
        for p_chain in prev_chains:
            for n_chain in next_chains:
                full_path = p_chain[::-1] + [node_id] + n_chain
                if len(full_path) > 2:
                    pts = len(full_path)
                    points += pts
                    captured.update(full_path)
                    details.append(f"月週期({len(full_path)}) +{pts}")
                    
                    for i in range(len(full_path) - 1):
                        edge = tuple(sorted([full_path[i], full_path[i+1]]))
                        highlight_edges.append(edge)
                        
        return points, captured, details, highlight_edges

    def calculate_impact(self, node_id, current_val, player):
        """ 計算放置後的得分與佔領 """
        total_points = 0
        captured_nodes = {node_id}
        self.scored_nodes = {node_id}  # 至少有當前節點
        all_details = []
        
        # A. 基礎配對
        p_points, p_captured, p_details, new_markers = self._calculate_pair_score(node_id, current_val)
        total_points += p_points
        captured_nodes.update(p_captured)
        self.scored_nodes.update(p_captured)
        all_details.extend(p_details)
        self.edge_markers.update(new_markers)

        # B. 週期連線 (全路徑組合)
        c_points, c_captured, c_details, highlight_edges = self._calculate_chain_score(node_id, current_val)
        total_points += c_points
        captured_nodes.update(c_captured)
        self.scored_nodes.update(c_captured)
        all_details.extend(c_details)
        
        # 生成加分邏輯字符串
        if all_details:
            log_msg = " + ".join(all_details) + f" = {total_points} 分"
        else:
            log_msg = "無加分"
        
        self.last_scoring_logs[player] = log_msg
        
        # 只有在有月週期時才更新高亮邊，否則保持當前顯示
        if highlight_edges:
            self.highlighted_edges = highlight_edges
        
        # C. 執行佔領
        if total_points > 0:
            for nid in captured_nodes:
                self.nodes[nid]['owner'] = player
        
        return total_points

    def play_move(self, node_id, card_idx):
        if self.nodes[node_id]['val'] is not None: return False
        
        hand = self.hands[self.turn]
        card_val = hand.pop(card_idx)
        
        self.nodes[node_id]['val'] = card_val
        
        points = self.calculate_impact(node_id, card_val, self.turn)
        self.scores[self.turn] += points
        
        new_card = self._draw(self.turn)
        if new_card is not None:
            hand.insert(card_idx, new_card)
            
        if all(n['val'] is not None for n in self.nodes.values()):
            self.game_over = True
            p1_area = sum(1 for n in self.nodes.values() if n['owner'] == 'P1')
            p2_area = sum(1 for n in self.nodes.values() if n['owner'] == 'P2')
            self.scores['P1'] += p1_area
            self.scores['P2'] += p2_area
            
        if not self.game_over:
            self.turn = 'P2' if self.turn == 'P1' else 'P1'
            
        return True

    def clone(self):
        """複製當前遊戲狀態，用於 AI 模擬"""
        new_game = MoonPhaseGame(self.adj_map)
        new_game.nodes = copy.deepcopy(self.nodes)
        new_game.scores = self.scores.copy()
        new_game.decks = copy.deepcopy(self.decks)
        new_game.hands = copy.deepcopy(self.hands)
        new_game.turn = self.turn
        new_game.game_over = self.game_over
        new_game.highlighted_edges = copy.deepcopy(self.highlighted_edges)
        new_game.edge_markers = copy.deepcopy(self.edge_markers)
        new_game.scored_nodes = self.scored_nodes.copy()
        new_game.last_scoring_logs = self.last_scoring_logs.copy()
        return new_game
