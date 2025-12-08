import random
import numpy as np
import copy
import os
from src.game_model import MoonPhaseGame

# 嘗試導入 TensorFlow，如果沒有安裝則優雅降級
try:
    from keras._tf_keras.keras import layers, models, optimizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow/Keras not found. AlphazeroAI will not work.")

# --- AI 控制框架 ---
class AIPlayer:
# ... (保留原有的 AIPlayer, RandomAI, GreedyAI) ...

    """
    AI 玩家基類。繼承此類並實現 decide_move() 來建立自己的 AI。
    
    使用方式：
    1. 建立子類並覆寫 decide_move()
    2. 傳入 MainWindow 的 ai_players 參數
    3. AI 會自動在其回合時執行
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def decide_move(self, game_state: dict, player: str) -> tuple:
        """
        決定一步棋。
        
        參數:
            game_state: {
                'nodes': {node_id: {'val': ..., 'owner': ...}},
                'hand': [card_val1, card_val2, card_val3],
                'scores': {'P1': score1, 'P2': score2},
                'adj_map': {...}
            }
            player: 'P1' 或 'P2'
        
        回傳:
            (card_idx, node_id) - 選擇第 card_idx 張牌，放在 node_id
            或 None 表示無法移動
        """
        raise NotImplementedError("子類必須實現 decide_move()")


class RandomAI(AIPlayer):
    """隨機 AI - 隨機選擇手牌和節點"""
    
    def decide_move(self, game_state: dict, player: str) -> tuple:
        hand = game_state['hand']
        nodes = game_state['nodes']
        
        # 找出所有空位
        empty_nodes = [nid for nid, data in nodes.items() if data['val'] is None]
        
        if not empty_nodes or not hand:
            return None
        
        card_idx = random.randint(0, len(hand) - 1)
        node_id = random.choice(empty_nodes)
        return (card_idx, node_id)


class GreedyAI(AIPlayer):
    """貪心 AI - 選擇能獲得最多分數的移動"""
    
    def decide_move(self, game_state: dict, player: str) -> tuple:
        hand = game_state['hand']
        nodes = game_state['nodes']
        adj_map = game_state['adj_map']
        
        empty_nodes = [nid for nid, data in nodes.items() if data['val'] is None]
        
        if not empty_nodes or not hand:
            return None
        
        best_move = None
        best_score = -1
        
        # 嘗試所有組合，找出最高分的
        for card_idx, card_val in enumerate(hand):
            for node_id in empty_nodes:
                # 簡單評分：根據相鄰節點計算潛在得分
                score = self._evaluate_move(node_id, card_val, nodes, adj_map)
                
                if score > best_score:
                    best_score = score
                    best_move = (card_idx, node_id)
        
        return best_move
    
    def _evaluate_move(self, node_id: int, card_val: int, nodes: dict, adj_map: dict) -> int:
        """評估在某個位置放置某張牌的分數"""
        score = 0
        
        for neighbor_id in adj_map.get(node_id, []):
            neighbor = nodes[neighbor_id]
            if neighbor['val'] is None:
                continue
            
            # 相鄰相同或互補加分
            if abs(card_val - neighbor['val']) == 4:
                score += 2
            elif card_val == neighbor['val']:
                score += 1
        
        return score


class AlphaZeroNetwork:
    """
    負責神經網路的建構、載入與預測
    """
    def __init__(self, input_dim, num_actions, model_path=None):
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.model = self.load_or_create_model(model_path)

    def load_or_create_model(self, path):
        if path and os.path.exists(path):
            print(f"Loading model from {path}")
            model = models.load_model(path)
            # 強制重新編譯以啟用 run_eagerly=True
            model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                          loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'},
                          run_eagerly=True)
            return model
        else:
            print("Creating new model")
            return self.build_model()

    def build_model(self):
        input_layer = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(512, activation='relu')(input_layer)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        policy = layers.Dense(self.num_actions, activation='softmax', name='policy')(x)
        value = layers.Dense(1, activation='tanh', name='value')(x)
        
        model = models.Model(inputs=input_layer, outputs=[policy, value])
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                      loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'},
                      run_eagerly=True)
        return model

    def predict(self, state_tensor):
        outputs = self.model.predict_on_batch(state_tensor)
        if hasattr(outputs[0], 'numpy'):
            policy_logits = outputs[0].numpy()
            value = outputs[1].numpy()
        else:
            policy_logits = outputs[0]
            value = outputs[1]
        return policy_logits, value

    def save(self, path):
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def fit(self, states, targets, **kwargs):
        return self.model.fit(states, targets, **kwargs)


class MoonZeroAdapter:
    """
    負責遊戲狀態與神經網路輸入/輸出之間的轉換
    """
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def encode_state(self, game_state, player):
        """
        將遊戲狀態編碼為向量。
        支援 game_state 為 dict 或 MoonPhaseGame 物件。
        """
        if isinstance(game_state, MoonPhaseGame):
            nodes = game_state.nodes
            hand = game_state.hands[player]
        else:
            nodes = game_state['nodes']
            hand = game_state['hand']
            
        features = []
        sorted_ids = sorted(nodes.keys())
        
        # 1. Node Features
        for nid in sorted_ids:
            data = nodes[nid]
            # Value: 0-7 (one-hot), None -> 8
            val_vec = [0]*9
            if data['val'] is None:
                val_vec[8] = 1
            else:
                val_vec[data['val']] = 1
            features.extend(val_vec)
            
            # Owner: P1, P2, None (one-hot relative to current player)
            # 0: Self, 1: Opponent, 2: None
            owner_vec = [0]*3
            if data['owner'] is None:
                owner_vec[2] = 1
            elif data['owner'] == player:
                owner_vec[0] = 1
            else:
                owner_vec[1] = 1
            features.extend(owner_vec)
            
        # 2. Hand Features (3 cards, each 0-7 or None)
        # One-hot 9 dim per card slot
        for card in hand:
            c_vec = [0]*9
            if card is None:
                c_vec[8] = 1
            else:
                c_vec[card] = 1
            features.extend(c_vec)
        
        # Pad hand if less than 3
        for _ in range(3 - len(hand)):
            features.extend([0]*8 + [1])

        return np.array(features).reshape(1, -1)

    def decode_action(self, action_idx, game_state):
        """
        Action Index = card_idx * num_nodes + node_id
        """
        if isinstance(game_state, MoonPhaseGame):
            nodes = game_state.nodes
        else:
            nodes = game_state['nodes']
            
        num_nodes = len(nodes)
        sorted_ids = sorted(nodes.keys())
        
        card_idx = action_idx // num_nodes
        node_list_idx = action_idx % num_nodes
        
        if node_list_idx < len(sorted_ids):
            node_id = sorted_ids[node_list_idx]
            return (card_idx, node_id)
        return None

    def get_valid_moves_mask(self, game_state):
        """回傳一個 mask 向量，標示哪些 action 是合法的"""
        if isinstance(game_state, MoonPhaseGame):
            nodes = game_state.nodes
            hand = game_state.hands[game_state.turn]
        else:
            nodes = game_state['nodes']
            hand = game_state['hand']
            
        num_nodes = len(nodes)
        sorted_ids = sorted(nodes.keys())
        
        mask = np.zeros(self.num_actions)
        
        empty_indices = [i for i, nid in enumerate(sorted_ids) if nodes[nid]['val'] is None]
        
        for c_idx in range(len(hand)):
            if hand[c_idx] is not None:
                for n_idx in empty_indices:
                    action_idx = c_idx * num_nodes + n_idx
                    if action_idx < self.num_actions:
                        mask[action_idx] = 1
        return mask


class MCTSNode:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}  # {action_idx: MCTSNode}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        
    @property
    def value(self):
        return self.value_sum / (self.visit_count + 1e-6)

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    def __init__(self, network, adapter):
        self.network = network
        self.adapter = adapter
        
    def search(self, game_state, player, num_simulations):
        root = MCTSNode()
        
        # 準備初始模擬狀態
        if isinstance(game_state, dict) and 'game_obj' in game_state:
            initial_game = game_state['game_obj'].clone()
        elif isinstance(game_state, MoonPhaseGame):
            initial_game = game_state.clone()
        else:
            print("Warning: MCTS requires 'game_obj' in game_state for accurate simulation.")
            return root
            
        for _ in range(num_simulations):
            node = root
            sim_game = initial_game.clone()
            
            # 1. Selection
            while not node.is_leaf():
                action_idx, node = self._select_child(node)
                move = self.adapter.decode_action(action_idx, sim_game)
                if move:
                    card_idx, node_id = move
                    sim_game.play_move(node_id, card_idx)
            
            # 2. Expansion & Evaluation
            if sim_game.game_over:
                s1 = sim_game.scores['P1']
                s2 = sim_game.scores['P2']
                if s1 > s2: winner = 'P1'
                elif s2 > s1: winner = 'P2'
                else: winner = 'Draw'
                
                if winner == 'Draw': value = 0
                else: value = 1 if winner == player else -1
            else:
                # Predict with NN
                state_tensor = self.adapter.encode_state(sim_game, sim_game.turn)
                policy_logits, value = self.network.predict(state_tensor)
                
                policy_probs = policy_logits[0]
                value = value[0][0]
                
                # Mask invalid moves
                valid_mask = self.adapter.get_valid_moves_mask(sim_game)
                policy_probs = policy_probs * valid_mask
                sum_probs = np.sum(policy_probs)
                if sum_probs > 0:
                    policy_probs /= sum_probs
                else:
                    policy_probs = valid_mask / np.sum(valid_mask)
                
                # Expand
                for action_idx, prob in enumerate(policy_probs):
                    if prob > 0:
                        node.children[action_idx] = MCTSNode(parent=node, prior=prob)
            
            # 3. Backup
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                node = node.parent
                value = -value # 對手視角反轉
                
        return root

    def _select_child(self, node):
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        for action, child in node.children.items():
            ucb = child.value + 1.0 * child.prior * np.sqrt(node.visit_count) / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child
        
        return best_action, best_child

    def select_action(self, root, temperature=1.0):
        if not root.children:
            return np.random.choice(range(self.adapter.num_actions))

        visits = [child.visit_count for child in root.children.values()]
        actions = list(root.children.keys())
        
        if temperature == 0:
            return actions[np.argmax(visits)]
        
        visits = np.array(visits)
        probs = visits ** (1 / temperature)
        probs /= np.sum(probs)
        return np.random.choice(actions, p=probs)


class AlphazeroAI(AIPlayer):
    """
    AlphaZero Agent
    協調 Network, Adapter 和 MCTS 進行決策
    """
    def __init__(self, name: str, model_path: str = None, input_dim=None, num_actions=None):
        super().__init__(name)
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for AlphazeroAI")
            
        self.input_dim = input_dim
        self.num_actions = num_actions
        
        # 初始化組件
        self.network = AlphaZeroNetwork(input_dim, num_actions, model_path)
        self.adapter = MoonZeroAdapter(num_actions)
        self.mcts = MCTS(self.network, self.adapter)
        
        # 為了相容性，保留 model 屬性指向 network.model
        self.model = self.network.model

    def decide_move(self, game_state: dict, player: str, training=False) -> tuple:
        # 1. 執行 MCTS
        root = self.mcts.search(game_state, player, num_simulations=50 if training else 200)
        
        # 2. 根據訪問次數選擇動作
        temperature = 1.0 if training else 0.1
        action_idx = self.mcts.select_action(root, temperature)
        
        # 3. 解碼動作
        return self.adapter.decode_action(action_idx, game_state)

    # 為了相容性，保留這些方法作為 Adapter 的代理
    def encode_state(self, game_state, player):
        return self.adapter.encode_state(game_state, player)
        
    def decode_action(self, action_idx, game_state):
        return self.adapter.decode_action(action_idx, game_state)
        
    def get_valid_moves_mask(self, game_state):
        return self.adapter.get_valid_moves_mask(game_state)
