import random
import numpy as np
import copy
import os
from src.game_model import MoonPhaseGame
from src.adj_map import DEFAULT_BOARD

# 嘗試導入 TensorFlow，如果沒有安裝則優雅降級
try:
    from keras._tf_keras.keras import layers, models, optimizers, Input, Model
    import tensorflow as tf
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
    使用 Graph Neural Network (GNN) 架構
    """
    def __init__(self, num_nodes, num_actions, model_path=None):
        self.num_nodes = num_nodes
        self.num_actions = num_actions
        self.model = self.load_or_create_model(model_path)

    def load_or_create_model(self, path):
        if path and os.path.exists(path):
            print(f"Loading model from {path}")
            try:
                # 嘗試載入模型
                # 注意：如果遇到 bad marshal data 錯誤，通常是因為 Python 版本不兼容或 Lambda 函數序列化問題
                # 在這種情況下，最簡單的方法是重建模型並載入權重，而不是載入整個模型結構
                model = models.load_model(path, safe_mode=False)
            except (TypeError, ValueError, Exception) as e:
                print(f"Warning: Failed to load model structure directly ({e}). Rebuilding model and loading weights...")
                # 如果載入失敗，則重新建構模型並嘗試只載入權重
                model = self.build_model()
                try:
                    model.load_weights(path)
                    print("Weights loaded successfully.")
                except Exception as w_e:
                    print(f"Error loading weights: {w_e}")
                    print("Creating new model instead.")
                    return self.build_model()
                
            # 強制重新編譯以啟用 run_eagerly=True
            model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                          loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'},
                          run_eagerly=True)
            return model
        else:
            print("Creating new model")
            return self.build_model()

    def build_model(self):
        # 1. Inputs
        # Node Features: (Batch, N, 12) [9 for val + 3 for owner]
        node_input = Input(shape=(self.num_nodes, 12), name='node_input')
        # Adjacency Matrix: (Batch, N, N)
        adj_input = Input(shape=(self.num_nodes, self.num_nodes), name='adj_input')
        # Hand Features: (Batch, 27) [3 cards * 9]
        hand_input = Input(shape=(27,), name='hand_input')
        # Action Mask: (Batch, Num_Actions)
        mask_input = Input(shape=(self.num_actions,), name='mask_input')

        # 2. Feature Fusion (Broadcasting)
        # Reshape Hand: (Batch, 27) -> (Batch, 1, 27)
        hand_reshaped = layers.Reshape((1, 27))(hand_input)
        # Tile: (Batch, N, 27)
        # Note: Keras Tile layer or Lambda with tf.tile
        # Capture num_nodes in a local variable to avoid capturing 'self' in lambda
        n_nodes = self.num_nodes
        hand_tiled = layers.Lambda(lambda x: tf.tile(x, [1, n_nodes, 1]))(hand_reshaped)
        
        # Concatenate: (Batch, N, 12+27=39)
        x = layers.Concatenate()([node_input, hand_tiled])

        # 3. GNN Backbone (Message Passing)
        # Stack 3 GNN blocks
        for i in range(3):
            # Aggregate: A * X -> (Batch, N, F)
            # Dot axes=(2, 1) means: adj[..., i, k] * x[..., k, j] -> out[..., i, j]
            x_agg = layers.Dot(axes=(2, 1))([adj_input, x])
            
            # Concatenate: (Batch, N, 2F)
            x_combined = layers.Concatenate()([x, x_agg])
            
            # Update: Dense
            x = layers.Dense(256, activation='relu')(x_combined)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # 4. Output Heads
        
        # Policy Head (Node-wise Prediction)
        # We need to predict 3 cards for each node -> 3 logits per node
        # (Batch, N, 3)
        policy_logits_node = layers.Dense(3)(x)
        
        # Flatten to (Batch, N*3) which is (Batch, Num_Actions)
        policy_logits = layers.Flatten()(policy_logits_node)
        
        # Apply Masking
        # valid moves keep logits, invalid moves get -1e9
        # mask is 1.0 for valid, 0.0 for invalid
        # (1 - mask) * -1e9
        inf_mask = layers.Lambda(lambda m: (1.0 - m) * -1e9)(mask_input)
        masked_logits = layers.Add()([policy_logits, inf_mask])
        
        policy_out = layers.Activation('softmax', name='policy')(masked_logits)
        
        # Value Head (Global Graph Prediction)
        # Global Average Pooling: (Batch, N, F) -> (Batch, F)
        global_pool = layers.GlobalAveragePooling1D()(x)
        
        v = layers.Dense(128, activation='relu')(global_pool)
        v = layers.Dropout(0.3)(v)
        value_out = layers.Dense(1, activation='tanh', name='value')(v)
        
        model = Model(inputs=[node_input, adj_input, hand_input, mask_input], 
                      outputs=[policy_out, value_out])
        
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                      loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'},
                      run_eagerly=True)
        return model

    def predict(self, inputs):
        # inputs is a list: [nodes, adj, hand, mask]
        outputs = self.model.predict_on_batch(inputs)
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
        # states is a list of arrays [nodes, adj, hand, mask]
        return self.model.fit(states, targets, **kwargs)


class MoonZeroAdapter:
    """
    負責遊戲狀態與神經網路輸入/輸出之間的轉換
    """
    def __init__(self, num_actions, adj_map=None):
        self.num_actions = num_actions
        self.adj_map = adj_map if adj_map else DEFAULT_BOARD.adj_map
        self.num_nodes = len(self.adj_map)
        self._build_adj_matrix()

    def _build_adj_matrix(self):
        # Create Adjacency Matrix (N, N)
        # Row-normalized with self-loops
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        
        sorted_ids = sorted(self.adj_map.keys())
        id_to_idx = {nid: i for i, nid in enumerate(sorted_ids)}
        
        for nid, neighbors in self.adj_map.items():
            i = id_to_idx[nid]
            # Self-loop
            adj[i, i] = 1.0
            for neighbor in neighbors:
                if neighbor in id_to_idx:
                    j = id_to_idx[neighbor]
                    adj[i, j] = 1.0
        
        # Row Normalization (D^-1 * A)
        row_sums = adj.sum(axis=1, keepdims=True)
        # Avoid division by zero (though self-loops prevent this)
        row_sums[row_sums == 0] = 1.0
        self.adj_matrix = adj / row_sums

    def encode_state(self, game_state, player):
        """
        將遊戲狀態編碼為 4 個輸入向量。
        回傳: [node_features, adj_matrix, hand_features, mask]
        """
        if isinstance(game_state, MoonPhaseGame):
            nodes = game_state.nodes
            hand = game_state.hands[player]
        else:
            nodes = game_state['nodes']
            hand = game_state['hand']
            
        sorted_ids = sorted(nodes.keys())
        
        # 1. Node Features (Batch, N, 12)
        node_features = []
        for nid in sorted_ids:
            data = nodes[nid]
            # Value: 0-7 (one-hot), None -> 8
            val_vec = [0]*9
            if data['val'] is None:
                val_vec[8] = 1
            else:
                val_vec[data['val']] = 1
            
            # Owner: P1, P2, None (one-hot relative to current player)
            # 0: Self, 1: Opponent, 2: None
            owner_vec = [0]*3
            if data['owner'] is None:
                owner_vec[2] = 1
            elif data['owner'] == player:
                owner_vec[0] = 1
            else:
                owner_vec[1] = 1
            
            node_features.append(val_vec + owner_vec)
            
        node_tensor = np.array(node_features, dtype=np.float32).reshape(1, self.num_nodes, 12)
        
        # 2. Adjacency Matrix (Batch, N, N)
        adj_tensor = self.adj_matrix.reshape(1, self.num_nodes, self.num_nodes).astype(np.float32)
        
        # 3. Hand Features (Batch, 27)
        hand_vec = []
        for card in hand:
            c_vec = [0]*9
            if card is None:
                c_vec[8] = 1
            else:
                c_vec[card] = 1
            hand_vec.extend(c_vec)
        
        # Pad hand if less than 3
        for _ in range(3 - len(hand)):
            hand_vec.extend([0]*8 + [1])
            
        hand_tensor = np.array(hand_vec, dtype=np.float32).reshape(1, 27)
        
        # 4. Action Mask (Batch, Num_Actions)
        mask_tensor = self.get_valid_moves_mask(game_state).reshape(1, self.num_actions).astype(np.float32)
        
        return [node_tensor, adj_tensor, hand_tensor, mask_tensor]

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
    def __init__(self, name: str, model_path: str = None, input_dim=None, num_actions=None, adj_map=None):
        super().__init__(name)
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for AlphazeroAI")
            
        self.num_actions = num_actions
        self.adj_map = adj_map if adj_map else DEFAULT_BOARD.adj_map
        self.num_nodes = len(self.adj_map)
        
        # 初始化組件
        self.network = AlphaZeroNetwork(self.num_nodes, num_actions, model_path)
        self.adapter = MoonZeroAdapter(num_actions, self.adj_map)
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
