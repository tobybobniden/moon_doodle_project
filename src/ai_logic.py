import os
# ⚠️ 必須在任何 TensorFlow 導入之前設定！
# TensorFlow 2.15 在 CC 12.0 GPU 上有 PTX 編譯相容性問題
# 模型建構時禁用 GPU，訓練時會自動使用 GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  <-- 已註解，因為我們現在使用 tf-nightly 支援 GPU

import random
import numpy as np
import copy
import logging
import tensorflow as tf
from src.game_model import MoonPhaseGame
from src.adj_map import DEFAULT_BOARD
# 嘗試導入 TensorFlow，如果沒有安裝則優雅降級
try:
    from keras import layers, models, optimizers, Input, Model
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TensorFlow/Keras not found. StudentOfGamesAI will not work.")

# --- Logger Setup ---
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

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
        """評估在某個位置放置某張牌的分數 (包括月週期)"""
        score = 0
        
        # === 部分 1：相鄰配對 (Pair Bonuses) ===
        for neighbor_id in adj_map.get(node_id, []):
            neighbor = nodes[neighbor_id]
            if neighbor['val'] is None:
                continue
            
            # 相鄰相同或互補加分
            if abs(card_val - neighbor['val']) == 4:
                score += 2
            elif card_val == neighbor['val']:
                score += 1
        
        # === 部分 2：月週期檢查 (Chain Detection) ===
        # 月週期分為兩個方向：順向 (card_val+1, +2, ...) 和逆向 (card_val-1, -2, ...)
        chains_forward = self._find_chains(node_id, card_val, 1, nodes, adj_map)
        chains_backward = self._find_chains(node_id, card_val, -1, nodes, adj_map)
        
        # 月週期評分：每條連續序列長度 >= 2 時計分
        for chain in chains_forward + chains_backward:
            if len(chain) >= 2:
                # 連續序列長度計分 (例如 3 個連續 = +3)
                score += len(chain)
        
        return score
    
    def _find_chains(self, start_node, current_val, step, nodes, adj_map):
        """
        DFS 尋找特定方向的連續月相序列
        
        參數:
            start_node: 起始節點
            current_val: 當前月相值 (0-7)
            step: 步長 (+1 順向, -1 逆向)
            nodes: 遊戲節點字典
            adj_map: 鄰接關係
        
        回傳:
            chains: 找到的所有序列列表，每個序列是節點 ID 列表
        """
        valid_chains = []
        visited = set()
        
        # BFS 找出所有連續節點
        queue = [(start_node, [start_node], current_val)]
        
        while queue:
            node, path, val = queue.pop(0)
            
            # 計算下一個月相值
            next_val = (val + step) % 8
            
            # 尋找鄰接點中是否有下一個月相值
            found_next = False
            for neighbor in adj_map.get(node, []):
                if neighbor not in visited and nodes[neighbor]['val'] == next_val:
                    found_next = True
                    new_path = path + [neighbor]
                    visited.add(neighbor)
                    queue.append((neighbor, new_path, next_val))
            
            # 如果沒有找到下一個，這個路徑是一條完整的序列
            if not found_next and len(path) > 1:
                valid_chains.append(path)
        
        return valid_chains


class SOGNetwork:
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
                    # Rebuild case: Must compile
                    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                                  loss={'policy': 'categorical_crossentropy', 'value': tf.keras.losses.Huber()},
                                  run_eagerly=False)
                except Exception as w_e:
                    print(f"Error loading weights: {w_e}")
                    print("Creating new model instead.")
                    return self.build_model()
                
            # If load_model succeeded, we assume it loaded the optimizer state too.
            # We do NOT re-compile here to avoid resetting the optimizer.
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
        # Hand Features: (Batch, 29) [3 cards * 9 + 2 scores]
        hand_input = Input(shape=(29,), name='hand_input')
        # Action Mask: (Batch, Num_Actions)
        mask_input = Input(shape=(self.num_actions,), name='mask_input')

        # 2. Feature Fusion (Broadcasting)
        # Reshape Hand: (Batch, 29) -> (Batch, 1, 29)
        hand_reshaped = layers.Reshape((1, 29))(hand_input)
        # Tile: (Batch, N, 29)
        # Note: Keras Tile layer or Lambda with tf.tile
        # Capture num_nodes in a local variable to avoid capturing 'self' in lambda
        n_nodes = self.num_nodes
        hand_tiled = layers.Lambda(lambda x: tf.tile(x, [1, n_nodes, 1]))(hand_reshaped)
        
        # Concatenate: (Batch, N, 12+29=41)
        x = layers.Concatenate()([node_input, hand_tiled])

        # 3. GNN Backbone (Message Passing)
        # Stack 8 GNN blocks (Updated from 3 to 8 as per SOG report)
        for i in range(8):
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
                      loss={'policy': 'categorical_crossentropy', 'value': tf.keras.losses.Huber()},
                      run_eagerly=False)
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


class SOGAdapter:
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
        
        # 3. Hand Features (Batch, 29) -> 27 cards + 2 scores
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
            
        # --- Fix: Add Scores to Global Features ---
        # Normalize scores (assuming max score ~50 for normalization stability)
        if isinstance(game_state, MoonPhaseGame):
            s1 = game_state.scores.get('P1', 0) / 50.0
            s2 = game_state.scores.get('P2', 0) / 50.0
            # Ensure relative perspective: [My Score, Opponent Score]
            if player == 'P1':
                hand_vec.extend([s1, s2])
            else:
                hand_vec.extend([s2, s1])
        else:
            # Fallback for dict state
            scores = game_state.get('scores', {'P1': 0, 'P2': 0})
            s1 = scores.get('P1', 0) / 50.0
            s2 = scores.get('P2', 0) / 50.0
            if player == 'P1':
                hand_vec.extend([s1, s2])
            else:
                hand_vec.extend([s2, s1])

        hand_tensor = np.array(hand_vec, dtype=np.float32).reshape(1, 29)
        
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


class SOGNode:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}  # {action_idx: SOGNode}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        
        # SOG / CFR specific
        self.regret_sum = 0.0
        self.strategy_sum = 0.0
        self.regrets = {} # {action_idx: cumulative_regret}
        self.policy_sum = {} # {action_idx: cumulative_policy}

    @property
    def value(self):
        return self.value_sum / (self.visit_count + 1e-6)

    def is_leaf(self):
        return len(self.children) == 0


class SOGSearch:
    """
    Student of Games (SOG) Search Implementation
    Uses a simplified Growing-Tree CFR (GT-CFR) approach.
    """
    def __init__(self, network, adapter):
        self.network = network
        self.adapter = adapter
        
    def search(self, game_state, player, num_simulations):
        root = SOGNode()
        
        # Prepare initial state
        if isinstance(game_state, dict) and 'game_obj' in game_state:
            initial_game = game_state['game_obj'].clone()
        elif isinstance(game_state, MoonPhaseGame):
            initial_game = game_state.clone()
        else:
            print("Warning: SOG requires 'game_obj' in game_state.")
            return root
            
        for _ in range(num_simulations):
            self._cfr_iteration(root, initial_game.clone(), player)
                
        return root

    def _cfr_iteration(self, node, game, root_player):
        # 1. Terminal State Check
        if game.game_over:
            s1 = game.scores['P1']
            s2 = game.scores['P2']
            if s1 > s2: winner = 'P1'
            elif s2 > s1: winner = 'P2'
            else: winner = 'Draw'
            
            if winner == 'Draw': return 0
            # Return value from the perspective of the current node's player
            return 1 if winner == game.turn else -1

        # 2. Expansion (if leaf)
        if node.is_leaf():
            # Predict with NN
            state_tensor = self.adapter.encode_state(game, game.turn)
            policy_logits, value = self.network.predict(state_tensor)
            
            policy_probs = policy_logits[0]
            value = value[0][0] # Value for current player
            
            # Mask invalid moves
            valid_mask = self.adapter.get_valid_moves_mask(game)
            policy_probs = policy_probs * valid_mask
            sum_probs = np.sum(policy_probs)
            if sum_probs > 0:
                policy_probs /= sum_probs
            else:
                policy_probs = valid_mask / np.sum(valid_mask)
            
            # Expand children
            for action_idx, prob in enumerate(policy_probs):
                if prob > 0:
                    node.children[action_idx] = SOGNode(parent=node, prior=prob)
                    node.regrets[action_idx] = 0.0
                    node.policy_sum[action_idx] = 0.0
            
            return value

        # 3. Regret Matching (Selection)
        # Calculate current strategy from regrets
        actions = list(node.children.keys())
        
        # Regret Matching+: max(R, 0)
        regrets = np.array([max(node.regrets[a], 0) for a in actions])
        sum_regrets = np.sum(regrets)
        
        if sum_regrets > 0:
            strategy = regrets / sum_regrets
        else:
            # If no positive regrets, use uniform or prior
            strategy = np.array([node.children[a].prior for a in actions])
            sum_prior = np.sum(strategy)
            if sum_prior > 0: strategy /= sum_prior
            else: strategy = np.ones(len(actions)) / len(actions)
            
        # Sample action based on strategy
        action_idx = np.random.choice(actions, p=strategy)
        child = node.children[action_idx]
        
        # 4. Recursive Call
        move = self.adapter.decode_action(action_idx, game)
        if move:
            card_idx, node_id = move
            game.play_move(node_id, card_idx)
            
        # Value returned is from the perspective of the NEXT player (opponent)
        # So we negate it to get value for CURRENT player
        child_value = -self._cfr_iteration(child, game, root_player)
        
        # 5. Update Regrets & Strategy Sum
        # Update visit count and value sum (MCTS-like stats for debugging/fallback)
        child.visit_count += 1
        child.value_sum += child_value
        
        # Calculate V(s) = sum(sigma(a) * Q(s, a))
        # We need Q(s, a) for all actions to update regrets.
        # Since we only explored one action, we use the child's value estimate as Q(s, a).
        # For unvisited children, we can use their value_sum/visit_count or 0.
        
        q_values = {}
        for a in actions:
            child_node = node.children[a]
            if child_node.visit_count > 0:
                q_values[a] = child_node.value
            else:
                # For unvisited nodes, assume 0 or parent value? 
                # Let's use 0 for zero-sum game as neutral.
                q_values[a] = 0.0
        
        # Override the Q-value for the action we just took with the fresh sample?
        # Or just use the accumulated average which is more stable.
        # Let's use the accumulated average (child.value) which includes the latest sample.
        
        node_value = sum(strategy[i] * q_values[a] for i, a in enumerate(actions))
        
        for i, a in enumerate(actions):
            # Regret Update: R(s, a) += Q(s, a) - V(s)
            r = q_values[a] - node_value
            node.regrets[a] += r
            
            # Update cumulative policy (for average strategy)
            # We weight by iteration t (here just +1)
            node.policy_sum[a] += strategy[i]
            
        return child_value

    def select_action(self, root, temperature=1.0):
        if not root.children:
            return np.random.choice(range(self.adapter.num_actions))

        # In SOG/CFR, we select based on the Average Strategy (Policy Sum)
        actions = list(root.children.keys())
        policy_sums = np.array([root.policy_sum[a] for a in actions])
        
        sum_policy = np.sum(policy_sums)
        if sum_policy > 0:
            probs = policy_sums / sum_policy
        else:
            probs = np.ones(len(actions)) / len(actions)
            
        if temperature == 0:
            return actions[np.argmax(probs)]
        
        if temperature != 1.0:
            probs = probs ** (1 / temperature)
            probs /= np.sum(probs)
            
        return np.random.choice(actions, p=probs)


class StudentOfGamesAI(AIPlayer):
    """
    Student of Games (SOG) Agent
    協調 Network, Adapter 和 Search Algorithm 進行決策
    """
    def __init__(self, name: str, model_path: str = None, input_dim=None, num_actions=None, adj_map=None):
        super().__init__(name)
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for StudentOfGamesAI")
            
        self.num_actions = num_actions
        self.adj_map = adj_map if adj_map else DEFAULT_BOARD.adj_map
        self.num_nodes = len(self.adj_map)
        
        # 初始化組件
        self.network = SOGNetwork(self.num_nodes, num_actions, model_path)
        self.adapter = SOGAdapter(num_actions, self.adj_map)
        
        # 使用 SOG Search
        self.search_algo = SOGSearch(self.network, self.adapter)
        
        # 為了相容性，保留 model 屬性指向 network.model
        self.model = self.network.model

    def decide_move(self, game_state: dict, player: str, training=False) -> tuple:
        # 1. 執行 Search (SOG/CFR)
        # 增加模擬次數，因為 CFR 需要更多迭代來收斂策略
        root = self.search_algo.search(game_state, player, num_simulations=200 if training else 400)
        
        # 2. 根據平均策略選擇動作
        temperature = 1.0 if training else 0.1
        action_idx = self.search_algo.select_action(root, temperature)
        
        # 3. 解碼動作
        return self.adapter.decode_action(action_idx, game_state)

    # 為了相容性，保留這些方法作為 Adapter 的代理
    def encode_state(self, game_state, player):
        return self.adapter.encode_state(game_state, player)
        
    def decode_action(self, action_idx, game_state):
        return self.adapter.decode_action(action_idx, game_state)
        
    def get_valid_moves_mask(self, game_state):
        return self.adapter.get_valid_moves_mask(game_state)
