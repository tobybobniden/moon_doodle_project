import os
import numpy as np
import pickle
from src.game_model import MoonPhaseGame
from src.ai_logic import AlphazeroAI, GreedyAI, RandomAI
from src.adj_map import DEFAULT_BOARD

# Configuration
NUM_NODES = len(DEFAULT_BOARD.adj_map)
INPUT_DIM = 12 * NUM_NODES + 29 # Updated to 29 (27 cards + 2 scores)
NUM_ACTIONS = 3 * NUM_NODES

# 設定模型路徑為專案根目錄下的 models 資料夾
# 優先使用 Linux 原生路徑以提升 I/O 效能 (解決 WSL /mnt/c 寫入慢的問題)
LINUX_HOME_PROJECT = "/home/bruh_bruh/moon_doodle_project"
if os.path.exists(LINUX_HOME_PROJECT):
    BASE_DIR = LINUX_HOME_PROJECT
    print(f"Using Linux native path for models: {BASE_DIR}")
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 確保 models 資料夾存在
MODELS_DIR = os.path.join(BASE_DIR, "models")
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

MODEL_PATH = os.path.join(MODELS_DIR, "alphazero_model.h5")
HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.pkl")

def game_to_dict(game):
    """將 MoonPhaseGame 物件轉換為 AI 需要的字典格式"""
    return {
        'nodes': game.nodes,
        'hand': game.hands[game.turn],
        'scores': game.scores,
        'adj_map': game.adj_map
    }

def get_action_index(card_idx, node_id, game):
    """將 (card_idx, node_id) 轉換為 action_idx"""
    sorted_ids = sorted(game.nodes.keys())
    try:
        node_list_idx = sorted_ids.index(node_id)
        return card_idx * len(sorted_ids) + node_list_idx
    except ValueError:
        return -1

def generate_greedy_data(ai, num_games=10):
    """
    生成 Greedy AI 的對戰數據用於預訓練 (Imitation Learning)
    P1 vs P2: 兩者都是 Greedy AI (50-50 分布)
    這樣可以確保模型對任意玩家位置都有均衡的學習機會
    """
    print(f"Generating {num_games} symmetric games using Greedy AI...")
    training_data = []
    game_results = []
    
    greedy = GreedyAI("Teacher")
    
    for i in range(num_games):
        game = MoonPhaseGame(DEFAULT_BOARD.adj_map)
        game_history = []
        
        step_count = 0
        while not game.game_over:
            step_count += 1
            current_player = game.turn
            current_state_snapshot = game.clone()
            score_before = game.scores[current_player]
            
            # 兩個玩家都使用 Greedy (確保數據的對稱性)
            state_dict = game_to_dict(game)
            move = greedy.decide_move(state_dict, current_player)
            
            if move:
                card_idx, node_id = move
                
                # 構建 Policy Target (One-hot)
                action_idx = get_action_index(card_idx, node_id, game)
                policy = np.zeros(NUM_ACTIONS)
                if action_idx >= 0 and action_idx < NUM_ACTIONS:
                    policy[action_idx] = 1.0
                
                # 執行移動
                game.play_move(node_id, card_idx)
                
                score_after = game.scores[current_player]
                score_gain = score_after - score_before
                
                # 記錄所有決策 (P1 和 P2 都學習 Greedy)
                game_history.append([current_state_snapshot, policy, current_player, score_gain])
            else:
                break
        
        # 處理結果
        t_data, g_info = _process_game_result(game, game_history, ai)
        training_data.extend(t_data)
        game_results.append(g_info)
        
    return training_data, game_results

def train_with_greedy(num_games=100, epochs=5):
    """使用 Greedy AI 進行快速預訓練"""
    print("Initializing AlphaZero AI for Pre-training...")
    ai = AlphazeroAI("Trainer", model_path=MODEL_PATH, input_dim=None, num_actions=NUM_ACTIONS)
    
    if not os.path.exists(MODEL_PATH):
        ai.model.save(MODEL_PATH)
        
    print(f"Starting Pre-training: {num_games} games, {epochs} epochs")
    
    # 1. Generate Data
    data, results = generate_greedy_data(ai, num_games)
    
    if not data:
        print("No data generated.")
        return

    # 2. Prepare Data
    nodes = np.vstack([d[0][0] for d in data]).astype(np.float32)
    adj = np.vstack([d[0][1] for d in data]).astype(np.float32)
    hand = np.vstack([d[0][2] for d in data]).astype(np.float32)
    mask = np.vstack([d[0][3] for d in data]).astype(np.float32)
    
    policies = np.vstack([d[1] for d in data]).astype(np.float32)
    values = np.array([d[2] for d in data]).astype(np.float32).reshape(-1, 1)
    
    # 3. Train
    print(f"Training on {len(data)} samples...")
    ai.model.fit(
        [nodes, adj, hand, mask], 
        {'policy': policies, 'value': values}, 
        epochs=epochs, 
        batch_size=32, 
        verbose=1
    )
    
    # 4. Save
    ai.model.save(MODEL_PATH)
    print(f"Pre-trained model saved to {MODEL_PATH}")

def _execute_game_step(ai, game, step_count):
    """執行遊戲的一步，回傳 (game_snapshot, policy, player)"""
    # MCTS search
    # 增加模擬次數以對抗隨機牌組的變異性 (50 -> 100)
    root = ai.mcts.search(game, game.turn, num_simulations=100)
    
    # Get policy from visit counts
    visits = np.array([child.visit_count for child in root.children.values()])
    actions = list(root.children.keys())
    
    if len(actions) == 0:
        return None
        
    policy = np.zeros(NUM_ACTIONS)
    sum_visits = np.sum(visits)
    
    if sum_visits > 0:
        probs = visits / sum_visits
        for a, p in zip(actions, probs):
            policy[a] = p
    else:
        probs = np.ones(len(actions)) / len(actions)
        for a, p in zip(actions, probs):
            policy[a] = p
    
    # Choose action
    if step_count < 30:
        action_idx = np.random.choice(actions, p=probs)
    else:
        action_idx = actions[np.argmax(visits)]
    
    # Apply move
    move = ai.decode_action(action_idx, game)
    if move:
        card_idx, node_id = move
        game.play_move(node_id, card_idx)
        return (game.clone(), policy, game.turn)
    else:
        print("Error: Invalid move decoded")
        return None

def _process_game_result(game, game_history, ai):
    """處理遊戲結果，生成訓練數據"""
    s1 = game.scores['P1']
    s2 = game.scores['P2']
    if s1 > s2: winner = 'P1'
    elif s2 > s1: winner = 'P2'
    else: winner = 'Draw'
    
    print(f"Game finished. Winner: {winner}, Score: {s1}-{s2}")
    
    # --- 重要修改：使用「相對獎勵」而非「絕對獎勵」---
    # 這解決了先後手位置偏差的問題。
    # 關鍵洞察：同一局遊戲中，贏家和輸家的奮鬥都同樣重要
    # 我們不應該因為"贏了"就給所有步驟+1，"輸了"就給所有步驟-1
    # 而應該給予「相對於對手的努力」的獎勵
    
    score_diff = s1 - s2  # P1 視角的分數差
    normalized_score_reward = np.clip(score_diff / 30.0, -1.0, 1.0)  # 規範化到 [-1, 1]
    
    training_data = []
    # Unpack 4 values now: state, policy, player, immediate_score_gain
    for state, policy, player, score_gain in game_history:
        # === 新獎勵系統：相對獎勵 ===
        # 核心思想：同一局遊戲中，P1 和 P2 的累積獎勵應該互為相反數
        # 這樣兩個玩家都會學到「對手學到了什麼，我就應該學到相反的」
        
        if winner == 'Draw':
            final_value = 0.0
        elif winner == 'P1':
            # P1 贏：P1 獲得正獎勵，P2 獲得負獎勵
            # 獎勵 = 勝利(+1) + 分數優勢(±0.5)
            if player == 'P1':
                final_value = 1.0 + 0.5 * normalized_score_reward
            else:  # player == 'P2'
                final_value = -1.0 - 0.5 * normalized_score_reward
        elif winner == 'P2':
            # P2 贏：P2 獲得正獎勵，P1 獲得負獎勵
            # 這是對稱的：-score_diff 從 P2 視角變成 +score_diff
            if player == 'P2':
                final_value = 1.0 - 0.5 * normalized_score_reward
            else:  # player == 'P1'
                final_value = -1.0 + 0.5 * normalized_score_reward
        
        # 2. 計算即時獎勵 (Short-term Goal) - 相對視角
        # 鼓勵得分，但以雙方均衡的方式
        immediate_value = np.clip(score_gain / 8.0, -1.0, 1.0)
        
        # 3. 混合獎勵 (Hybrid Reward) 
        # 80% 勝負，20% 得分
        value = 0.8 * final_value + 0.2 * immediate_value
        
        state_tensor = ai.encode_state(state, player)
        training_data.append((state_tensor, policy, value))
    
    game_info = {
        'winner': winner,
        'p1_score': s1,
        'p2_score': s2,
        'steps': len(game_history)
    }
    
    # --- 4. 數據增強 (Data Augmentation) ---
    # 利用棋盤的 4-fold 旋轉對稱性擴充數據
    augmented_data = augment_data(training_data, ai.num_nodes)
    
    return augmented_data, game_info

def augment_data(training_data, num_nodes):
    """
    對訓練數據進行 D4 群增強 (8種變換：4旋轉 + 4反射)
    training_data: list of (state_tensor, policy, value)
    state_tensor: [node_tensor, adj_tensor, hand_tensor, mask_tensor]
    """
    augmented = []
    augmented.extend(training_data) # 原始數據
    
    # 定義基本映射
    # Inner: 1-8, Middle: 9-16, Outer: 17-24
    
    def get_rotation_map(times):
        # times=1 (90), 2 (180), 3 (270)
        shift = times * 2
        mapping = np.arange(num_nodes)
        
        # Apply shift to each shell
        mapping[0:8] = np.roll(mapping[0:8], -shift)
        mapping[8:16] = np.roll(mapping[8:16], -shift)
        mapping[16:24] = np.roll(mapping[16:24], -shift)
        return mapping

    def get_flip_map():
        # Reflection across Axis 1-5 (Vertical)
        mapping = np.arange(num_nodes)
        indices = np.arange(8)
        flipped_indices = (8 - indices) % 8
        
        mapping[0:8] = mapping[0:8][flipped_indices]
        mapping[8:16] = mapping[8:16][flipped_indices]
        mapping[16:24] = mapping[16:24][flipped_indices]
        return mapping

    # 生成所有變換映射
    transforms = []
    
    # 1. 純旋轉 (90, 180, 270)
    for k in [1, 2, 3]:
        transforms.append(get_rotation_map(k))
        
    # 2. 反射 (Flip + Rotation 0, 90, 180, 270)
    # 這包含了水平、垂直、以及兩條對角線的反射
    flip_map = get_flip_map()
    for k in [0, 1, 2, 3]:
        rot_map = get_rotation_map(k)
        # 組合映射: M[i] = flip[rot[i]]
        # 代表先 Flip 再 Rotate
        combined_map = flip_map[rot_map]
        transforms.append(combined_map)

    for rot_map in transforms:
        for item in training_data:
            state_inputs, policy, value = item
            node_tensor, adj_tensor, hand_tensor, mask_tensor = state_inputs
            
            # 1. Rotate Node Features (Batch, N, 12)
            new_node_tensor = node_tensor[:, rot_map, :]
            
            # 2. Rotate Adjacency (Batch, N, N)
            new_adj_tensor = adj_tensor[:, rot_map, :][:, :, rot_map]
            
            # 3. Hand Features (Batch, 29) - Invariant
            new_hand_tensor = hand_tensor.copy()
            
            # 4. Rotate Policy (Batch, Num_Actions)
            num_cards = 3
            policy_matrix = policy.reshape(num_cards, num_nodes)
            new_policy_matrix = policy_matrix[:, rot_map]
            new_policy = new_policy_matrix.flatten()
            
            # 5. Rotate Mask (Batch, Num_Actions)
            mask_matrix = mask_tensor.reshape(1, num_cards, num_nodes)
            new_mask_matrix = mask_matrix[:, :, rot_map]
            new_mask_tensor = new_mask_matrix.reshape(1, -1)
            
            augmented.append(([new_node_tensor, new_adj_tensor, new_hand_tensor, new_mask_tensor], new_policy, value))
            
    return augmented

def self_play(ai, num_games=5, on_move_callback=None):
    training_data = []
    game_results = []
    
    for i in range(num_games):
        print(f"Starting self-play game {i+1}/{num_games}")
        game = MoonPhaseGame(DEFAULT_BOARD.adj_map)
        game_history = []
        
        if on_move_callback:
            on_move_callback(game)
        
        step_count = 0
        while not game.game_over:
            step_count += 1
            # 記錄當前狀態用於訓練（在移動前）
            current_state_snapshot = game.clone()
            current_player = game.turn
            # 記錄移動前的分數
            score_before = game.scores[current_player]
            
            result = _execute_game_step(ai, game, step_count)
            if result:
                # result: (game_after_move, policy, next_player)
                # 注意：game 物件已經被修改了，所以 game.scores 已經是新的分數
                score_after = game.scores[current_player]
                score_gain = score_after - score_before
                
                _, policy, _ = result
                # 儲存: [state, policy, player, score_gain]
                game_history.append([current_state_snapshot, policy, current_player, score_gain])
                
                if on_move_callback:
                    on_move_callback(game)
            else:
                break
            
        t_data, g_info = _process_game_result(game, game_history, ai)
        training_data.extend(t_data)
        game_results.append(g_info)
            
    return training_data, game_results

def train(on_move_callback=None):
    print("Initializing AlphaZero AI...")
    # input_dim is no longer used but kept for signature compatibility if needed, or we can remove it.
    # We pass None for input_dim as it's not used in the new GNN architecture.
    ai = AlphazeroAI("Trainer", model_path=MODEL_PATH, input_dim=None, num_actions=NUM_ACTIONS)
    
    # 如果模型不存在，先編譯並儲存一次
    if not os.path.exists(MODEL_PATH):
        ai.model.save(MODEL_PATH)
    
    # Load history
    start_epoch = 0
    loss_history = []
    game_record = []
    
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, 'rb') as f:
                data = pickle.load(f)
                # Assuming format: [epoch, loss_history, game_record]
                if len(data) >= 3:
                    start_epoch = data[0]
                    loss_history = data[1]
                    game_record = data[2]
                    print(f"Loaded history: starting from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load history: {e}")
    
    data_buffer = [] # Buffer for experience replay (last 3 iterations)

    total_epochs = 1000
    for epoch in range(start_epoch, total_epochs):
        print(f"\n=== Epoch {epoch+1}/{total_epochs} ===")
        
        # 1. Self Play
        data, results = self_play(ai, num_games=5, on_move_callback=on_move_callback) # 每次迭代玩 5 場
        game_record.extend(results)
        
        if not data:
            continue
            
        # Update Buffer (Sliding Window of size 3)
        data_buffer.append(data)
        if len(data_buffer) > 3:
            data_buffer.pop(0)
            
        # Combine data
        training_data = []
        for batch in data_buffer:
            training_data.extend(batch)
            
        # 2. Prepare Data
        # data[i] = (state_inputs, policy, value)
        # state_inputs = [node_tensor, adj_tensor, hand_tensor, mask_tensor]
        
        nodes = np.vstack([d[0][0] for d in training_data]).astype(np.float32)
        adj = np.vstack([d[0][1] for d in training_data]).astype(np.float32)
        hand = np.vstack([d[0][2] for d in training_data]).astype(np.float32)
        mask = np.vstack([d[0][3] for d in training_data]).astype(np.float32)
        
        policies = np.vstack([d[1] for d in training_data]).astype(np.float32)
        values = np.array([d[2] for d in training_data]).astype(np.float32).reshape(-1, 1)
        
        # 3. Train
        print(f"Training on {len(training_data)} samples (Buffer: {len(data_buffer)} batches)...")
        # print(f"Data Shapes - States: {states.shape}, Policies: {policies.shape}, Values: {values.shape}")
        
        try:
            # Pass list of inputs for GNN
            # verbose=0 to suppress progress bar spam in notebooks
            # 增加 epochs (3 -> 5) 讓模型有更多機會擬合複雜的隨機牌組策略
            history = ai.model.fit(
                [nodes, adj, hand, mask], 
                {'policy': policies, 'value': values}, 
                epochs=5, 
                batch_size=32, 
                verbose=1
            )
            
            # Record loss
            epoch_losses = {k: np.mean(v) for k, v in history.history.items()}
            loss_history.append(epoch_losses)
            
            # Manual print for cleaner output
            print(f"  > Training Loss: {epoch_losses['loss']:.4f} (Policy: {epoch_losses['policy_loss']:.4f}, Value: {epoch_losses['value_loss']:.4f})")
            
        except Exception as e:
            print(f"Error during model.fit: {e}")
            raise e
        
        # 4. Save
        ai.model.save(MODEL_PATH)
        
        # 5. Save History
        with open(HISTORY_PATH, 'wb') as f:
            pickle.dump([epoch + 1, loss_history, game_record], f)
            
        print(f"Model saved to {MODEL_PATH}")
        print(f"History saved to {HISTORY_PATH}")

if __name__ == "__main__":
    train()
