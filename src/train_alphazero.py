import os
import numpy as np
import pickle
from src.game_model import MoonPhaseGame
from src.ai_logic import AlphazeroAI
from src.adj_map import DEFAULT_BOARD

# Configuration
NUM_NODES = len(DEFAULT_BOARD.adj_map)
INPUT_DIM = 12 * NUM_NODES + 27
NUM_ACTIONS = 3 * NUM_NODES

# 設定模型路徑為專案根目錄下的 models 資料夾
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "alphazero_model.h5")
HISTORY_PATH = os.path.join(BASE_DIR, "models", "training_history.pkl")

def _execute_game_step(ai, game, step_count):
    """執行遊戲的一步，回傳 (game_snapshot, policy, player)"""
    # MCTS search
    root = ai.mcts.search(game, game.turn, num_simulations=50)
    
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
    
    training_data = []
    for state, policy, player in game_history:
        if winner == 'Draw':
            value = 0
        elif winner == player:
            value = 1
        else:
            value = -1
        
        state_tensor = ai.encode_state(state, player)
        training_data.append((state_tensor, policy, value))
    
    game_info = {
        'winner': winner,
        'p1_score': s1,
        'p2_score': s2,
        'steps': len(game_history)
    }
    return training_data, game_info

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
            
            result = _execute_game_step(ai, game, step_count)
            if result:
                # 注意：_execute_game_step 回傳的是移動後的狀態，但我們需要移動前的狀態配上 policy
                # 這裡需要修正邏輯：policy 是基於移動前的狀態計算的
                # 所以我們應該儲存 (current_state_snapshot, policy, current_player)
                _, policy, _ = result
                game_history.append([current_state_snapshot, policy, current_player])
                
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
    ai = AlphazeroAI("Trainer", model_path=MODEL_PATH, input_dim=INPUT_DIM, num_actions=NUM_ACTIONS)
    
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
    
    total_epochs = 1000
    for epoch in range(start_epoch, total_epochs):
        print(f"\n=== Epoch {epoch+1}/{total_epochs} ===")
        
        # 1. Self Play
        data, results = self_play(ai, num_games=2, on_move_callback=on_move_callback) # 每次迭代玩 2 場
        game_record.extend(results)
        
        if not data:
            continue
            
        # 2. Prepare Data
        # 強制轉換為 float32 並調整形狀，避免 Keras 類型錯誤
        states = np.vstack([d[0] for d in data]).astype(np.float32)
        policies = np.vstack([d[1] for d in data]).astype(np.float32)
        values = np.array([d[2] for d in data]).astype(np.float32).reshape(-1, 1)
        
        # 3. Train
        print(f"Training on {len(data)} samples...")
        print(f"Data Shapes - States: {states.shape}, Policies: {policies.shape}, Values: {values.shape}")
        
        try:
            history = ai.model.fit(states, {'policy': policies, 'value': values}, epochs=2, batch_size=32, verbose=1)
            
            # Record loss
            epoch_losses = {k: np.mean(v) for k, v in history.history.items()}
            loss_history.append(epoch_losses)
            
        except Exception as e:
            print(f"Error during model.fit: {e}")
            # 嘗試印出更多除錯資訊
            print(f"States dtype: {states.dtype}")
            print(f"Policies dtype: {policies.dtype}")
            print(f"Values dtype: {values.dtype}")
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
