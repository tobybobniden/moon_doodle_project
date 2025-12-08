# 月相棋盤遊戲 - 完整文檔

## 目錄
1. [遊戲概述](#遊戲概述)
2. [AI 控制系統](#ai-控制系統)
3. [AI 延遲功能](#ai-延遲功能)
4. [連接線標記系統](#連接線標記系統)
5. [系統架構](#系統架構)
6. [快速參考](#快速參考)

---

## 遊戲概述

### 基本規則
- **月相值**：0-7，表示 8 個月相（🌑 🌒 🌓 🌔 🌕 🌖 🌗 🌘）
- **棋盤**：14 個連接的節點組成的圖形結構
- **玩家**：2 名玩家輪流落子
- **得分方式**：
  - **同象配對** (相同月相相鄰)：+1 分
  - **互補配對** (相差 4 的月相相鄰)：+2 分
  - **月週期** (連續序列)：每個節點獲得加分

### 遊戲流程
1. 玩家選擇手中的一張牌
2. 玩家在棋盤上的空位放置該牌
3. 系統計算得分並高亮相關連接線
4. 回合結束，切換到另一位玩家
5. 重複直到棋盤填滿

---

## AI 控制系統

### 快速開始

#### 1. 使用預設 AI

```python
from New_moon_game import *

# 雙 AI：隨機 AI vs 貪心 AI
ai_players = {
    'P1': RandomAI('RandomBot'),
    'P2': GreedyAI('GreedyBot')
}
window = MainWindow(ai_players, ai_delay=0.5)
```

#### 2. 自定義 AI（最重要！）

所有 AI 都必須繼承 `AIPlayer` 類並實現 `decide_move()` 方法：

```python
from New_moon_game import AIPlayer

class MyCustomAI(AIPlayer):
    def decide_move(self, game_state: dict, player: str) -> tuple:
        """
        決定一步棋。
        
        參數:
            game_state: 遊戲狀態字典
                {
                    'nodes': {node_id: {'val': 月相值(0-7), 'owner': 'P1'/'P2'/None}, ...},
                    'hand': [card_val1, card_val2, card_val3],
                    'scores': {'P1': score1, 'P2': score2},
                    'adj_map': {node_id: [neighbor1, neighbor2, ...], ...}
                }
            player: 'P1' 或 'P2'
        
        回傳:
            (card_idx, node_id) - 選擇第 card_idx 張牌（0-2），放在 node_id
            None - 無法做出合法移動（通常不會發生）
        """
        # 你的 AI 邏輯
        ...
        return (card_idx, node_id)
```

### 預設 AI 實現

#### RandomAI（隨機 AI）
```python
class RandomAI(AIPlayer):
    def decide_move(self, game_state: dict, player: str) -> tuple:
        # ... (省略實作細節)
        return (card_idx, node_id)
```

#### GreedyAI（貪心 AI）
```python
class GreedyAI(AIPlayer):
    def decide_move(self, game_state: dict, player: str) -> tuple:
        # ... (省略實作細節)
        return best_move
```

#### AlphaZeroAI (強化學習 AI)
基於 AlphaGo Zero 架構的實作，包含以下組件：
- **AlphaZeroNetwork**: 負責神經網路 (TensorFlow/Keras) 的預測與訓練。
- **MCTS (Monte Carlo Tree Search)**: 負責搜尋最佳落子策略。
- **MoonZeroAdapter**: 負責遊戲狀態與神經網路輸入之間的轉換。

```python
from src.ai_logic import AlphazeroAI

# 載入訓練好的模型
ai = AlphazeroAI('AlphaZero', model_path='models/alphazero_model.h5', 
                 input_dim=..., num_actions=...)
```

### 進階 AI 範例

#### 3. 連接 ML 模型

```python
import torch
import torch.nn as nn

class NeuralNetAI(AIPlayer):
    def __init__(self, name: str, model_path: str):
        super().__init__(name)
        self.model = torch.load(model_path)
        self.model.eval()
    
    def decide_move(self, game_state: dict, player: str) -> tuple:
        state_tensor = self._encode_state(game_state, player)
        
        with torch.no_grad():
            output = self.model(state_tensor)
        
        card_idx, node_id = self._decode_output(output, game_state)
        return (card_idx, node_id)
    
    def _encode_state(self, game_state, player):
        # 將遊戲狀態編碼為神經網絡輸入
        pass
    
    def _decode_output(self, output, game_state):
        # 從模型輸出解析移動
        pass
```

#### 4. 使用強化學習（DQN）

```python
import numpy as np

class DQN_AI(AIPlayer):
    def __init__(self, name: str, q_network):
        super().__init__(name)
        self.q_network = q_network
    
    def decide_move(self, game_state: dict, player: str) -> tuple:
        legal_actions = self._get_legal_actions(game_state)
        
        best_action = None
        best_q_value = -np.inf
        
        for card_idx, node_id in legal_actions:
            state_action = self._encode_state_action(game_state, card_idx, node_id)
            q_value = self.q_network(state_action)
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = (card_idx, node_id)
        
        return best_action
    
    def _get_legal_actions(self, game_state):
        hand = game_state['hand']
        nodes = game_state['nodes']
        empty_nodes = [nid for nid, data in nodes.items() if data['val'] is None]
        
        return [(i, nid) for i in range(len(hand)) for nid in empty_nodes]
```

### 遊戲狀態詳解

#### game_state['nodes']
```python
{
    1: {'val': 3, 'owner': 'P1'},      # 節點 1：月相值 3，P1 佔有
    2: {'val': None, 'owner': None},   # 節點 2：空位
    3: {'val': 5, 'owner': 'P2'},      # 節點 3：月相值 5，P2 佔有
    ...
}
```

#### game_state['hand']
```python
[2, 4, 7]  # 三張手牌，月相值分別為 2, 4, 7
```

#### game_state['scores']
```python
{'P1': 15, 'P2': 12}  # P1 得 15 分，P2 得 12 分
```

#### game_state['adj_map']
```python
{
    1: [2, 3, 8],
    2: [1, 4],
    ...
}
# 表示節點連接關係
```

### 完整使用範例

```python
from New_moon_game import *
import sys
from PyQt5.QtWidgets import QApplication

# 定義自己的 AI
class SmartAI(AIPlayer):
    def decide_move(self, game_state, player):
        hand = game_state['hand']
        nodes = game_state['nodes']
        adj_map = game_state['adj_map']
        
        best_move = None
        best_score = 0
        
        for card_idx, card_val in enumerate(hand):
            for node_id in [n for n, d in nodes.items() if d['val'] is None]:
                score = 0
                for neighbor in adj_map.get(node_id, []):
                    neighbor_val = nodes[neighbor]['val']
                    if neighbor_val is not None:
                        if abs(card_val - neighbor_val) == 4:
                            score += 2
                        elif card_val == neighbor_val:
                            score += 1
                
                if score > best_score:
                    best_score = score
                    best_move = (card_idx, node_id)
        
        return best_move or (0, list(nodes.keys())[0])

# 啟動遊戲
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    ai_players = {
        'P1': SmartAI('SmartBot'),
        'P2': GreedyAI('GreedyBot')
    }
    
    window = MainWindow(ai_players, ai_delay=0.5)
    window.show()
    sys.exit(app.exec_())
```

### 調試技巧

**1. 列印遊戲狀態**
```python
print(f"Hand: {game_state['hand']}")
print(f"Available nodes: {[n for n, d in game_state['nodes'].items() if d['val'] is None]}")
```

**2. 驗證移動合法性**
```python
empty_nodes = [nid for nid, data in nodes.items() if data['val'] is None]
assert node_id in empty_nodes, f"Node {node_id} is not empty!"
assert 0 <= card_idx <= 2, f"Invalid card index {card_idx}!"
```

### 注意事項

- `decide_move()` 必須回傳 `(card_idx, node_id)` 的元組
- `card_idx` 必須在 0-2 之間
- `node_id` 必須是棋盤上的合法節點，且該節點必須是空的
- 不要在 AI 中修改 `game_state`，它是參考傳遞的
- 避免在 `decide_move()` 中進行耗時計算（會卡住 UI）

### 常見問題

**Q: 我的 AI 模型需要輸入什麼格式？**
A: 你完全可以自由選擇。只需在 `decide_move()` 中將 `game_state` 轉換為你的模型期望的格式即可。

**Q: 可以讓 AI 之間互相對戰嗎？**
A: 當然！設置 `ai_players = {'P1': AIPlayer1, 'P2': AIPlayer2}` 即可。

**Q: 如何保存和加載訓練好的 AI？**
A: 保存模型在你的 `decide_move()` 中使用它。標準做法是在 `__init__` 時載入模型。

**Q: AI 反應太慢怎麼辦？**
A: 簡化決策邏輯，或使用快速推理方法（如 ONNX Runtime）。

---

## AI 延遲功能

### 概述
在雙 AI 模式下，可以為 AI 移動添加延遲，使遊戲進度變慢，便於觀察。

### 基本用法

#### 1. 無延遲（快速執行）
```python
ai_players = {
    'P1': GreedyAI('Bot1'),
    'P2': GreedyAI('Bot2')
}
window = MainWindow(ai_players, ai_delay=0)
```

#### 2. 有延遲（預設 0.5 秒）
```python
ai_players = {
    'P1': RandomAI('RandomBot'),
    'P2': GreedyAI('GreedyBot')
}
window = MainWindow(ai_players, ai_delay=0.5)  # 每個 AI 移動延遲 0.5 秒
```

#### 3. 自訂延遲時間
```python
# 1 秒延遲
window = MainWindow(ai_players, ai_delay=1.0)

# 2 秒延遲（觀察較為複雜的移動）
window = MainWindow(ai_players, ai_delay=2.0)
```

### 參數說明

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `ai_players` | dict | None | AI 玩家配置 |
| `ai_delay` | float | 0.5 | 每次 AI 移動間的延遲時間（秒） |

### 使用場景

- **快速演示** (`ai_delay=0`): 快速看完整場遊戲
- **正常觀察** (`ai_delay=0.5`): 可以清楚看到每一步的結果
- **詳細分析** (`ai_delay=1.0-2.0`): 仔細分析 AI 的決策過程

### 實現細節

延遲會在：
1. AI 做出決策並執行落子後
2. 遊戲尚未結束的情況下
3. 在重繪棋盤之前

應用延遲，使 UI 有足夠時間更新，玩家能清楚看到每一步的變化。

### 使用範例

運行預設配置（雙 Greedy AI，0.5 秒延遲）：
```bash
python New_moon_game.py
```

修改延遲時間（編輯 `__main__` 區塊）：
```python
window = MainWindow(ai_players, ai_delay=1.5)  # 改為 1.5 秒
```

---

## 連接線標記系統

### 概述
在得分時，連接線上會顯示視覺標記，表示該連接的類型和得分方式。

### 標記類型

#### 1. 實心點 ●（月相組合）
- **觸發條件**: 當兩個相同月相的節點相鄰時
- **得分**: +1 分
- **外觀**: 連接線上顯示一個實心金色點

#### 2. 兩個空心點 ○ ○（滿月組合）
- **觸發條件**: 當兩個互補月相（相差4）的節點相鄰時
- **得分**: +2 分
- **外觀**: 連接線上顯示兩個空心金色點（一左一右）

#### 3. 無標記（月週期）
- **觸發條件**: 當形成連續的月相序列時
- **得分**: 依序列長度（例如5個節點連線 = +5 分）
- **外觀**: 連接線高亮但無額外標記

### 視覺效果

```
得分示例1：同象組合
月相8 ——● —— 月相8    (+1 分)

得分示例2：互補組合
月相1 ——○○—— 月相5    (+2 分)

得分示例3：月週期
月相0 —— 月相1 —— 月相2 —— 月相3 —— 月相4    (+5 分)
```

### 遊戲顯示

- 所有得分連線用**金色**高亮顯示
- 標記點位置在連接線的中點處
- 同時獲得多個分數時，所有連線同時高亮

### 功能流程

1. 玩家落子後，系統計算可得分的連接
2. 連接線根據類型標記（●或○○）
3. 金色高亮連線顯示直到下一個玩家的移動
4. 遊戲結束時清除所有高亮

### 實現細節

系統在 `MoonPhaseGame` 中維護 `edge_markers` 字典：
```python
self.edge_markers = {
    (u, v): 'pair',        # 月相組合（實心點）
    (v, w): 'complement',  # 互補組合（兩個空心點）
    (x, y): 'chain'        # 月週期（無標記）
}
```

在 `GameBoardWidget.paintEvent()` 中繪製標記：
- 實心點：填充圓形，半徑 6px
- 空心點：描邊圓形，半徑 6px，線寬 2px，間距 20px

---

## 系統架構

### MVC 架構設計

```
┌─────────────────────────────────────┐
│     MainWindow (Controller)          │
│  - 遊戲流程管理                      │
│  - 事件處理（點擊、選牌）            │
│  - UI 更新邏輯                       │
└────────────┬────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌──────────────┐  ┌──────────────────┐
│MoonPhaseGame │  │GameBoardWidget   │
│   (Model)    │  │    (View)        │
│- 遊戲狀態    │  │- 棋盤繪製        │
│- 計分邏輯    │  │- 節點渲染        │
│- 移動驗證    │  │- 連線視覺化      │
└──────────────┘  │- 標記點繪製      │
                  └──────────────────┘
```

### AI 框架

```
┌─────────────────────┐
│    AIPlayer         │
│   (Abstract Base)   │
│                     │
│ + decide_move()     │
└──────────┬──────────┘
           │
    ┌──────┴──────┬──────────┐
    │             │          │
    ▼             ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│RandomAI  │ │GreedyAI  │ │CustomAI  │
│- 隨機    │ │- 貪心    │ │- 自定義  │
│ 決策     │ │ 決策     │ │ 邏輯     │
└──────────┘ └──────────┘ └──────────┘
```

### 檔案結構

```
moon_doodle_project/
├── Moon_Game_Launcher.ipynb      # 專案啟動器 (Jupyter Notebook)
├── README.md                     # 統一文檔（本檔案）
├── models/                       # 存放訓練好的模型
│   └── alphazero_model.h5
└── src/                          # 原始碼目錄
    ├── __init__.py
    ├── New_moon_game.py          # 遊戲主程式與 UI
    ├── ai_logic.py               # AI 邏輯 (AlphaZero, MCTS, Greedy, Random)
    ├── game_model.py             # 遊戲核心邏輯
    ├── game_view.py              # 遊戲視圖元件
    ├── train_alphazero.py        # AlphaZero 訓練腳本
    └── train_with_ui.py          # 帶 UI 的訓練腳本
```

### 關鍵類別

**MoonPhaseGame (src.game_model)**
- 管理遊戲狀態和邏輯
- 負責計分、移動驗證、得分計算
- 記錄高亮邊和邊標記

**GameBoardWidget (src.game_view)**
- 使用 QPainter 繪製棋盤
- 處理節點點擊事件
- 繪製連線、節點、標記點

**MainWindow (src.New_moon_game)**
- 協調遊戲流程
- 處理 AI 自動執行
- 管理 UI 組件更新

**AIPlayer (src.ai_logic)**
- 所有 AI 必須繼承
- 定義 `decide_move()` 介面
- 接收遊戲狀態，返回移動決策

---

## 快速參考

### 啟動遊戲

推薦使用 `Moon_Game_Launcher.ipynb` 來啟動遊戲或訓練。

若要使用命令行啟動：

```bash
# 確保在 moon_doodle_project 目錄下
python -m src.New_moon_game
```

### 程式碼範例

```python
from src.New_moon_game import *
from src.ai_logic import GreedyAI, RandomAI
import sys
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)

# 配置 AI
ai_players = {
    'P1': GreedyAI('Bot1'),
    'P2': RandomAI('Bot2')
}

# 啟動窗口（ai_delay 單位：秒）
window = MainWindow(ai_players, ai_delay=0.5)
window.show()
sys.exit(app.exec_())
```

### 常用命令

| 操作 | 方式 |
|------|------|
| 選擇手牌 | 點擊下方的牌 |
| 落子 | 點擊棋盤的空位 |
| 重新開始 | 遊戲結束後點擊「重新開始」按鈕 |

### 得分顯示

- 頂部顯示 P1/P2 的當前分數
- 中央高亮連線和標記點
- 下方顯示得分計算說明（如「互補 +2 + 月週期(5) +5 = 7 分」）

### 配色參考

| 元素 | 顏色 | 用途 |
|------|------|------|
| 背景 | #2b2b2b | 深灰色背景 |
| 普通連線 | #bdc3c7 | 灰色 |
| 得分連線 | #FFD700 | 金色高亮 |
| 標記點 | #FFD700 | 金色點 |
| P1 Halo | #36a066 | 綠色框 |
| P2 Halo | #e73ca5 | 粉紅框 |

### 遊戲配置

| 參數 | 預設值 | 說明 |
|------|--------|------|
| 節點數 | 14 | 棋盤節點數 |
| 月相值範圍 | 0-7 | 8 個月相 |
| 手牌數 | 3 | 每位玩家的手牌數 |
| 牌庫大小 | 50 | 抽牌池大小 |
| AI 延遲 | 0.5s | 預設延遲時間 |

---

## 更新歷史

### 版本 1.0 - 核心功能
- ✅ 基礎遊戲邏輯和 UI
- ✅ AI 控制框架
- ✅ 計分系統

### 版本 1.1 - 視覺增強
- ✅ 高亮連接線
- ✅ 粗體加分說明
- ✅ 月相色彩漸層

### 版本 1.2 - 交互優化
- ✅ AI 延遲功能
- ✅ 連接線標記系統（實心點/空心點）
- ✅ 改進的 UI 佈局

### 版本 1.3 - 架構重構與 AlphaZero 強化
- ✅ 專案結構標準化 (src/models 分離)
- ✅ AlphaZero 架構重構 (Network/Adapter/MCTS 分離)
- ✅ 效能優化 (啟用 Eager Execution, predict_on_batch)
- ✅ 新增 Jupyter Notebook 啟動器

### 版本 1.4 - 程式碼品質與訓練紀錄
- ✅ **降低循環複雜度**：重構核心檔案 (`train_alphazero.py`, `game_model.py`, `game_view.py`, `New_moon_game.py`)，將大型函數拆解為小型私有方法，提升可讀性與維護性。
- ✅ **訓練歷程紀錄**：`train_alphazero.py` 新增 `training_history.pkl` 機制，自動儲存 Loss 曲線與對戰結果，並支援斷點續訓。
- ✅ **數據分析工具**：更新 `Moon_Game_Launcher.ipynb`，新增訓練數據視覺化功能，可直接繪製 Loss 趨勢圖與勝率變化曲線。

---

## 許可証與支持

本項目為教育和研究用途。歡迎修改和擴展！

有任何問題或建議，請參考各模塊的註釋或測試檔案。

