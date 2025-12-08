# 專案更新與除錯報告 (v1.5)

**日期**: 2025年12月9日  
**版本**: 1.5 (GNN Architecture Upgrade)

---

## 1. 重大架構更新：圖神經網路 (GNN)

本版本將 AlphaZero 的核心決策網路從傳統的 MLP/CNN 架構全面升級為 **Graph Neural Network (GNN)**，以適應非網格狀的棋盤拓撲結構。

### 1.1 核心變更 (`src/ai_logic.py`)

*   **多模態輸入 (Multi-modal Inputs)**:
    模型現在接收 4 個獨立的輸入張量，而非單一的扁平化向量：
    1.  **Node Features** `(Batch, N, 12)`: 每個節點的狀態（月相值 + 擁有者）。
    2.  **Adjacency Matrix** `(Batch, N, N)`: 棋盤的連接關係矩陣（包含 Self-loops）。
    3.  **Hand Features** `(Batch, 27)`: 玩家手牌資訊。
    4.  **Action Mask** `(Batch, Num_Actions)`: 合法動作遮罩。

*   **Graph Broadcasting 機制**:
    *   **問題**: 手牌資訊是全域 (Global) 的，但 GNN 是在節點層級 (Local) 運作。
    *   **解法**: 實作了 Broadcasting 機制，將手牌向量複製並串接到每一個節點的特徵向量上，使每個節點在進行訊息傳遞時都能「看見」手牌。

*   **GNN Backbone (Message Passing)**:
    *   使用 `layers.Dot` 實現鄰居聚合 (Aggregation)：$X_{agg} = A \cdot X$。
    *   堆疊了 3 層 GNN Block，每層包含：聚合 -> 串接 -> Dense -> BatchNorm -> Dropout。

### 1.2 訓練流程適配 (`src/train_alphazero.py`)

*   **資料管線重構**:
    *   `_process_game_result` 與 `self_play` 現在會收集並回傳 4 個輸入特徵。
    *   `train` 函數中的 `model.fit` 已更新為接收 `[nodes, adj, hand, mask]` 列表。

---

## 2. 除錯紀錄 (Bug Fixes)

在升級過程中，我們解決了以下關鍵錯誤：

### 2.1 `RecursionError: maximum recursion depth exceeded`
*   **症狀**: 在儲存模型或進行 `deepcopy` 時程式崩潰。
*   **原因**: `AlphaZeroNetwork.build_model` 中的 `Lambda` 層使用了 `lambda x: tf.tile(x, [1, self.num_nodes, 1])`。這裡的 `lambda` 閉包捕獲了 `self` (Network 實例)，而 Network 又持有 Model，Model 又持有 Layer，Layer 又持有 Lambda，形成了 `self -> model -> layer -> lambda -> self` 的循環引用。
*   **修復**: 將 `self.num_nodes` 提取為局部變數 `n_nodes`，讓 lambda 只捕獲整數值。
    ```python
    # Fix
    n_nodes = self.num_nodes
    hand_tiled = layers.Lambda(lambda x: tf.tile(x, [1, n_nodes, 1]))(hand_reshaped)
    ```

### 2.2 `ValueError: bad marshal data`
*   **症狀**: 載入舊模型或跨環境載入時失敗。
*   **原因**: Python 的 `lambda` 函數序列化 (marshal) 不穩定，容易因版本差異導致讀取失敗。
*   **修復**: 實作了強健的載入機制 (`load_or_create_model`)。
    1.  優先嘗試 `load_model(..., safe_mode=False)`。
    2.  若失敗，自動退回「**重建模型結構 + 僅載入權重**」的模式。

### 2.3 `ValueError: Requested the deserialization of a Lambda layer...`
*   **症狀**: Keras 3+ 安全性限制，拒絕載入含 Lambda 的模型。
*   **修復**: 在 `load_model` 中明確傳入 `safe_mode=False`。

### 2.4 Notebook 輸出混亂
*   **症狀**: 訓練進度條重複顯示，日誌難以閱讀。
*   **修復**: 將 `model.fit` 的 `verbose` 設為 `0`，並改用手動 `print` 輸出簡潔的 Loss 資訊。

---

## 3. 下一步計畫

*   **超參數調優**: 目前 GNN 層數為 3，Hidden Units 為 256，可根據訓練收斂速度進行調整。
*   **動態拓撲測試**: 目前雖支援動態拓撲，但訓練仍基於 `DEFAULT_BOARD`。未來可嘗試在訓練中隨機變換 `adj_map` 以增強泛化能力。
*   **資料擴增 (Data Augmentation) 研究**: 
    *   **對稱性擴增 (Symmetry Augmentation)**: 這是 AlphaZero 類算法中極為有效的技巧。若棋盤具有幾何對稱性（如旋轉、鏡像），我們可以在訓練時將一盤棋局變換為多個等價的訓練樣本（例如：旋轉 90/180/270 度後，局面評估值應不變，策略分佈則對應旋轉）。
    *   **效益**: 這能「免費」將訓練數據量擴增 2~8 倍，顯著提升樣本效率 (Sample Efficiency)。
    *   **實作考量**: 需確認目前的 GNN 架構與 Adjacency Matrix 是否容易實作節點編號的置換 (Permutation) 來對應棋盤旋轉。
