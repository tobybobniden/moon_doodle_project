import sys
import time
import os

# --- 關鍵修復：禁用 Qt OpenGL 整合 ---
# 解決 WSL2 + TensorFlow 環境下的 LLVM 版本衝突導致的 Segfault
os.environ['QT_XCB_GL_INTEGRATION'] = 'none'

# --- PyQt5 導入 ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QMessageBox, QFrame)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QTimer

# --- 導入地圖配置 ---
from src.adj_map import BOARDS, DEFAULT_BOARD

# --- 導入遊戲模組 ---
from src.game_utils import PHASES, OWNER_COLORS
from src.game_model import MoonPhaseGame
from src.ai_logic import AIPlayer, RandomAI, GreedyAI
from src.game_view import GameBoardWidget

# --- 3. 主視窗介面 (Controller) ---
class MainWindow(QMainWindow):
    def __init__(self, ai_players: dict = None, ai_delay: float = 0.5, board_config=None):
        """
        初始化遊戲窗口。
        
        參數:
            ai_players: {'P1': AIPlayer實例, 'P2': AIPlayer實例} 或 None
                        如果 P1/P2 沒有 AI，則由人類玩家控制
            ai_delay: AI 移動之間的延遲時間（秒），預設 0.5 秒
            board_config: BoardConfig 實例，如果為 None 則使用 DEFAULT_BOARD
        """
        super().__init__()
        self.setWindowTitle("Moon Phase Strategy (PvP Graph) - PyQt5")
        self.resize(900, 700)
        
        # 使用傳入的地圖配置或預設值
        self.board_config = board_config if board_config else DEFAULT_BOARD
        self.game = MoonPhaseGame(self.board_config.adj_map)
        
        self.selected_card_idx = None
        self.game_ended = False
        self.ai_players = ai_players or {}  # {'P1': AIPlayer, 'P2': AIPlayer}
        self.ai_delay = ai_delay  # AI 延遲時間（秒）
        
        self._init_ui()
        
        # 延遲啟動遊戲邏輯，確保窗口先顯示
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self._start_game)

    def _init_ui(self):
        """初始化 UI 組件"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # 1. 頂部資訊列
        self._init_info_panel(main_layout)
        
        # 2. 棋盤區域
        self.board_widget = GameBoardWidget(self.board_config, self.game)
        self.board_widget.node_clicked.connect(self.on_node_click)
        main_layout.addWidget(self.board_widget, stretch=1)
        
        # 3. 加分邏輯顯示區
        self._init_log_panel(main_layout)
        
        # 4. 手牌與控制區
        self._init_control_panel(main_layout)
        
        # 5. 遊戲結束面板
        self._init_game_over_panel(main_layout)

    def _init_info_panel(self, parent_layout):
        info_layout = QHBoxLayout()
        self.lbl_p1_score = QLabel("P1 Score: 0")
        self.lbl_p2_score = QLabel("P2 Score: 0")
        self.lbl_turn = QLabel("Turn: Player 1")
        
        for lbl in [self.lbl_p1_score, self.lbl_p2_score, self.lbl_turn]:
            lbl.setFont(QFont("WenQuanYi Micro Hei", 14, QFont.Weight.Bold))
            lbl.setFrameStyle(QFrame.Panel | QFrame.Sunken) 
            lbl.setMargin(5)
            
        info_layout.addWidget(self.lbl_p1_score)
        info_layout.addStretch()
        info_layout.addWidget(self.lbl_turn)
        info_layout.addStretch()
        info_layout.addWidget(self.lbl_p2_score)
        parent_layout.addLayout(info_layout)

    def _init_log_panel(self, parent_layout):
        log_layout = QHBoxLayout()
        
        self.lbl_log_p1 = QLabel("P1: 就緒")
        self.lbl_log_p1.setFont(QFont("WenQuanYi Micro Hei", 20))
        self.lbl_log_p1.setStyleSheet("color: #36a066; background-color: #222; padding: 5px; border-radius: 3px;")
        self.lbl_log_p1.setWordWrap(True)
        
        self.lbl_log_p2 = QLabel("P2: 就緒")
        self.lbl_log_p2.setFont(QFont("WenQuanYi Micro Hei", 20))
        self.lbl_log_p2.setStyleSheet("color: #e73ca5; background-color: #222; padding: 5px; border-radius: 3px;")
        self.lbl_log_p2.setWordWrap(True)
        
        log_layout.addWidget(self.lbl_log_p1, stretch=1)
        log_layout.addWidget(self.lbl_log_p2, stretch=1)
        parent_layout.addLayout(log_layout)

    def _init_control_panel(self, parent_layout):
        hand_layout_container = QVBoxLayout()
        self.lbl_hand_msg = QLabel("請選擇一張手牌...")
        self.lbl_hand_msg.setAlignment(Qt.AlignCenter)
        hand_layout_container.addWidget(self.lbl_hand_msg)
        
        self.hand_buttons_layout = QHBoxLayout()
        self.hand_btns = []
        hand_layout_container.addLayout(self.hand_buttons_layout)
        
        self.game_control_container = QWidget()
        self.game_control_container.setLayout(hand_layout_container)
        parent_layout.addWidget(self.game_control_container)

    def _init_game_over_panel(self, parent_layout):
        self.game_over_panel = QVBoxLayout()
        
        self.lbl_result = QLabel()
        self.lbl_result.setFont(QFont("WenQuanYi Micro Hei", 16, QFont.Weight.Bold))
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.game_over_panel.addWidget(self.lbl_result)
        
        btn_retry = QPushButton("重新開始")
        btn_retry.setFont(QFont("WenQuanYi Micro Hei", 12, QFont.Weight.Bold))
        btn_retry.clicked.connect(self.restart_game)
        self.game_over_panel.addWidget(btn_retry)
        
        self.game_over_container = QWidget()
        self.game_over_container.setLayout(self.game_over_panel)
        self.game_over_container.hide()
        parent_layout.addWidget(self.game_over_container)

    def _start_game(self):
        """延遲啟動遊戲邏輯，允許窗口先顯示"""
        self.update_ui()

    def update_ui(self):
        """ 刷新所有介面元素 """
        self._update_info_labels()
        self.board_widget.update()
        
        # 如果當前是 AI 玩家，自動執行移動
        if not self.game_ended and self.game.turn in self.ai_players:
            self.execute_ai_move()
            return
        
        self._update_hand_buttons()

    def _update_info_labels(self):
        """更新分數、回合與日誌標籤"""
        self.lbl_p1_score.setText(f"P1: {self.game.scores['P1']}")
        self.lbl_p2_score.setText(f"P2: {self.game.scores['P2']}")
        self.lbl_turn.setText(f"Current Turn: {self.game.turn}")
        
        self.lbl_log_p1.setText(f"P1: {self.game.last_scoring_logs['P1']}")
        self.lbl_log_p2.setText(f"P2: {self.game.last_scoring_logs['P2']}")
        
        color = OWNER_COLORS[self.game.turn]
        self.lbl_turn.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 18px;")

    def _update_hand_buttons(self):
        """更新手牌按鈕（僅人類玩家）"""
        # 清除舊按鈕
        for btn in self.hand_btns:
            self.hand_buttons_layout.removeWidget(btn)
            btn.deleteLater()
        self.hand_btns = []
        
        current_hand = self.game.hands[self.game.turn]
        color = OWNER_COLORS[self.game.turn]
        
        for idx, card_val in enumerate(current_hand):
            if card_val is None: continue
            
            btn = self._create_hand_button(idx, card_val, color)
            self.hand_buttons_layout.addWidget(btn)
            self.hand_btns.append(btn)

    def _create_hand_button(self, idx, card_val, color):
        btn_text = f"{PHASES[card_val]} ({card_val})"
        btn = QPushButton(btn_text)
        btn.setFixedSize(100, 60)
        btn.setFont(QFont("WenQuanYi Micro Hei", 16))
        
        if self.selected_card_idx == idx:
            btn.setStyleSheet(f"background-color: #f1c40f; border: 3px solid {color};")
        else:
            btn.setStyleSheet("background-color: #ecf0f1;")
            
        btn.clicked.connect(lambda checked, i=idx: self.select_card(i))
        return btn
    
    def execute_ai_move(self):
        """執行 AI 玩家的移動"""
        ai_player = self.ai_players[self.game.turn]
        
        # 準備遊戲狀態
        game_state = {
            'nodes': self.game.nodes,
            'hand': self.game.hands[self.game.turn],
            'scores': self.game.scores,
            'adj_map': self.game.adj_map,
            'game_obj': self.game
        }
        
        # 取得 AI 決策
        move = ai_player.decide_move(game_state, self.game.turn)
        
        if move:
            card_idx, node_id = move
            success = self.game.play_move(node_id, card_idx)
            
            if success:
                # AI 成功落子
                if self.game.game_over:
                    self.show_game_over()
                else:
                    # AI 移動完成後，直接更新 UI
                    # 由於 update_ui 中已經包含了下一次 AI 移動的延遲邏輯，
                    # 這裡不需要額外的 sleep 或 processEvents
                    self.update_ui()
            else:
                pass  # AI 無法落子（通常不會發生）
        else:
            pass  # AI 無法做出決策

    def select_card(self, idx):
        self.selected_card_idx = idx
        self.lbl_hand_msg.setText(f"已選擇: {PHASES[self.game.hands[self.game.turn][idx]]}，請點擊棋盤落子")
        self.update_ui()

    def on_node_click(self, node_id):
        if self.selected_card_idx is None:
            QMessageBox.warning(self, "提示", "請先選擇一張手牌！")
            return
            
        success = self.game.play_move(node_id, self.selected_card_idx)
        
        if success:
            self.selected_card_idx = None
            self.lbl_hand_msg.setText("落子成功！換下一位")
            self.update_ui()
            
            if self.game.game_over:
                self.show_game_over()
        else:
            QMessageBox.warning(self, "無效", "該位置已經有牌了，請選擇空位。")

    def show_game_over(self):
        if self.game_ended: return
        self.game_ended = True
        s1, s2 = self.game.scores['P1'], self.game.scores['P2']
        
        # 計算領地佔有數
        p1_area = sum(1 for n in self.game.nodes.values() if n['owner'] == 'P1')
        p2_area = sum(1 for n in self.game.nodes.values() if n['owner'] == 'P2')
        total_nodes = len(self.game.nodes)
        
        winner = self._determine_winner(s1, s2)
        
        self._print_game_stats(s1, s2, winner, p1_area, p2_area, total_nodes)
        self._update_game_over_ui(s1, s2, winner)

    def _determine_winner(self, s1, s2):
        if s1 > s2: return "Player 1 獲勝！"
        elif s2 > s1: return "Player 2 獲勝！"
        return "平手"

    def _print_game_stats(self, s1, s2, winner, p1_area, p2_area, total_nodes):
        print("\n" + "="*40)
        print(f" 遊戲結束 (Game Over) ")
        print("="*40)
        print(f" 獲勝者: {winner}")
        print("-" * 20)
        print(f" 最終分數 (Final Scores):")
        print(f"   Player 1: {s1}")
        print(f"   Player 2: {s2}")
        print("-" * 20)
        print(f" 領地佔有 (Territory Control):")
        print(f"   Player 1: {p1_area} / {total_nodes} ({p1_area/total_nodes*100:.1f}%)")
        print(f"   Player 2: {p2_area} / {total_nodes} ({p2_area/total_nodes*100:.1f}%)")
        print("="*40 + "\n")

    def _update_game_over_ui(self, s1, s2, winner):
        self.lbl_p1_score.setText(f"P1: {s1}")
        self.lbl_p2_score.setText(f"P2: {s2}")
        
        self.game_control_container.hide()
        
        result_text = f"遊戲結束！\n\nPlayer 1 總分: {s1}\nPlayer 2 總分: {s2}\n\n🏆 {winner}"
        self.lbl_result.setText(result_text)
        self.game_over_container.show()
    
    def restart_game(self):
        """重新開始遊戲"""
        self.game = MoonPhaseGame(self.board_config.adj_map)
        self.selected_card_idx = None
        self.game_ended = False
        
        # 重置 UI
        self.board_widget.game = self.game
        self.lbl_log_p1.setText("P1: 就緒")
        self.lbl_log_p2.setText("P2: 就緒")
        self.game_control_container.show()
        self.game_over_container.hide()
        
        self.update_ui()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 預設：雙 Greedy AI，延遲 0 秒（即時）
    ai_players = {
        'P2': GreedyAI('GreedyBot1')
    }
    
    # 使用 'large' 地圖配置
    # 可選: BOARDS['small'], BOARDS['medium'], BOARDS['large']
    selected_board = BOARDS['large']
    
    window = MainWindow(ai_players=ai_players, ai_delay=0, board_config=selected_board)
    window.show()
    sys.exit(app.exec_())

# ============ AI 配置範例 ============
# 
# 1. 只有人類玩家（預設）
# window = MainWindow()

# 2. P1 是隨機 AI，P2 是人類
# ai_players = {'P1': RandomAI('RandomBot')}
# window = MainWindow(ai_players)

# 3. 雙 AI（Random vs Greedy）
# ai_players = {
#     'P1': RandomAI('RandomBot'),
#     'P2': GreedyAI('GreedyBot')
# }
# window = MainWindow(ai_players)

# 4. 自定義 AI（見下方示例）
# class MyCustomAI(AIPlayer):
#     def decide_move(self, game_state, player):
#         # 自訂邏輯...
#         return (card_idx, node_id)
# 
# ai_players = {'P1': MyCustomAI('MyBot')}
# window = MainWindow(ai_players)

# ===================================
