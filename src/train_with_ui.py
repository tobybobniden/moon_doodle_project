import sys
import time
import copy
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import QThread, pyqtSignal, Qt

from src.New_moon_game import MainWindow
from src.train_alphazero import train
from src.game_model import MoonPhaseGame

class TrainingThread(QThread):
    game_updated = pyqtSignal(object)  # Signal to emit game state (MoonPhaseGame object)

    def run(self):
        # Define callback to emit signal
        def on_move(game):
            # We need to emit a copy or ensure thread safety
            # Since we are just visualizing, a deepcopy is safest but slow
            # Let's try emitting the game object directly, but be careful not to modify it in GUI
            # Actually, deepcopy is better to avoid race conditions if GUI reads while AI writes
            self.game_updated.emit(game.clone())

        train(on_move_callback=on_move)

class TrainingWindow(MainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlphaZero Training Visualization")
        
        # Add a label to indicate training status
        self.lbl_status = QLabel("Training in progress...")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold; padding: 5px;")
        
        # Insert status label at the top
        central_widget = self.centralWidget()
        layout = central_widget.layout()
        layout.insertWidget(0, self.lbl_status)
        
        # Disable interaction
        self.board_widget.setEnabled(False)
        self.game_control_container.hide() # Hide hand buttons
        
        # Start training thread
        self.training_thread = TrainingThread()
        self.training_thread.game_updated.connect(self.on_game_update)
        self.training_thread.start()

    def on_game_update(self, game_state):
        # Update the game state in the GUI
        self.game = game_state
        self.board_widget.game = self.game
        
        # Update UI elements
        self.lbl_p1_score.setText(f"P1 Score: {self.game.scores['P1']}")
        self.lbl_p2_score.setText(f"P2 Score: {self.game.scores['P2']}")
        self.lbl_turn.setText(f"Turn: {'Player 1' if self.game.turn == 'P1' else 'Player 2'}")
        
        self.lbl_log_p1.setText(f"P1: {self.game.last_scoring_logs['P1']}")
        self.lbl_log_p2.setText(f"P2: {self.game.last_scoring_logs['P2']}")
        
        # Force repaint
        self.board_widget.update()
        
        if self.game.game_over:
             self.lbl_status.setText("Game Finished. Starting next game...")
        else:
             self.lbl_status.setText(f"Training... Turn: {self.game.turn}")

    def closeEvent(self, event):
        # Stop thread on close
        if self.training_thread.isRunning():
            self.training_thread.terminate() # Force kill for now, better to have a stop flag
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainingWindow()
    window.show()
    sys.exit(app.exec_())
