import math
import networkx as nx
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QRect

from src.game_utils import PHASES, phase_to_color, OWNER_COLORS

# --- 2. 視覺化組件 (View - Board) ---
class GameBoardWidget(QWidget):
    node_clicked = pyqtSignal(int)

    def __init__(self, board_config, game_instance):
        super().__init__()
        self.game = game_instance
        self.board_config = board_config
        self.adj_map = board_config.adj_map
        self.setMinimumSize(500, 500)
        
        G = nx.Graph(self.adj_map)
        self.pos = board_config.get_layout(G)
        self.node_radius = board_config.node_radius
        self.padding = board_config.padding

    def get_screen_coords(self, nx_coords):
        w, h = self.width() - 2*self.padding, self.height() - 2*self.padding
        x = int((nx_coords[0] + 1) / 2 * w + self.padding)
        y = int((-nx_coords[1] + 1) / 2 * h + self.padding)
        return QPoint(x, y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 繪製背景
        painter.fillRect(self.rect(), QColor("#2b2b2b"))
        
        self._draw_connections(painter)
        self._draw_highlights(painter)
        self._draw_markers(painter)
        self._draw_nodes(painter)

    def _draw_connections(self, painter):
        # 先繪製普通連線（從節點邊界到邊界，避免被圓形覆蓋）
        normal_pen = QPen(QColor("#bdc3c7"), 6)
        normal_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(normal_pen)
        drawn_edges = set()
        for u, neighbors in self.adj_map.items():
            u_pos = self.get_screen_coords(self.pos[u])
            for v in neighbors:
                edge = tuple(sorted((u, v)))
                if edge not in drawn_edges:
                    # 跳過高亮邊，稍後繪製
                    if edge not in self.game.highlighted_edges:
                        v_pos = self.get_screen_coords(self.pos[v])
                        self._draw_line_between_nodes(painter, u_pos, v_pos)
                    drawn_edges.add(edge)

    def _draw_highlights(self, painter):
        # 後繪製高亮的月週期連線（更粗、更亮的顏色）
        if self.game.highlighted_edges:
            highlight_color = QColor("#FFD700")  # 金色
            highlight_pen = QPen(highlight_color, 10)
            highlight_pen.setCapStyle(Qt.RoundCap)
            painter.setPen(highlight_pen)
            
            for edge in self.game.highlighted_edges:
                u, v = edge
                u_pos = self.get_screen_coords(self.pos[u])
                v_pos = self.get_screen_coords(self.pos[v])
                self._draw_line_between_nodes(painter, u_pos, v_pos)

    def _draw_line_between_nodes(self, painter, u_pos, v_pos):
        dx = v_pos.x() - u_pos.x()
        dy = v_pos.y() - u_pos.y()
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 0:
            dx /= dist
            dy /= dist
            line_start = QPoint(int(u_pos.x() + dx * self.node_radius), 
                              int(u_pos.y() + dy * self.node_radius))
            line_end = QPoint(int(v_pos.x() - dx * self.node_radius), 
                            int(v_pos.y() - dy * self.node_radius))
            painter.drawLine(line_start, line_end)

    def _draw_markers(self, painter):
        # 繪製所有邊上的標記點（配對/互補標記）
        marker_radius = 6
        for edge, marker_type in self.game.edge_markers.items():
            u, v = edge
            u_pos = self.get_screen_coords(self.pos[u])
            v_pos = self.get_screen_coords(self.pos[v])
            
            mid_x = (u_pos.x() + v_pos.x()) // 2
            mid_y = (u_pos.y() + v_pos.y()) // 2
            mid_point = QPoint(mid_x, mid_y)
            
            if marker_type == 'pair':
                painter.setBrush(QBrush(QColor("#FFD700")))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(mid_point, marker_radius, marker_radius)
            
            elif marker_type == 'complement':
                circle_pen = QPen(QColor("#FFD700"), 2)
                circle_pen.setCapStyle(Qt.RoundCap)
                painter.setPen(circle_pen)
                painter.setBrush(Qt.NoBrush)
                
                dx = v_pos.x() - u_pos.x()
                dy = v_pos.y() - u_pos.y()
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 0:
                    perp_x = -dy / dist * 10
                    perp_y = dx / dist * 10
                    
                    left_point = QPoint(int(mid_x + perp_x), int(mid_y + perp_y))
                    painter.drawEllipse(left_point, marker_radius, marker_radius)
                    
                    right_point = QPoint(int(mid_x - perp_x), int(mid_y - perp_y))
                    painter.drawEllipse(right_point, marker_radius, marker_radius)

    def _draw_nodes(self, painter):
        font = QFont("WenQuanYi Micro Hei", 16, QFont.Weight.Bold)
        painter.setFont(font)
        
        for nid, data in self.game.nodes.items():
            center = self.get_screen_coords(self.pos[nid])
            val = data['val']
            owner = data['owner']
            
            # A. 擁有者外框 (Halo)
            owner_color = QColor(OWNER_COLORS[owner])
            painter.setBrush(Qt.NoBrush) # PyQt5: NoBrush
            halo_pen = QPen(owner_color, 6)
            painter.setPen(halo_pen)
            painter.drawEllipse(center, self.node_radius + 2, self.node_radius + 2)
            
            # B. 月相本體（統一使用素色背景）
            bg_color = QColor("#ecf0f1")
            
            if val is not None:
                text = PHASES[val]
                text_color = Qt.black
            else:
                text = str(nid)
                text_color = Qt.gray # PyQt5: Qt.gray
                
            painter.setBrush(QBrush(bg_color))
            painter.setPen(Qt.NoPen) # PyQt5: NoPen
            painter.drawEllipse(center, self.node_radius, self.node_radius)
            
            # C. 文字
            painter.setPen(QColor(text_color))
            rect =  QRect(center.x()-30, center.y()-30, 60, 60)
            painter.drawText(rect, Qt.AlignCenter, text) # PyQt5: Qt.AlignCenter

    def mousePressEvent(self, event):
        # 簡單的點擊偵測 (距離判斷)
        click_pos = event.pos()
        for nid, nx_coords in self.pos.items():
            center = self.get_screen_coords(nx_coords)
            dist = math.sqrt((click_pos.x() - center.x())**2 + (click_pos.y() - center.y())**2)
            if dist <= self.node_radius + 5:
                self.node_clicked.emit(nid)
                break
