import math
from PyQt5.QtGui import QColor

# --- 設定與常數 ---
PHASES = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]
PHASE_COLORS = {
    0: "#0974a9", 1: "#4a4a4a", 2: "#6e6e6e", 3: "#a0a0a0",
    4: "#f2f2f2", 5: "#a0a0a0", 6: "#6e6e6e", 7: "#4a4a4a"
}
OWNER_COLORS = {'P1': "#36a066", 'P2': "#e73ca5", None: "#95a5a6"}

# 根據月相 index (0..7) 回傳固定顏色
def phase_to_color(phase_index: int) -> QColor:
    """
    將月相索引映射到固定的顏色。
    phase_index: 0..7
    """
    color_code = PHASE_COLORS.get(phase_index, "#95a5a6")
    return QColor(color_code)
