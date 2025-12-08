import math
from PyQt5.QtGui import QColor

# --- 設定與常數 ---
PHASES = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]
PHASE_COLORS = {
    0: "#0974a9", 1: "#4a4a4a", 2: "#6e6e6e", 3: "#a0a0a0",
    4: "#f2f2f2", 5: "#a0a0a0", 6: "#6e6e6e", 7: "#4a4a4a"
}
OWNER_COLORS = {'P1': "#36a066", 'P2': "#e73ca5", None: "#95a5a6"}

# 根據月相 index (0..7) 回傳藍→紫的循環漸層顏色
def phase_to_color(phase_index: int) -> QColor:
    """
    將月相索引映射到藍色 (約 220°) -> 紫色 (約 270°) 的 HSL 漸層，
    並根據月象循環在 full moon 附近提高亮度。輸出為 QColor。
    phase_index: 0..7
    """
    t = (phase_index % 8) / 7.0
    # Hue 從 220°（藍）到 270°（紫）線性插值
    hue = 220.0 + 50.0 * t
    saturation = 200  # 0..255
    # 讓亮度在滿月（t ~ 0.5）達到峰值（週期性）
    lightness = 150.0 + 80.0 * math.cos(2 * math.pi * (t - 0.5))
    h = int(hue) % 360
    s = max(0, min(255, int(saturation)))
    l = max(0, min(255, int(lightness)))
    return QColor.fromHsl(h, s, l)
