import networkx as nx

class BoardConfig:
    def __init__(self, name, adj_map, layout_func, node_radius=40, padding=50):
        self.name = name
        self.adj_map = adj_map
        self.layout_func = layout_func
        self.node_radius = node_radius
        self.padding = padding

    def get_layout(self, G):
        return self.layout_func(G)

# --- Map 1: Small (9 nodes) ---
MAP_SMALL = {
    1: [2, 4],
    2: [1, 3, 5],
    3: [2, 6],
    4: [1, 5, 7],
    5: [2, 4, 6, 8],
    6: [3, 5, 9],
    7: [4, 8],
    8: [5, 7, 9], 
    9: [6, 8], 
}

def layout_small(G):
    return nx.spring_layout(G, iterations=200)

# --- Map 2: Medium (14 nodes) ---
MAP_MEDIUM = {
  1: [2, 3, 8],
  2: [1, 4],
  3: [1, 5],
  4: [2, 6, 7],
  5: [3, 9, 10],
  6: [4, 11],
  7: [4, 12, 8],
  8: [1, 7, 9],
  9: [5, 8, 13],
  10: [5, 14],
  11: [6, 12],
  12: [7, 11],
  13: [9, 14],
  14: [10, 13]
}

def layout_medium(G):
    return nx.kamada_kawai_layout(G)

# --- Map 3: Large (24 nodes) ---
MAP_LARGE = {
    1: [2, 8, 9],
    2: [1, 3, 6],
    3: [2, 4, 11],
    4: [3, 5, 8],
    5: [4, 6, 13],
    6: [2, 5, 7],
    7: [6, 8, 15],
    8: [1, 4, 7], 
    9: [10, 16, 1], 
    10: [9, 11, 18],
    11: [10, 12, 3],
    12: [11, 13, 20],
    13: [12, 14, 5],
    14: [13, 15, 22],
    15: [14, 16, 7],
    16: [9, 15 ,24],
    17: [18, 24],
    18: [10, 17, 19],
    19: [18, 20],
    20: [12, 19, 21],
    21: [20, 22],
    22: [14, 21, 23],
    23: [22, 24],
    24: [16, 17, 23]
}

def layout_large(G):
    shells = [
        list(range(1, 9)),   # Inner
        list(range(9, 17)),  # Middle
        list(range(17, 25))  # Outer
    ]
    return nx.shell_layout(G, nlist=shells, rotate=0)

# Exported dictionary
BOARDS = {
    "small": BoardConfig("Small (9 nodes)", MAP_SMALL, layout_small),
    "medium": BoardConfig("Medium (14 nodes)", MAP_MEDIUM, layout_medium),
    "large": BoardConfig("Large (24 nodes)", MAP_LARGE, layout_large),
}

DEFAULT_BOARD = BOARDS["large"]