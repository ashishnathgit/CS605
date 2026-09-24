"""
Weighted A* on the graph from the assignment figure.

    f(n) = w1 * g(n) + w2 * h(n)

    (w1, w2) = (1, 0)  ->  Uniform-Cost Search / Dijkstra   (blind)
    (w1, w2) = (1, 1)  ->  A*                               (optimal if h admissible)
    (w1, w2) = (1, w)  ->  Weighted A*                      (cost <= w * optimal)
    (w1, w2) = (0, 1)  ->  Greedy Best-First                (fast, not optimal)

    grid calculation (0,0)


"""



import heapq
from itertools import count
from math import hypot

# --------------------------------------------------------------------------
# 1. THE GRAPH  (undirected -- edges added both ways below)
#    >>> VERIFY THESE AGAINST YOUR HANDOUT <<<
# --------------------------------------------------------------------------
EDGES = [
    ("S", "B", 32), ("S", "E", 27),
    ("B", "C", 26), ("B", "I", 33),
    ("C", "A", 40), ("C", "J", 21),
    ("A", "D", 26),
    ("D", "F", 37),
    ("E", "H", 30), ("E", "N", 36),
    ("F", "K", 12), ("F", "M", 23),
    ("H", "L", 30),
    ("J", "L", 22),
    ("K", "M", 34), ("K", "P", 22),
    ("L", "R", 31),
    ("M", "P", 35), ("M", "R", 35),
    ("N", "Q", 28),
    ("O", "Q", 23), ("O", "R", 34),
    ("P", "G", 27),
    ("R", "G", 42),
]

GRAPH = {}
for u, v, w in EDGES:
    GRAPH.setdefault(u, []).append((v, w))
    GRAPH.setdefault(v, []).append((u, w))

# --------------------------------------------------------------------------
# 2. THE HEURISTIC h(n)
#    The figure gives no h-table, but it IS drawn on a grid, so we read off
#    (column, row) positions and use straight-line distance.
#    Raw SLD would overestimate, so we auto-scale by the smallest
#    (edge weight / edge length) ratio -- that makes h admissible by
#    construction for this layout.
# --------------------------------------------------------------------------
COORDS = {
    "S": (0.5, 0.0), "A": (6.0, 0.0),
    "B": (2.0, 1.0), "C": (4.0, 1.0), "D": (7.0, 1.0), "E": (0.5, 1.5),
    "H": (1.5, 2.5), "I": (3.0, 2.5), "J": (4.0, 2.5),
    "F": (6.0, 2.5), "K": (7.0, 2.5),
    "L": (3.0, 3.5), "M": (5.0, 3.5),
    "N": (0.5, 4.5), "O": (2.0, 4.5), "P": (7.0, 4.5),
    "Q": (1.5, 5.0), "R": (4.0, 5.0), "G": (8.0, 5.0),
}


def _euclid(a, b):
    (x1, y1), (x2, y2) = COORDS[a], COORDS[b]
    return hypot(x1 - x2, y1 - y2)


def make_sld_heuristic(goal):
    """Straight-line distance to `goal`, scaled to stay admissible."""
    scale = min(w / _euclid(u, v) for u, v, w in EDGES if _euclid(u, v) > 0)
    return lambda n: scale * _euclid(n, goal)


def zero_heuristic(goal):
    """h(n) = 0 for all n -- turns A* into Uniform-Cost Search."""
    return lambda n: 0.0


# --------------------------------------------------------------------------
# 3. WEIGHTED A*
# --------------------------------------------------------------------------
def weighted_a_star(start, goal, w1=1.0, w2=1.0, heuristic=None):
    """Returns (path, path_cost, nodes_expanded)."""
    h = heuristic if heuristic is not None else make_sld_heuristic(goal)
    tie = count()                      # stable tie-break, avoids comparing str

    g_score = {start: 0.0}
    parent = {start: None}
    open_list = [(w1 * 0.0 + w2 * h(start), next(tie), start)]
    closed = set()
    expanded = 0

    while open_list:
        f, _, node = heapq.heappop(open_list)
        if node in closed:             # lazy deletion of stale entries
            continue
        closed.add(node)
        expanded += 1

        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1], g_score[goal], expanded

        for nbr, cost in GRAPH.get(node, []):
            if nbr in closed:
                continue
            tentative_g = g_score[node] + cost
            if tentative_g < g_score.get(nbr, float("inf")):
                g_score[nbr] = tentative_g
                parent[nbr] = node
                f_new = w1 * tentative_g + w2 * h(nbr)
                heapq.heappush(open_list, (f_new, next(tie), nbr))

    return None, float("inf"), expanded


# --------------------------------------------------------------------------
# 4. DEMO: sweep the weights and watch the trade-off
# --------------------------------------------------------------------------
if __name__ == "__main__":
    START, GOAL = "S", "G"
    hfun = make_sld_heuristic(GOAL)

    settings = [
        ("Uniform-Cost (blind)", 1.0, 0.0),
        ("A*                  ", 1.0, 1.0),
        ("Weighted A* (w=1.5) ", 1.0, 1.5),
        ("Weighted A* (w=3)   ", 1.0, 3.0),
        ("Greedy Best-First   ", 0.0, 1.0),
    ]

    print(f"Search from {START} to {GOAL}\n")
    print(f"{'algorithm':<22}{'w1':>5}{'w2':>6}{'cost':>8}{'expanded':>10}  path")
    print("-" * 78)
    for name, w1, w2 in settings:
        path, cost, exp = weighted_a_star(START, GOAL, w1, w2, hfun)
        route = " -> ".join(path) if path else "no path"
        print(f"{name:<22}{w1:>5.1f}{w2:>6.1f}{cost:>8.0f}{exp:>10}  {route}")