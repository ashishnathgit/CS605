"""
CS605 Lab 7 (1) - MENACE: Matchbox Educable Noughts And Crosses Engine (Michie, 1963).

One 'matchbox' per board position that MENACE (playing X, moving first) can face,
after merging positions that are rotations/reflections of each other.  Each box
holds coloured 'beads', one colour per legal move.  To move, MENACE draws a bead
at random (probability proportional to bead counts).  After the game every box
used is reinforced:  win +3 beads, draw +1, loss -1 (the bead is confiscated).
Initial beads per move: 4 / 3 / 2 / 1 for MENACE's 1st / 2nd / 3rd / 4th move.

Crucial parts of the implementation (highlighted in the report):
  * canonical(board)  - symmetry reduction (8 symmetries of the square)
  * choose_move()     - bead draw = stochastic policy pi(a|s) proportional to beads
  * reinforce()       - delayed reward credited to every (box, bead) of the game
  * empty box rule    - if a box runs out of beads MENACE resigns (and the box is refilled)

Run:  python lab7/menace.py
"""
import json
import os
import random
from functools import lru_cache

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab7")
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
# the 8 symmetries as index permutations: new[i] = old[perm[i]]
_ROT = [6, 3, 0, 7, 4, 1, 8, 5, 2]
_REF = [2, 1, 0, 5, 4, 3, 8, 7, 6]


def _compose(p, q):
    return [p[q[i]] for i in range(9)]


SYMS = []
_p = list(range(9))
for _ in range(4):
    SYMS.append(_p)
    SYMS.append(_compose(_p, _REF))
    _p = _compose(_p, _ROT)


def winner(b):
    for a, c, d in LINES:
        if b[a] != "." and b[a] == b[c] == b[d]:
            return b[a]
    return "D" if "." not in b else None


def canonical(board):
    """Lexicographically smallest symmetric image, and the permutation that produced it."""
    best, best_p = None, None
    for p in SYMS:
        img = "".join(board[p[i]] for i in range(9))
        if best is None or img < best:
            best, best_p = img, p
    return best, best_p


class Menace:
    INIT = {0: 4, 2: 3, 4: 2, 6: 1}           # beads per move, keyed by #pieces on board

    def __init__(self, rng):
        self.boxes = {}
        self.rng = rng

    def box(self, canon):
        if canon not in self.boxes:
            n = canon.count("X") + canon.count("O")
            moves = [i for i, c in enumerate(canon) if c == "."]
            # drop moves that are symmetric duplicates inside this position
            seen, uniq = set(), []
            for m in moves:
                imgs = []
                for p in SYMS:
                    if "".join(canon[p[i]] for i in range(9)) == canon:
                        imgs.append(p.index(m))
                key = min(imgs)
                if key not in seen:
                    seen.add(key)
                    uniq.append(m)
            self.boxes[canon] = {m: self.INIT.get(n, 1) for m in uniq}
        return self.boxes[canon]

    def choose_move(self, board):
        canon, p = canonical(board)
        beads = self.box(canon)
        if sum(beads.values()) == 0:
            return None, canon, None              # empty box -> resign
        moves, w = zip(*beads.items())
        m_c = self.rng.choices(moves, weights=w)[0]
        return p[m_c], canon, m_c                 # map canonical move back to the real board

    def reinforce(self, history, result):
        delta = {"X": 3, "D": 1, "O": -1}[result]
        for canon, m in history:
            box = self.boxes[canon]
            box[m] = max(0, box[m] + delta)
            if sum(box.values()) == 0:           # Michie refilled emptied boxes
                for k in box:
                    box[k] = 1


def random_player(board, rng):
    return rng.choice([i for i, c in enumerate(board) if c == "."])


@lru_cache(maxsize=None)
def minimax(board, player):
    w = winner(board)
    if w == "X":
        return 1, None
    if w == "O":
        return -1, None
    if w == "D":
        return 0, None
    best = None
    for i, c in enumerate(board):
        if c != ".":
            continue
        nb = board[:i] + player + board[i + 1:]
        v, _ = minimax(nb, "O" if player == "X" else "X")
        if best is None or (player == "X" and v > best[0]) or (player == "O" and v < best[0]):
            best = (v, i)
    return best


def optimal_player(board, rng):
    """Perfect O: picks uniformly among minimax-optimal moves."""
    vals = []
    for i, c in enumerate(board):
        if c == ".":
            v, _ = minimax(board[:i] + "O" + board[i + 1:], "X")
            vals.append((v, i))
    best = min(v for v, _ in vals)
    return rng.choice([i for v, i in vals if v == best])


def play(menace, opponent, rng):
    board, history = "." * 9, []
    while True:
        mv, canon, m_c = menace.choose_move(board)
        if mv is None:
            return "O", history                  # resignation counts as a loss
        history.append((canon, m_c))
        board = board[:mv] + "X" + board[mv + 1:]
        w = winner(board)
        if w:
            return w, history
        o = opponent(board, rng)
        board = board[:o] + "O" + board[o + 1:]
        w = winner(board)
        if w:
            return w, history


def train(opponent, games=3000, seed=0, window=100):
    rng = random.Random(seed)
    menace = Menace(rng)
    results = []
    for _ in range(games):
        res, hist = play(menace, opponent, rng)
        menace.reinforce(hist, res)
        results.append(res)
    arr = np.array(results)
    roll = {k: np.convolve((arr == k).astype(float), np.ones(window) / window, mode="valid") for k in "XDO"}
    return menace, results, roll


def count_positions():
    """All positions with X to move reachable in legal play (before any game ends)."""
    seen = set()

    def rec(board, player):
        if winner(board):
            return
        if player == "X":
            seen.add(canonical(board)[0])
        for i, c in enumerate(board):
            if c == ".":
                rec(board[:i] + player + board[i + 1:], "O" if player == "X" else "X")
    rec("." * 9, "X")
    return len(seen)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    summary = {"positions_X_to_move_canonical": count_positions()}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (name, opp) in zip(axes, [("random opponent", random_player), ("perfect (minimax) opponent", optimal_player)]):
        rolls, finals = [], []
        for seed in range(5):
            m, res, roll = train(opp, seed=seed)
            rolls.append(roll)
            last = np.array(res[-500:])
            finals.append({k: float(np.mean(last == k)) for k in "XDO"})
        for k, lab, col in (("X", "MENACE wins", "#2a9d8f"), ("D", "draws", "#e9a03b"), ("O", "MENACE loses", "#d1495b")):
            mean = np.mean([r[k] for r in rolls], axis=0)
            ax.plot(np.arange(len(mean)) + 100, mean, color=col, label=lab, lw=1.2)
        ax.set_title(f"MENACE vs {name}", fontsize=9); ax.set_xlabel("games played")
        ax.grid(alpha=0.3)
        summary[name] = dict(last500_mean={k: float(np.mean([f[k] for f in finals])) for k in "XDO"},
                             boxes_used=len(m.boxes),
                             first_box=dict(sorted(m.boxes["." * 9].items(), key=lambda kv: -kv[1])))
    axes[0].set_ylabel("rate (100-game moving average)"); axes[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "menace_learning.png"), dpi=150); plt.close()
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(os.path.join(RESULTS, "lab7_menace_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
