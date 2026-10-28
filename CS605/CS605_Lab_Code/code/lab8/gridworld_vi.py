"""
CS605 Lab 8 (1) - Value iteration on the 4x3 stochastic grid world (AIMA Fig. 17.1).

Bellman update  U(s) <- R(s) + gamma * max_a sum_s' P(s'|s,a) U(s'),  U(terminal) = +-1.
Moves succeed with probability 0.8 and slip at right angles with 0.1 each; bumping
into a wall or the obstacle at (2,2) leaves the agent in place.
gamma = 1 is used whenever the problem is proper (R(s) < 0 makes every policy that
avoids the exits infinitely bad).  For R(s) >= 0 an agent can collect reward
forever, U diverges for gamma = 1, so gamma = 0.99 is used and stated.

Run:  python lab8/gridworld_vi.py
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab8")
COLS, ROWS = 4, 3
WALL = {(2, 2)}
TERMINAL = {(4, 3): 1.0, (4, 2): -1.0}
ACTIONS = {"U": (0, 1), "D": (0, -1), "L": (-1, 0), "R": (1, 0)}
PERP = {"U": ("L", "R"), "D": ("L", "R"), "L": ("U", "D"), "R": ("U", "D")}
ARROW = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
STATES = [(x, y) for x in range(1, COLS + 1) for y in range(1, ROWS + 1) if (x, y) not in WALL]


def move(s, a):
    dx, dy = ACTIONS[a]
    n = (s[0] + dx, s[1] + dy)
    if n in WALL or not (1 <= n[0] <= COLS and 1 <= n[1] <= ROWS):
        return s
    return n


def transitions(s, a):
    p1, p2 = PERP[a]
    return [(0.8, move(s, a)), (0.1, move(s, p1)), (0.1, move(s, p2))]


def value_iteration(r, gamma, eps=1e-6, max_iter=100000):
    U = {s: TERMINAL.get(s, 0.0) for s in STATES}
    for it in range(1, max_iter + 1):
        delta, U2 = 0.0, {}
        for s in STATES:
            if s in TERMINAL:
                U2[s] = TERMINAL[s]
                continue
            best = max(sum(p * U[s2] for p, s2 in transitions(s, a)) for a in ACTIONS)
            U2[s] = r + gamma * best
            delta = max(delta, abs(U2[s] - U[s]))
        U = U2
        thr = eps * (1 - gamma) / gamma if gamma < 1 else eps
        if delta < thr:
            break
    pi = {}
    for s in STATES:
        if s in TERMINAL:
            continue
        q = {a: sum(p * U[s2] for p, s2 in transitions(s, a)) for a in ACTIONS}
        m = max(q.values())
        pi[s] = "".join(sorted(a for a, v in q.items() if abs(v - m) < 1e-9))  # ties reported
    return U, pi, it


def draw(ax, U, pi, title):
    for s in STATES:
        x, y = s
        col = "#d8f3dc" if TERMINAL.get(s, 0) > 0 else "#ffd6d6" if TERMINAL.get(s, 0) < 0 else "white"
        ax.add_patch(plt.Rectangle((x - 1, y - 1), 1, 1, fc=col, ec="k", lw=0.6))
        txt = f"{U[s]:+.3f}" if abs(U[s]) < 1e3 else f"{U[s]:.3g}"
        ax.text(x - 0.5, y - 0.35, txt, ha="center", va="center", fontsize=7)
        if s in pi:
            ax.text(x - 0.5, y - 0.68, "".join(ARROW[a] for a in pi[s]), ha="center", va="center", fontsize=11)
    for w in WALL:
        ax.add_patch(plt.Rectangle((w[0] - 1, w[1] - 1), 1, 1, fc="#888", ec="k", lw=0.6))
    ax.set_xlim(0, 4); ax.set_ylim(0, 3); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=8)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    cases = [(-0.04, 1.0), (-2.0, 1.0), (0.1, 0.99), (0.02, 0.99), (1.0, 0.99)]
    fig, axes = plt.subplots(1, len(cases), figsize=(15, 2.8))
    summary = {}
    for ax, (r, g) in zip(axes, cases):
        U, pi, it = value_iteration(r, g)
        summary[f"r={r}"] = dict(gamma=g, iterations=it, U={f"{s}": U[s] for s in STATES},
                                 policy={f"{s}": pi[s] for s in pi})
        draw(ax, U, pi, f"r(s) = {r},  γ = {g}  ({it} sweeps)")
        print(f"r={r:+.2f} gamma={g}: {it} sweeps; U(1,1)={U[(1,1)]:.4f}; policy:",
              {f"{s}": pi[s] for s in sorted(pi, key=lambda s: (-s[1], s[0]))})
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "gridworld_policies.png"), dpi=170); plt.close()

    # threshold sweep: where does the optimal policy change as r(s) varies?
    sweep = []
    prev = None
    for r in np.round(np.arange(-2.0, 0.0001, 0.005), 4):
        _, pi, _ = value_iteration(float(r), 1.0, eps=1e-8)
        key = tuple(pi[s] for s in sorted(pi))
        if key != prev:
            sweep.append(dict(r=float(r), policy={f"{s}": pi[s] for s in sorted(pi)}))
            prev = key
    summary["policy_change_points_gamma1"] = sweep
    json.dump(summary, open(os.path.join(RESULTS, "lab8_gridworld_summary.json"), "w"), indent=2)
    print("policy change points (gamma=1):", [d["r"] for d in sweep])


if __name__ == "__main__":
    main()
