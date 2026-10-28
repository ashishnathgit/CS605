"""Figures for Labs 1 and 2 from their JSON summaries."""
import json
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = os.path.join(os.path.dirname(__file__), "..", "..", "results")


def main():
    s = json.load(open(os.path.join(R, "lab1", "lab1_summary.json")))
    d = pd.DataFrame(s["rabbit_scaling"])
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].semilogy(d.k, d.raw, "o-", label="all strings (2k+1)!/(k!)²")
    ax[0].semilogy(d.k, d.reach, "s-", label="reachable from start")
    ax[0].semilogy(d.k, d.bfs_exp, "^-", label="BFS expanded")
    ax[0].semilogy(d.k, d.dfs_exp, "v-", label="DFS expanded")
    ax[0].set_xlabel("rabbits per side k"); ax[0].set_ylabel("count"); ax[0].legend(fontsize=7)
    ax[1].plot(d.k, d.bfs_front, "^-", label="BFS max frontier (queue)")
    ax[1].plot(d.k, d.dfs_front, "v-", label="DFS max frontier (stack)")
    ax[1].set_xlabel("rabbits per side k"); ax[1].set_ylabel("nodes held"); ax[1].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(R, "lab1", "rabbit_scaling.png"), dpi=150); plt.close()

    s2 = json.load(open(os.path.join(R, "lab2", "lab2_summary.json")))
    d2 = pd.DataFrame(s2["scaling"])
    fig, ax = plt.subplots(figsize=(4.6, 3))
    for h, m, lab in (("h0", "o", "h0 = 0 (uniform cost)"), ("h1", "s", "h1 (count bound)"),
                      ("h2", "^", "h2 (length bound)")):
        q = d2[d2.h == h]
        ax.loglog(q.n, q.expanded, m + "-", label=lab)
    q = d2[d2.h == "h0"]
    ax.loglog(q.n, q.pairs, "k:", label="all sentence pairs nA·nB")
    ax.set_xlabel("sentences per document"); ax.set_ylabel("nodes expanded")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(os.path.join(R, "lab2", "astar_scaling.png"), dpi=150); plt.close()


if __name__ == "__main__":
    main()
