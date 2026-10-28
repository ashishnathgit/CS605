"""Figures for Lab 3 from results/lab3/lab3_summary.csv."""
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab3")
COL = {"HC": "#777777", "Beam-3": "#2a9d8f", "Beam-4": "#264653", "VND": "#d1495b"}


def main():
    s = pd.read_csv(os.path.join(RESULTS, "lab3_summary.csv"))
    s["ratio"] = (s["m"] / s["n"]).round(2)
    for metric, ylabel, fname in (("success", "success rate", "lab3_success.png"),
                                  ("penetrance", "mean penetrance L/T", "lab3_penetrance.png")):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
        for ax, n in zip(axes, sorted(s.n.unique())):
            d = s[s.n == n]
            for alg, c in COL.items():
                for h, ls in (("h1", "--"), ("h2", "-")):
                    q = d[(d.algorithm == alg) & (d.heuristic == h)].sort_values("ratio")
                    ax.plot(q.ratio, q[metric], ls, marker="o", ms=3, color=c,
                            label=f"{alg} / {h}" if n == s.n.min() else None)
            ax.set_title(f"n = {n}", fontsize=9); ax.set_xlabel("clause/variable ratio m/n")
            ax.grid(alpha=0.3)
            if metric == "penetrance":
                ax.set_yscale("log")
        axes[0].set_ylabel(ylabel)
        axes[0].legend(fontsize=6, ncol=2)
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS, fname), dpi=150); plt.close()


if __name__ == "__main__":
    main()
