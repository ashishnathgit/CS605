"""
CS605 Lab 4 - Solving the scrambled-Lena jigsaw with simulated annealing.

Formulation
-----------
The 512x512 image is a 4x4 grid of 128x128 tiles (16 tiles, no rotation).
State     : a permutation pi, pi[p] = tile placed at grid position p (p = 0..15, row-major)
Size      : 16! ~ 2.09e13 states
Operators : swap the tiles at two positions (90 %), and occasionally swap two whole
            rows/columns or cyclically shift all rows/columns (10 %) - the block moves
            let SA escape arrangements that are correct only up to a translation
Energy    : E(pi) = sum over horizontally adjacent (p, q) of D_LR(pi[p], pi[q])
                  + sum over vertically   adjacent (p, q) of D_TB(pi[p], pi[q])
            Two seam dissimilarities D are compared:
            (a) 'pred': prediction-based L1 dissimilarity (Pomeranz et al., 2011) - the
                last pixel column of the left tile is linearly extrapolated one pixel and
                compared with the first column of the right tile, in both directions;
            (b) 'mgc' : Mahalanobis gradient compatibility (Gallagher, 2012) - the
                intensity jump across the seam is scored against the distribution of
                gradients just inside each tile, so a seam that crosses a genuine,
                strong image edge is not penalised as heavily as an unnatural one.
Goal      : minimum-energy arrangement (the true image has natural, smooth seams).
SA        : Metropolis acceptance exp(-dE / T), geometric cooling T <- alpha T,
            several independent restarts, best state kept (Kirkpatrick et al. 1983).

Run:  python lab4/jigsaw_sa.py     (from the code/ directory)
"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "scrambled_lena.mat")
RESULTS = os.path.join(HERE, "..", "..", "results", "lab4")
G = 4                      # grid size


def load_octave_text_matrix(path):
    """scrambled_lena.mat is an Octave *text* file (column-major values, one per line)."""
    with open(path) as f:
        lines = f.read().split("\n")
    k = next(i for i, l in enumerate(lines) if l.startswith("# ndims"))
    r, c = map(int, lines[k + 1].split())
    vals = np.array([int(x) for x in lines[k + 2:] if x.strip()][: r * c], dtype=np.float64)
    return vals.reshape(c, r).T          # column-major -> (rows, cols)


def cut_tiles(img, g=G):
    h = img.shape[0] // g
    return [img[r * h:(r + 1) * h, c * h:(c + 1) * h] for r in range(g) for c in range(g)]


def assemble(tiles, perm, g=G):
    rows = [np.hstack([tiles[perm[r * g + c]] for c in range(g)]) for r in range(g)]
    return np.vstack(rows)


def dissimilarity_matrices(tiles):
    """D_LR[i, j]: cost of tile j to the right of tile i; D_TB[i, j]: j below i."""
    n = len(tiles)
    DLR, DTB = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                DLR[i, j] = DTB[i, j] = np.inf
                continue
            a, b = tiles[i], tiles[j]
            # left -> right prediction and right -> left prediction
            pr = 2 * a[:, -1] - a[:, -2]
            pl = 2 * b[:, 0] - b[:, 1]
            DLR[i, j] = np.abs(pr - b[:, 0]).sum() + np.abs(pl - a[:, -1]).sum()
            pd = 2 * a[-1, :] - a[-2, :]
            pu = 2 * b[0, :] - b[1, :]
            DTB[i, j] = np.abs(pd - b[0, :]).sum() + np.abs(pu - a[-1, :]).sum()
    return DLR, DTB


def mgc_matrices(tiles, eps=1.0):
    """Mahalanobis gradient compatibility (grey-level, 1-D covariance) in both directions."""
    n = len(tiles)
    DLR, DTB = np.full((n, n), np.inf), np.full((n, n), np.inf)

    def seam(a_edge, a_in, b_edge, b_in):
        ga = a_edge - a_in                       # gradient just inside tile a
        gb = b_edge - b_in                       # gradient just inside tile b
        d_ab = ((b_edge - a_edge - ga.mean()) ** 2 / (ga.var() + eps)).sum()
        d_ba = ((a_edge - b_edge - gb.mean()) ** 2 / (gb.var() + eps)).sum()
        return d_ab + d_ba

    for i in range(n):
        for j in range(n):
            if i != j:
                a, b = tiles[i], tiles[j]
                DLR[i, j] = seam(a[:, -1], a[:, -2], b[:, 0], b[:, 1])
                DTB[i, j] = seam(a[-1, :], a[-2, :], b[0, :], b[1, :])
    return DLR, DTB


def energy(perm, DLR, DTB, g=G):
    P = np.asarray(perm).reshape(g, g)
    return DLR[P[:, :-1], P[:, 1:]].sum() + DTB[P[:-1, :], P[1:, :]].sum()


def initial_temperature(DLR, DTB, rng, samples=300, g=G):
    """T0 = median uphill energy change of random swaps from random states (accept ~37 %)."""
    ups = []
    for _ in range(samples):
        P = rng.permutation(g * g)
        p, q = rng.choice(g * g, 2, replace=False)
        Q = P.copy(); Q[p], Q[q] = Q[q], Q[p]
        d = energy(Q, DLR, DTB) - energy(P, DLR, DTB)
        if d > 0:
            ups.append(d)
    return float(np.median(ups))


def simulated_annealing(DLR, DTB, rng, T0=None, alpha=0.9995, iters=60000, g=G, p_block=0.1):
    n = g * g
    P = rng.permutation(n).reshape(g, g)
    E = energy(P.ravel(), DLR, DTB)
    T = T0 if T0 is not None else initial_temperature(DLR, DTB, rng)
    best_P, best_E = P.copy(), E
    trace = []
    for it in range(iters):
        if rng.random() >= p_block:                      # swap two tiles
            p, q = rng.choice(n, 2, replace=False)
            (r1, c1), (r2, c2) = divmod(p, g), divmod(q, g)
            P[r1, c1], P[r2, c2] = P[r2, c2], P[r1, c1]
            E2 = energy(P.ravel(), DLR, DTB)
            dE = E2 - E
            if dE > 0 and rng.random() >= np.exp(-dE / T):
                P[r1, c1], P[r2, c2] = P[r2, c2], P[r1, c1]   # reject: undo the swap
            else:
                E = E2
        else:                                            # block move
            P2 = P.copy()
            a, b = rng.choice(g, 2, replace=False)
            kind = rng.integers(4)
            if kind == 0:
                P2[[a, b], :] = P2[[b, a], :]               # swap two rows
            elif kind == 1:
                P2[:, [a, b]] = P2[:, [b, a]]               # swap two columns
            elif kind == 2:
                P2 = np.roll(P2, 1 + rng.integers(g - 1), axis=0)   # cyclic row shift
            else:
                P2 = np.roll(P2, 1 + rng.integers(g - 1), axis=1)   # cyclic column shift
            E2 = energy(P2.ravel(), DLR, DTB)
            dE = E2 - E
            if dE <= 0 or rng.random() < np.exp(-dE / T):
                P, E = P2, E2
        if E < best_E:
            best_E, best_P = E, P.copy()
        T *= alpha
        if it % 200 == 0:
            trace.append((it, E, best_E, T))
    best_E = energy(best_P.ravel(), DLR, DTB)
    return best_P.ravel(), best_E, trace


def neighbour_accuracy(perm, ref_perm, g=G):
    """Fraction of the 2g(g-1) adjacency relations of the reference reproduced by perm."""
    def pairs(pm):
        P = np.asarray(pm).reshape(g, g)
        s = set()
        for r in range(g):
            for c in range(g):
                if c < g - 1:
                    s.add(("LR", P[r, c], P[r, c + 1]))
                if r < g - 1:
                    s.add(("TB", P[r, c], P[r + 1, c]))
        return s
    ref = pairs(ref_perm)
    return len(pairs(perm) & ref) / len(ref)


def solve(DLR, DTB, restarts=10, seed0=0):
    runs = []
    for seed in range(restarts):
        rng = np.random.default_rng(seed0 + seed)
        perm, E, trace = simulated_annealing(DLR, DTB, rng)
        runs.append(dict(seed=seed, energy=float(E), perm=perm.tolist(), trace=trace))
    best = min(runs, key=lambda r: r["energy"])
    return runs, best


def show(img, solved, title_left, title_right, path):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255); ax[0].set_title(title_left, fontsize=10)
    ax[1].imshow(solved, cmap="gray", vmin=0, vmax=255); ax[1].set_title(title_right, fontsize=10)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
        for k in range(1, G):
            a.axhline(k * 128 - 0.5, color="w", lw=0.4, alpha=0.5)
            a.axvline(k * 128 - 0.5, color="w", lw=0.4, alpha=0.5)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    os.makedirs(RESULTS, exist_ok=True)
    img = load_octave_text_matrix(DATA)
    tiles = cut_tiles(img)
    identity = np.arange(G * G)
    summary = {}
    metrics = {"pred": dissimilarity_matrices(tiles), "mgc": mgc_matrices(tiles)}
    bests = {}
    for name, (DLR, DTB) in metrics.items():
        runs, best = solve(DLR, DTB)
        greedy = []
        for seed in range(10):                       # T ~ 0: randomised pure descent baseline
            rng = np.random.default_rng(100 + seed)
            _, E, _ = simulated_annealing(DLR, DTB, rng, T0=1e-12)
            greedy.append(float(E))
        swap_only = []                               # ablation: no block moves
        for seed in range(10):
            rng = np.random.default_rng(200 + seed)
            _, E, _ = simulated_annealing(DLR, DTB, rng, p_block=0.0)
            swap_only.append(float(E))
        bests[name] = best
        summary[name] = dict(E_scrambled=float(energy(identity, DLR, DTB)),
                             best_energy=best["energy"], best_perm=best["perm"],
                             sa_energies=[r["energy"] for r in runs],
                             sa_hits_best=sum(r["perm"] == best["perm"] for r in runs),
                             greedy_energies=greedy,
                             greedy_hits_best=sum(abs(e - best["energy"]) < 1e-6 for e in greedy),
                             swap_only_energies=swap_only,
                             swap_only_hits_best=sum(abs(e - best["energy"]) < 1e-6 for e in swap_only),
                             T0=initial_temperature(DLR, DTB, np.random.default_rng(0)))
        print(name, {k: v for k, v in summary[name].items() if k != "best_perm"})
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        for r in runs[:5]:
            tr = np.array(r["trace"])
            ax.plot(tr[:, 0], tr[:, 1], lw=0.8, label=f"restart {r['seed']}")
        ax.set_xlabel("iteration"); ax.set_ylabel("energy E(\u03c0)"); ax.set_yscale("log")
        ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title(f"SA energy traces ({name})", fontsize=10)
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS, f"jigsaw_energy_trace_{name}.png"), dpi=150)
        plt.close()

    # The MGC optimum is the visually correct image; use it as reference arrangement
    ref = np.array(bests["mgc"]["perm"])
    for name in metrics:
        summary[name]["neighbour_acc_vs_mgc_best"] = neighbour_accuracy(bests[name]["perm"], ref)
        summary[name]["E_of_mgc_solution"] = float(energy(ref, *metrics[name]))
    solved = assemble(tiles, ref)
    show(img, assemble(tiles, np.array(bests["pred"]["perm"])),
         "Scrambled input", f"SA optimum with D_pred (E = {bests['pred']['energy']:,.0f})",
         os.path.join(RESULTS, "jigsaw_pred_solution.png"))
    show(img, solved, "Scrambled input", "SA optimum with D_MGC",
         os.path.join(RESULTS, "jigsaw_before_after.png"))
    plt.imsave(os.path.join(RESULTS, "lena_solved.png"), solved, cmap="gray", vmin=0, vmax=255)
    with open(os.path.join(RESULTS, "lab4_jigsaw_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("MGC best permutation (tile index at each grid position):\n", ref.reshape(G, G))
    for name in metrics:
        print(name, "neighbour accuracy vs MGC solution:", summary[name]["neighbour_acc_vs_mgc_best"],
              " E(mgc solution) under this metric:", summary[name]["E_of_mgc_solution"])


if __name__ == "__main__":
    main()
