"""
CS605 Lab 4 (in-lab problem) - Tour of Rajasthan with simulated annealing.

20 tourist locations (approximate latitude/longitude). Cost of travel between two
places is taken to be proportional to the great-circle distance (haversine), as
the problem statement allows. SA uses 2-opt segment reversal moves.  For 20
cities the exact optimum is also computed with Held-Karp dynamic programming so
that the SA result can be checked.

The same SA code can read TSPLIB files (e.g. the VLSI instance xqf131.tsp from
math.uwaterloo.ca/tsp/vlsi) via load_tsplib(); those files were not reachable
from the environment in which these results were produced.

Run:  python lab4/tsp_rajasthan_sa.py
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab4")

PLACES = {
    "Jaipur": (26.912, 75.787), "Ajmer": (26.449, 74.639), "Pushkar": (26.490, 74.551),
    "Jodhpur": (26.238, 73.024), "Jaisalmer": (26.915, 70.908), "Bikaner": (28.022, 73.312),
    "Udaipur": (24.585, 73.712), "Mount Abu": (24.593, 72.708), "Chittorgarh": (24.888, 74.626),
    "Kumbhalgarh": (25.148, 73.580), "Ranakpur": (25.116, 73.473), "Nathdwara": (24.938, 73.823),
    "Bundi": (25.438, 75.640), "Kota": (25.213, 75.865), "Sawai Madhopur": (26.017, 76.502),
    "Bharatpur": (27.217, 77.490), "Alwar": (27.553, 76.635), "Mandawa": (28.055, 75.149),
    "Abhaneri": (27.007, 76.607), "Osian": (26.720, 72.910),
}


def haversine_matrix(coords):
    lat, lon = np.radians(coords[:, 0])[:, None], np.radians(coords[:, 1])[:, None]
    dlat, dlon = lat - lat.T, lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def load_tsplib(path):
    """Minimal TSPLIB reader for EUC_2D NODE_COORD_SECTION files."""
    pts, on = [], False
    for line in open(path):
        line = line.strip()
        if line.startswith("NODE_COORD_SECTION"):
            on = True
            continue
        if line.startswith("EOF"):
            break
        if on and line:
            _, x, y = line.split()[:3]
            pts.append((float(x), float(y)))
    pts = np.array(pts)
    return np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)), pts


def tour_length(t, D):
    return D[t, np.roll(t, -1)].sum()


def sa_tsp(D, rng, iters=200000, alpha=None, T0=None):
    n = len(D)
    t = rng.permutation(n)
    L = tour_length(t, D)
    if T0 is None:
        T0 = 0.1 * L / n * 10
    Tend = 1e-3 * T0
    alpha = alpha or (Tend / T0) ** (1 / iters)
    T, best, bestL, trace = T0, t.copy(), L, []
    for it in range(iters):
        i, j = sorted(rng.choice(n, 2, replace=False))
        if j - i < 1 or (i == 0 and j == n - 1):
            continue
        a, b, c, d = t[i - 1], t[i], t[j], t[(j + 1) % n]
        dL = D[a, c] + D[b, d] - D[a, b] - D[c, d]          # 2-opt: reverse t[i..j]
        if dL < 0 or rng.random() < np.exp(-dL / T):
            t[i:j + 1] = t[i:j + 1][::-1]
            L += dL
            if L < bestL:
                bestL, best = L, t.copy()
        T *= alpha
        if it % 1000 == 0:
            trace.append((it, L, bestL))
    return best, tour_length(best, D), trace


def held_karp(D):
    """Exact TSP by dynamic programming over subsets, O(2^n n^2) - fine for n = 20."""
    n = len(D)
    m = n - 1                                  # city 0 is the fixed start
    INF = np.inf
    dp = np.full((1 << m, m), INF)
    parent = np.full((1 << m, m), -1, dtype=np.int8)
    for j in range(m):
        dp[1 << j, j] = D[0, j + 1]
    Dm = D[1:, 1:]
    for mask in range(1, 1 << m):
        cur = dp[mask]
        if not np.isfinite(cur).any():
            continue
        # extend to every city k not in mask
        cand = cur[:, None] + Dm                # cand[j, k] = dp[mask, j] + d(j, k)
        best_j = np.argmin(cand, axis=0)
        best_v = cand[best_j, np.arange(m)]
        for k in range(m):
            if mask & (1 << k):
                continue
            nm = mask | (1 << k)
            if best_v[k] < dp[nm, k]:
                dp[nm, k] = best_v[k]
                parent[nm, k] = best_j[k]
    full = (1 << m) - 1
    last = int(np.argmin(dp[full] + D[1:, 0]))
    opt = dp[full, last] + D[last + 1, 0]
    tour, mask, j = [], full, last
    while j != -1:
        tour.append(j + 1)
        pj = parent[mask, j]
        mask ^= 1 << j
        j = int(pj)
    return [0] + tour[::-1], float(opt)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    names = list(PLACES)
    coords = np.array([PLACES[k] for k in names])
    D = haversine_matrix(coords)
    runs = []
    for seed in range(10):
        t, L, trace = sa_tsp(D, np.random.default_rng(seed))
        runs.append((L, t, trace))
    runs.sort(key=lambda r: r[0])
    bestL, best, trace = runs[0]
    opt_tour, optL = held_karp(D)
    k = int(np.where(best == 0)[0][0])
    best = np.roll(best, -k)
    print(f"SA best = {bestL:.1f} km, Held-Karp optimum = {optL:.1f} km")
    print("SA lengths over 10 restarts:", [round(r[0], 1) for r in runs])
    print("Tour:", " -> ".join(names[i] for i in best), "-> Jaipur")

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    loop = np.append(best, best[0])
    ax[0].plot(coords[loop, 1], coords[loop, 0], "-o", ms=3, lw=1)
    for i, nme in enumerate(names):
        ax[0].annotate(nme, (coords[i, 1], coords[i, 0]), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax[0].set_xlabel("longitude (°E)"); ax[0].set_ylabel("latitude (°N)")
    ax[0].set_title(f"SA tour: {bestL:.0f} km (optimum {optL:.0f} km)", fontsize=9)
    tr = np.array(trace)
    ax[1].plot(tr[:, 0], tr[:, 1], lw=0.8, label="current")
    ax[1].plot(tr[:, 0], tr[:, 2], lw=1.2, label="best")
    ax[1].axhline(optL, color="k", ls="--", lw=0.8, label="Held-Karp optimum")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("tour length (km)"); ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "tsp_rajasthan.png"), dpi=150); plt.close()
    json.dump(dict(names=names, sa_best_km=bestL, optimum_km=optL,
                   sa_restart_km=[r[0] for r in runs], tour=[names[i] for i in best],
                   optimal_tour=[names[i] for i in opt_tour]),
              open(os.path.join(RESULTS, "lab4_tsp_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
