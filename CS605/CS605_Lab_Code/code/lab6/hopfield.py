"""
CS605 Lab 6 - Hopfield networks.

(1) 10x10 binary associative memory (N = 100 bipolar units, Hebbian weights)
(2) Storage capacity: fraction of stored patterns that remain fixed points / are
    recalled, as the number of random patterns P grows (theory: P_max ~ 0.138 N)
(3) Error-correcting capability: recall success vs. number of flipped bits
(4) Eight-rook problem with a 0/1 Hopfield network (energy-function design)
(5) 10-city TSP with the continuous Hopfield-Tank network

References: MacKay, Information Theory, Inference and Learning Algorithms, Ch. 42;
Hopfield & Tank (1985), Biological Cybernetics 52:141-152.

Run:  python lab6/hopfield.py
"""
import itertools
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab6")
N_SIDE = 10
N = N_SIDE * N_SIDE

GLYPHS = {
    "A": ["....##....", "...####...", "..##..##..", ".##....##.", ".##....##.",
          ".########.", ".########.", ".##....##.", ".##....##.", ".##....##."],
    "T": ["##########", "##########", "....##....", "....##....", "....##....",
          "....##....", "....##....", "....##....", "....##....", "....##...."],
    "X": ["##......##", ".##....##.", "..##..##..", "...####...", "....##....",
          "....##....", "...####...", "..##..##..", ".##....##.", "##......##"],
    "O": ["..######..", ".########.", "##......##", "##......##", "##......##",
          "##......##", "##......##", "##......##", ".########.", "..######.."],
    "L": ["##........", "##........", "##........", "##........", "##........",
          "##........", "##........", "##........", "#########.", "#########."],
}


def glyph(rows):
    return np.array([[1 if c == "#" else -1 for c in r] for r in rows]).ravel()


# ----------------------------------------------------------------------------
# (1)-(3) associative memory
# ----------------------------------------------------------------------------
def hebbian(patterns):
    X = np.asarray(patterns, float)
    W = X.T @ X / X.shape[1]
    np.fill_diagonal(W, 0)
    return W


def recall(W, s, rng, max_sweeps=50):
    """Asynchronous updates in random order until no unit changes."""
    s = s.copy()
    for _ in range(max_sweeps):
        changed = 0
        for i in rng.permutation(len(s)):
            h = W[i] @ s
            new = 1 if h > 0 else (-1 if h < 0 else s[i])
            if new != s[i]:
                s[i] = new
                changed += 1
        if changed == 0:
            break
    return s


def energy_bipolar(W, s):
    return -0.5 * s @ W @ s


def flip(s, k, rng):
    s = s.copy()
    idx = rng.choice(len(s), k, replace=False)
    s[idx] *= -1
    return s


def capacity_experiment(rng, Ps=range(1, 31), trials=20, noise_bits=0):
    rows = []
    for P in Ps:
        stable_bits, perfect, recall_ok = [], [], []
        for _ in range(trials):
            X = rng.choice([-1, 1], size=(P, N))
            W = hebbian(X)
            # bit stability of stored patterns under one synchronous step
            h = X @ W
            stable_bits.append(np.mean(np.sign(h) == X))
            perfect.append(np.mean(np.all(np.sign(h) == X, axis=1)))
            mu = rng.integers(P)
            probe = flip(X[mu], noise_bits, rng) if noise_bits else X[mu]
            out = recall(W, probe, rng)
            recall_ok.append(np.mean(out == X[mu]) >= 0.99)
        rows.append(dict(P=P, alpha=P / N, bit_stability=float(np.mean(stable_bits)),
                         pattern_stable=float(np.mean(perfect)), recall=float(np.mean(recall_ok))))
    return rows


def error_correction(rng, Ps=(1, 3, 5, 8, 10, 12), ks=range(0, 51, 2), trials=60):
    out = {}
    for P in Ps:
        curve = []
        for k in ks:
            ok = []
            for _ in range(trials):
                X = rng.choice([-1, 1], size=(P, N))
                W = hebbian(X)
                mu = rng.integers(P)
                res = recall(W, flip(X[mu], k, rng), rng)
                ok.append(np.array_equal(res, X[mu]))
            curve.append(float(np.mean(ok)))
        out[P] = curve
    return list(ks), out


# ----------------------------------------------------------------------------
# (4) Eight-rook problem
# ----------------------------------------------------------------------------
def rook_network(n=8, A=2.0, B=2.0):
    """E = A/2 sum_i (sum_j v_ij - 1)^2 + B/2 sum_j (sum_i v_ij - 1)^2,  v in {0,1}.
    With v^2 = v this equals  -1/2 sum w v v - sum I v + const with
       w[(i,j),(i,l)] = -A (j != l),   w[(i,j),(k,j)] = -B (i != k),   I_ij = (A + B)/2."""
    M = n * n
    W = np.zeros((M, M))
    for i in range(n):
        for j in range(n):
            a = i * n + j
            for l in range(n):
                if l != j:
                    W[a, i * n + l] = -A
            for k in range(n):
                if k != i:
                    W[a, k * n + j] = -B
    I = np.full(M, (A + B) / 2)
    return W, I


def run_rook(W, I, rng, n=8, max_sweeps=100):
    v = (rng.random(n * n) < 0.5).astype(float)
    for sweep in range(max_sweeps):
        changed = 0
        for a in rng.permutation(n * n):
            new = 1.0 if W[a] @ v + I[a] > 0 else 0.0
            if new != v[a]:
                v[a] = new
                changed += 1
        if changed == 0:
            break
    board = v.reshape(n, n)
    valid = np.all(board.sum(0) == 1) and np.all(board.sum(1) == 1)
    E = -0.5 * v @ W @ v - I @ v
    return board, bool(valid), sweep + 1, float(E)


# ----------------------------------------------------------------------------
# (5) Hopfield-Tank TSP
# ----------------------------------------------------------------------------
def tsp_weights(D, A, B, C, D_):
    """Weight matrix T[(x,i),(y,j)] of Hopfield & Tank (1985), eq. (8)."""
    n = len(D)
    T = np.zeros((n, n, n, n))
    I_ = np.eye(n)
    for x in range(n):
        for i in range(n):
            for y in range(n):
                for j in range(n):
                    T[x, i, y, j] = (-A * I_[x, y] * (1 - I_[i, j])
                                     - B * I_[i, j] * (1 - I_[x, y])
                                     - C
                                     - D_ * D[x, y] * (I_[j, (i + 1) % n] + I_[j, (i - 1) % n]))
    return T.reshape(n * n, n * n)


def hopfield_tank(D, rng, A=500, B=500, C=200, Dd=500, n_eff=15, u0=0.02, dt=1e-5, steps=8000):
    """Continuous dynamics du/dt = -u/tau + T V + I,  V = (1 + tanh(u/u0))/2, tau = 1.
    The inhibitory terms are applied in vectorised form (equivalent to T @ V)."""
    n = len(D)
    u = -u0 / 2 * np.log(n - 1) + 0.001 * rng.uniform(-1, 1, (n, n))   # sum V ~ n at start
    for _ in range(steps):
        V = 0.5 * (1 + np.tanh(u / u0))
        row = V.sum(1, keepdims=True) - V                 # same city, other positions
        col = V.sum(0, keepdims=True) - V                 # same position, other cities
        tot = V.sum()
        neigh = np.roll(V, -1, axis=1) + np.roll(V, 1, axis=1)
        du = -u - A * row - B * col - C * (tot - n_eff) - Dd * (D @ neigh)
        u = u + dt * du
    V = 0.5 * (1 + np.tanh(u / u0))
    return V


def decode_tour(V):
    n = len(V)
    S = (V > 0.5).astype(int)
    valid = np.all(S.sum(0) == 1) and np.all(S.sum(1) == 1)
    tour = list(np.argmax(V, axis=0))                     # city at each position
    return tour, bool(valid and len(set(tour)) == n)


def tour_len(t, D):
    return float(sum(D[t[i], t[(i + 1) % len(t)]] for i in range(len(t))))


def brute_force_tsp(D):
    n = len(D)
    best, bt = np.inf, None
    for p in itertools.permutations(range(1, n)):
        if p[0] > p[-1]:
            continue
        t = (0,) + p
        L = tour_len(t, D)
        if L < best:
            best, bt = L, t
    return list(bt), best


def two_opt(t, D):
    t = list(t)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(t) - 1):
            for j in range(i + 1, len(t)):
                a, b, c, d = t[i - 1], t[i], t[j], t[(j + 1) % len(t)]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-12:
                    t[i:j + 1] = t[i:j + 1][::-1]
                    improved = True
    return t


# ----------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(6)
    summary = {}

    # (1) associative memory of 5 glyphs
    pats = np.array([glyph(g) for g in GLYPHS.values()])
    W = hebbian(pats)
    fig, ax = plt.subplots(3, len(pats), figsize=(1.6 * len(pats), 5))
    demo = []
    for c, (name, p) in enumerate(zip(GLYPHS, pats)):
        noisy = flip(p, 20, rng)
        out = recall(W, noisy, rng)
        demo.append(dict(letter=name, bits_wrong_after=int(np.sum(out != p)),
                         E_noisy=float(energy_bipolar(W, noisy)), E_recalled=float(energy_bipolar(W, out))))
        for r, img in enumerate([p, noisy, out]):
            ax[r, c].imshow(img.reshape(N_SIDE, N_SIDE), cmap="gray_r", vmin=-1, vmax=1)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        ax[0, c].set_title(name)
    ax[0, 0].set_ylabel("stored"); ax[1, 0].set_ylabel("20 bits flipped"); ax[2, 0].set_ylabel("recalled")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "hopfield_letters.png"), dpi=150); plt.close()
    summary["letters_demo"] = demo
    overlaps = (pats @ pats.T) / N
    summary["letter_overlaps"] = overlaps.round(2).tolist()
    print("letters demo:", demo)

    # (2) capacity
    cap = capacity_experiment(rng)
    summary["capacity"] = cap
    cap_noisy = capacity_experiment(rng, noise_bits=10)
    summary["capacity_noise10"] = cap_noisy
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    Ps = [r["P"] for r in cap]
    ax.plot(Ps, [r["bit_stability"] for r in cap], "o-", ms=3, label="fraction of stable bits")
    ax.plot(Ps, [r["pattern_stable"] for r in cap], "s-", ms=3, label="pattern is a fixed point")
    ax.plot(Ps, [r["recall"] for r in cap], "^-", ms=3, label="recall ≥ 99 % (clean probe)")
    ax.plot(Ps, [r["recall"] for r in cap_noisy], "v-", ms=3, label="recall ≥ 99 % (10 bits flipped)")
    ax.axvline(0.138 * N, color="k", ls="--", lw=0.8)
    ax.text(0.138 * N + 0.3, 0.05, "0.138N", fontsize=7)
    ax.axvline(N / (2 * np.log(N)), color="gray", ls=":", lw=0.8)
    ax.text(N / (2 * np.log(N)) - 4.6, 0.05, "N/(2 ln N)", fontsize=7)
    ax.set_xlabel("number of stored random patterns P"); ax.set_ylabel("fraction")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "hopfield_capacity.png"), dpi=150); plt.close()
    print("capacity:", [(r["P"], round(r["pattern_stable"], 2), round(r["recall"], 2)) for r in cap])

    # (3) error correction
    ks, curves = error_correction(rng)
    summary["error_correction"] = dict(k=ks, curves={str(P): c for P, c in curves.items()})
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for P, c in curves.items():
        ax.plot(ks, c, "o-", ms=2.5, label=f"P = {P}")
    ax.set_xlabel("bits flipped in the probe (out of 100)"); ax.set_ylabel("P(exact recall)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "hopfield_error_correction.png"), dpi=150); plt.close()
    thresh = {}
    for P, c in curves.items():
        good = [k for k, v in zip(ks, c) if v >= 0.9]
        thresh[str(P)] = max(good) if good else None
    summary["max_bits_corrected_90pct"] = thresh
    print("max bits corrected (>=90% exact recall):", thresh)

    # (4) eight rooks
    Wr, Ir = rook_network()
    res = [run_rook(Wr, Ir, rng) for _ in range(200)]
    summary["rook"] = dict(trials=len(res), valid=sum(r[1] for r in res),
                           mean_sweeps=float(np.mean([r[2] for r in res])),
                           energies=sorted(set(round(r[3], 6) for r in res)),
                           distinct_solutions=len({r[0].tobytes() for r in res if r[1]}),
                           n_weights_nonzero_symmetric=int(np.count_nonzero(np.triu(Wr))),
                           n_weights_total=int(Wr.size))
    print("rook:", summary["rook"])
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.4))
    for a, r in zip(ax, res[:3]):
        a.imshow((np.indices((8, 8)).sum(0) % 2), cmap="Greys", alpha=0.25)
        yy, xx = np.nonzero(r[0])
        a.scatter(xx, yy, marker="s", s=90, color="#d1495b")
        a.set_xticks([]); a.set_yticks([]); a.set_title(f"E = {r[3]:.0f}", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "hopfield_rooks.png"), dpi=150); plt.close()

    # (5) TSP, 10 cities in the unit square
    crng = np.random.default_rng(10)
    cities = crng.random((10, 2))
    D = np.sqrt(((cities[:, None] - cities[None]) ** 2).sum(-1))
    opt_t, opt_L = brute_force_tsp(D)
    T = tsp_weights(D, 500, 500, 200, 500)
    trials = []
    for s in range(50):
        V = hopfield_tank(D, np.random.default_rng(1000 + s))
        tour, valid = decode_tour(V)
        L = tour_len(tour, D) if valid else None
        L2 = tour_len(two_opt(tour, D), D) if valid else None
        trials.append(dict(valid=valid, length=L, after_2opt=L2, tour=[int(c) for c in tour]))
    valid = [t for t in trials if t["valid"]]
    summary["tsp"] = dict(cities=cities.tolist(), optimum=opt_L, optimal_tour=opt_t, trials=len(trials),
                          valid=len(valid),
                          mean_ratio=float(np.mean([t["length"] / opt_L for t in valid])) if valid else None,
                          best_ratio=float(min(t["length"] / opt_L for t in valid)) if valid else None,
                          hit_optimum=int(sum(abs(t["length"] - opt_L) < 1e-9 for t in valid)),
                          n_neurons=100, weight_matrix_entries=int(T.size),
                          nonzero_offdiag_weights=int(np.count_nonzero(T - np.diag(np.diag(T)))),
                          unique_symmetric_weights=int(100 * 99 / 2), biases=100)
    print("tsp:", {k: v for k, v in summary["tsp"].items() if k not in ("cities",)})
    sweep = []
    for Dd in (200, 300, 500, 700):                 # validity vs. tour-quality trade-off
        Ls = []
        for s in range(50):
            tour, ok = decode_tour(hopfield_tank(D, np.random.default_rng(1000 + s), Dd=Dd))
            if ok:
                Ls.append(tour_len(tour, D))
        sweep.append(dict(D=Dd, valid_frac=len(Ls) / 50,
                          mean_ratio=float(np.mean(Ls) / opt_L) if Ls else None,
                          best_ratio=float(min(Ls) / opt_L) if Ls else None))
    summary["tsp_sweep"] = sweep
    print("tsp sweep:", sweep)
    if valid:
        best = min(valid, key=lambda t: t["length"])
        fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.2))
        for a, t, ttl in ((ax[0], best["tour"], f"Hopfield-Tank: {best['length']:.3f}"),
                          (ax[1], opt_t, f"optimum: {opt_L:.3f}")):
            lp = list(t) + [t[0]]
            a.plot(cities[lp, 0], cities[lp, 1], "-o", ms=4)
            for i, (x, y) in enumerate(cities):
                a.annotate(str(i), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
            a.set_title(ttl, fontsize=9); a.set_aspect("equal")
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "hopfield_tsp.png"), dpi=150); plt.close()
    with open(os.path.join(RESULTS, "lab6_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)


if __name__ == "__main__":
    main()
