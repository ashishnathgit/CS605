"""
CS605 Lab 3 - Uniform random k-SAT generation and local search.

Part B: generate_ksat(k, m, n)  -> uniform random k-SAT instance
Part C: Hill-Climbing, Beam Search (w = 3, 4), Variable Neighbourhood Descent
        (3 neighbourhoods), each with two heuristic functions, compared on
        success rate, nodes generated, and penetrance.

State     : a complete truth assignment x in {0,1}^n
Goal test : every clause has at least one true literal
Heuristics (maximised):
   h1(x) = number of satisfied clauses
   h2(x) = h1(x) + (1/(m+1)) * #clauses with >= 2 true literals
           (same ordering as h1 except that ties are broken in favour of
            assignments whose satisfied clauses are 'robust', i.e. would survive
            a single flip; this reduces the plateaus where h1 gives no guidance)
Penetrance (Khemani, Ch. 3): P = L / T, where L = length of the path (number of
moves) from the start state to the solution and T = total number of nodes
generated.  Failed runs have P = 0.
"""
import numpy as np


# ----------------------------------------------------------------------------
# Part B: instance generator
# ----------------------------------------------------------------------------
def generate_ksat(k, m, n, rng):
    """Each clause: k distinct variables, each negated with probability 1/2.

    Returns (vars, signs): int arrays of shape (m, k); signs[c, j] = 1 means the
    literal is positive (x_v), 0 means negated (~x_v).
    """
    if k > n:
        raise ValueError("k must not exceed n")
    var = np.array([rng.choice(n, size=k, replace=False) for _ in range(m)])
    sign = rng.integers(0, 2, size=(m, k))
    return var, sign


def to_dimacs(var, sign, n):
    lines = [f"p cnf {n} {len(var)}"]
    for vs, ss in zip(var, sign):
        lits = [(v + 1) if s else -(v + 1) for v, s in zip(vs, ss)]
        lines.append(" ".join(map(str, lits)) + " 0")
    return "\n".join(lines)


def pretty(var, sign):
    cl = []
    for vs, ss in zip(var, sign):
        cl.append("(" + " v ".join(("" if s else "~") + f"x{v+1}" for v, s in zip(vs, ss)) + ")")
    return " ^ ".join(cl)


def is_satisfiable(var, sign, n):
    """Ground truth with MiniSat (via python-sat) - used only to label instances."""
    from pysat.solvers import Minisat22
    with Minisat22() as s:
        for vs, ss in zip(var, sign):
            s.add_clause([int(v + 1) if b else -int(v + 1) for v, b in zip(vs, ss)])
        return s.solve()


# ----------------------------------------------------------------------------
# Evaluation of (batches of) assignments
# ----------------------------------------------------------------------------
class Evaluator:
    def __init__(self, var, sign, heuristic="h1"):
        self.var, self.sign = var, sign.astype(bool)
        self.m = len(var)
        self.heuristic = heuristic
        self.generated = 0                         # T in the penetrance formula

    def true_counts(self, X):
        X = np.atleast_2d(X)
        return (X[:, self.var] == self.sign).sum(axis=2)      # (N, m)

    def score(self, X, count=True):
        t = self.true_counts(X)
        if count:
            self.generated += t.shape[0]
        sat = (t >= 1).sum(axis=1).astype(float)
        if self.heuristic == "h2":
            sat = sat + (t >= 2).sum(axis=1) / (self.m + 1)
        return sat, (t >= 1).all(axis=1)

    def n_sat(self, x):
        return int((self.true_counts(x) >= 1).sum())


def flip_neighbours(x, flips):
    """flips: list of tuples of variable indices -> matrix of neighbour assignments."""
    X = np.repeat(x[None, :], len(flips), axis=0)
    for r, f in enumerate(flips):
        X[r, list(f)] ^= True
    return X


# ----------------------------------------------------------------------------
# Search algorithms
# ----------------------------------------------------------------------------
def hill_climbing(ev, n, rng, max_steps=1000, x0=None):
    """Steepest-ascent hill climbing on the 1-flip neighbourhood; stops at a local optimum."""
    x = rng.integers(0, 2, n).astype(bool) if x0 is None else x0.copy()
    cur, done = ev.score(x)
    cur, done = cur[0], done[0]
    flips = [(v,) for v in range(n)]
    steps = 0
    while not done and steps < max_steps:
        X = flip_neighbours(x, flips)
        sc, sat = ev.score(X)
        best = np.flatnonzero(sc == sc.max())
        b = rng.choice(best)
        if sc[b] <= cur:
            break                                   # local maximum / plateau
        x, cur, done = X[b], sc[b], sat[b]
        steps += 1
    return x, done, steps


def beam_search(ev, n, rng, width=3, max_steps=1000):
    """Local beam search: keep the `width` best distinct states among all 1-flip successors."""
    beam = rng.integers(0, 2, (width, n)).astype(bool)
    sc, sat = ev.score(beam)
    depth = np.zeros(width, dtype=int)
    if sat.any():
        return beam[np.argmax(sat)], True, 0
    flips = [(v,) for v in range(n)]
    best_so_far, stall = sc.max(), 0
    for step in range(1, max_steps + 1):
        cand = np.concatenate([flip_neighbours(b, flips) for b in beam])
        cdepth = np.repeat(depth + 1, n)
        csc, csat = ev.score(cand)
        if csat.any():
            i = np.flatnonzero(csat)[0]
            return cand[i], True, int(cdepth[i])
        # unique rows, random tie-breaking, keep top `width`
        keys = np.packbits(cand, axis=1)
        _, uniq = np.unique(keys, axis=0, return_index=True)
        order = uniq[np.lexsort((rng.random(len(uniq)), -csc[uniq]))]
        sel = order[:width]
        beam, sc, depth = cand[sel], csc[sel], cdepth[sel]
        if sc.max() > best_so_far:
            best_so_far, stall = sc.max(), 0
        else:
            stall += 1
            if stall >= 2 * n:                       # no progress: give up
                break
    return beam[0], False, int(depth[0])


def vnd(ev, n, rng, max_steps=1000):
    """Variable Neighbourhood Descent with
         N1: flip one variable                          (n neighbours)
         N2: flip two variables                         (n(n-1)/2 neighbours)
         N3: flip three variables taken from the unsatisfied clauses
    Returns to N1 after every improving move; stops when N3 fails to improve."""
    x = rng.integers(0, 2, n).astype(bool)
    cur, done = ev.score(x)
    cur, done = cur[0], done[0]
    N1 = [(v,) for v in range(n)]
    N2 = [(a, b) for a in range(n) for b in range(a + 1, n)]
    steps, k = 0, 1
    while not done and steps < max_steps:
        if k == 1:
            flips = N1
        elif k == 2:
            flips = N2
        else:
            t = ev.true_counts(x)[0]
            cand = np.unique(ev.var[t == 0].ravel())
            flips = [(a, b, c) for ai, a in enumerate(cand) for bi, b in enumerate(cand[ai + 1:], ai + 1)
                     for c in cand[bi + 1:]]
            if not flips:
                break
        X = flip_neighbours(x, flips)
        sc, sat = ev.score(X)
        b = rng.choice(np.flatnonzero(sc == sc.max()))
        if sc[b] > cur:
            x, cur, done = X[b], sc[b], sat[b]
            steps += 1
            k = 1
        else:
            k += 1
            if k > 3:
                break
    return x, done, steps


ALGORITHMS = {
    "HC": lambda ev, n, rng: hill_climbing(ev, n, rng),
    "Beam-3": lambda ev, n, rng: beam_search(ev, n, rng, width=3),
    "Beam-4": lambda ev, n, rng: beam_search(ev, n, rng, width=4),
    "VND": lambda ev, n, rng: vnd(ev, n, rng),
}
