"""
CS605 Lab 2 - Plagiarism detection by A* sentence alignment.

State      (i, j): the first i sentences of document A and the first j sentences
           of document B have been aligned.
Start      (0, 0)            Goal  (n_A, n_B)
Actions    ALIGN  (i, j) -> (i+1, j+1)  cost = d(A_i, B_j)  (normalised Levenshtein)
           SKIP_A (i, j) -> (i+1, j)    cost = p
           SKIP_B (i, j) -> (i, j+1)    cost = p
           SKIP_BOTH (i, j) -> (i+1, j+1) cost = 2p (kept for completeness; never
           cheaper than ALIGN because d <= 1 <= 2p when p >= 0.5)
g(n)       accumulated cost of the partial alignment
h(n)       three choices are compared:
           h0 = 0                               (uniform-cost / Dijkstra)
           h1 = p * |(n_A - i) - (n_B - j)|     (unmatched sentences must be skipped)
           h2 = max(h1, H_A, H_B) where H_A = sum over remaining A sentences of
                min(p, min_{j'>=j} LB(A_i', B_j')) and LB is the length lower bound
                |len(a) - len(b)| / max(len(a), len(b)) <= d(a, b).   (H_B symmetric)
           All three are admissible, so A* returns a minimum-cost alignment.

Run:  python lab2/plagiarism_astar.py      (from the code/ directory)
"""
import json
import os
import random
import re
import string
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.search import best_first_search  # noqa: E402
from lab2.test_docs import TEST_CASES  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab2")
PLAGIARISM_THRESHOLD = 0.30     # normalised edit distance at or below -> flagged
SUSPICIOUS_THRESHOLD = 0.50


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def normalise(sentence):
    s = sentence.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


# ----------------------------------------------------------------------------
# Levenshtein distance (dynamic programming, two-row version)
# ----------------------------------------------------------------------------
def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,              # deletion
                           cur[j - 1] + 1,           # insertion
                           prev[j - 1] + (ca != cb)))  # substitution / match
        prev = cur
    return prev[-1]


def normalised_distance(a, b):
    if not a and not b:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


# ----------------------------------------------------------------------------
# Search problem
# ----------------------------------------------------------------------------
class AlignmentProblem:
    def __init__(self, sents_a, sents_b, skip_penalty=0.5, heuristic="h2"):
        self.A, self.B = sents_a, sents_b
        self.nA, self.nB = len(sents_a), len(sents_b)
        self.p = skip_penalty
        self.initial_state = (0, 0)
        self.goal = (self.nA, self.nB)
        self.cache = {}
        self.lev_calls = 0
        self.kind = heuristic
        if heuristic == "h2":
            self._prepare_length_bound()

    # lazily evaluated and cached edit distances: only pairs that A* touches are computed
    def cost(self, i, j):
        key = (i, j)
        if key not in self.cache:
            self.lev_calls += 1
            self.cache[key] = normalised_distance(self.A[i], self.B[j])
        return self.cache[key]

    def _prepare_length_bound(self):
        la = np.array([len(s) for s in self.A], float)[:, None]
        lb = np.array([len(s) for s in self.B], float)[None, :]
        LB = np.abs(la - lb) / np.maximum(np.maximum(la, lb), 1)
        LB = np.minimum(LB, self.p)
        # sufA[i, j] = sum_{i'>=i} min(p, min_{j'>=j} LB[i', j'])
        nA, nB = self.nA, self.nB
        rowmin = np.full((nA + 1, nB + 1), self.p)      # min over j'>=j for each row
        colmin = np.full((nA + 1, nB + 1), self.p)      # min over i'>=i for each column
        for j in range(nB - 1, -1, -1):
            rowmin[:nA, j] = np.minimum(LB[:, j], rowmin[:nA, j + 1])
        for i in range(nA - 1, -1, -1):
            colmin[i, :nB] = np.minimum(LB[i, :], colmin[i + 1, :nB])
        self.HA = np.zeros((nA + 1, nB + 1))
        self.HB = np.zeros((nA + 1, nB + 1))
        for j in range(nB + 1):
            self.HA[:nA, j] = np.cumsum(rowmin[:nA, j][::-1])[::-1]
        for i in range(nA + 1):
            self.HB[i, :nB] = np.cumsum(colmin[i, :nB][::-1])[::-1]

    def heuristic(self, s):
        i, j = s
        if self.kind == "h0":
            return 0.0
        h1 = self.p * abs((self.nA - i) - (self.nB - j))
        if self.kind == "h1":
            return h1
        return max(h1, self.HA[i, j], self.HB[i, j])

    def is_goal(self, s):
        return s == self.goal

    def successors(self, s):
        i, j = s
        if i < self.nA and j < self.nB:
            yield ("ALIGN", (i + 1, j + 1), self.cost(i, j))
        if i < self.nA:
            yield ("SKIP_A", (i + 1, j), self.p)
        if j < self.nB:
            yield ("SKIP_B", (i, j + 1), self.p)


def align(text_a, text_b, skip_penalty=0.5, heuristic="h2"):
    raw_a, raw_b = split_sentences(text_a), split_sentences(text_b)
    A, B = [normalise(s) for s in raw_a], [normalise(s) for s in raw_b]
    prob = AlignmentProblem(A, B, skip_penalty, heuristic)
    t0 = time.perf_counter()
    res = best_first_search(prob, lambda n: n.g + prob.heuristic(n.state), f"A*({heuristic})")
    elapsed = time.perf_counter() - t0
    pairs = []
    for (action, state), (_, prev) in zip(res.solution.path()[1:], res.solution.path()[:-1]):
        i, j = prev
        if action == "ALIGN":
            d = prob.cost(i, j)
            label = ("PLAGIARISED" if d <= PLAGIARISM_THRESHOLD else
                     "suspicious" if d <= SUSPICIOUS_THRESHOLD else "different")
            pairs.append(dict(action=action, i=i, j=j, dist=d, label=label,
                              a=raw_a[i], b=raw_b[j]))
        elif action == "SKIP_A":
            pairs.append(dict(action=action, i=i, j=None, dist=None, label="unmatched", a=raw_a[i], b=""))
        else:
            pairs.append(dict(action=action, i=None, j=j, dist=None, label="unmatched", a="", b=raw_b[j]))
    flagged_b = sum(1 for p in pairs if p["label"] == "PLAGIARISED")
    stats = dict(cost=res.cost, expanded=res.nodes_expanded, generated=res.nodes_generated,
                 lev_calls=prob.lev_calls, total_pairs=prob.nA * prob.nB, seconds=elapsed,
                 plagiarism_ratio=flagged_b / max(len(B), 1))
    return pairs, stats


# ----------------------------------------------------------------------------
# Synthetic benchmark for scaling (random 'sentences' with controlled edits)
# ----------------------------------------------------------------------------
def synthetic_pair(n, rng, edit_rate=0.15):
    vocab = ["water", "stone", "river", "market", "village", "temple", "rain", "season", "road",
             "trader", "farmer", "canal", "cloth", "spice", "fort", "lake", "hill", "city",
             "the", "a", "of", "and", "in", "was", "built", "near", "many", "old", "new"]
    A = [" ".join(rng.choice(vocab) for _ in range(rng.randint(8, 16))) for _ in range(n)]
    B = []
    for s in A:
        r = rng.random()
        if r < 0.1:
            continue                                       # deletion of a sentence
        w = s.split()
        for k in range(len(w)):
            if rng.random() < edit_rate:
                w[k] = rng.choice(vocab)
        B.append(" ".join(w))
        if r > 0.92:
            B.append(" ".join(rng.choice(vocab) for _ in range(rng.randint(8, 16))))  # insertion
    return ". ".join(A) + ".", ". ".join(B) + "."


def main():
    os.makedirs(RESULTS, exist_ok=True)
    out_lines, summary = [], {"test_cases": {}, "scaling": []}
    for name, a, b in TEST_CASES:
        out_lines.append("=" * 90)
        out_lines.append(name)
        per_h = {}
        for h in ("h0", "h1", "h2"):
            pairs, st = align(a, b, heuristic=h)
            per_h[h] = st
        pairs, st = align(a, b, heuristic="h2")
        for p in pairs:
            d = "  -  " if p["dist"] is None else f"{p['dist']:.3f}"
            ia = "-" if p["i"] is None else p["i"] + 1
            jb = "-" if p["j"] is None else p["j"] + 1
            out_lines.append(f"  A{ia!s:<2} B{jb!s:<2} {p['action']:<7} d={d}  {p['label']}")
        dists = [p["dist"] for p in pairs if p["dist"] is not None]
        out_lines.append(f"  total cost = {st['cost']:.3f}, mean aligned distance = {np.mean(dists):.3f}, "
                         f"flagged fraction of B = {st['plagiarism_ratio']:.2f}")
        for h, s in per_h.items():
            out_lines.append(f"  {h}: expanded={s['expanded']:4d} generated={s['generated']:4d} "
                             f"lev_calls={s['lev_calls']:3d}/{s['total_pairs']} cost={s['cost']:.3f} "
                             f"time={s['seconds']*1e3:.2f} ms")
        summary["test_cases"][name] = dict(
            pairs=[{k: v for k, v in p.items() if k not in ("a", "b")} | {"a": p["a"], "b": p["b"]} for p in pairs],
            mean_dist=float(np.mean(dists)), flagged_ratio=st["plagiarism_ratio"],
            search={h: {k: v for k, v in s.items()} for h, s in per_h.items()})

    out_lines.append("\nScaling study on synthetic documents (mean of 3 seeds)")
    out_lines.append(f"{'n':>4} {'heur':>4} {'expanded':>9} {'lev calls':>10} {'pairs':>7} {'ms':>9}")
    for n in (10, 20, 40, 80):
        for h in ("h0", "h1", "h2"):
            acc = []
            for seed in range(3):
                ta, tb = synthetic_pair(n, random.Random(seed))
                _, st = align(ta, tb, heuristic=h)
                acc.append(st)
            row = dict(n=n, h=h, expanded=np.mean([s["expanded"] for s in acc]),
                       lev_calls=np.mean([s["lev_calls"] for s in acc]),
                       pairs=np.mean([s["total_pairs"] for s in acc]),
                       ms=np.mean([s["seconds"] for s in acc]) * 1e3,
                       cost=np.mean([s["cost"] for s in acc]))
            summary["scaling"].append(row)
            out_lines.append(f"{n:>4} {h:>4} {row['expanded']:>9.0f} {row['lev_calls']:>10.0f} "
                             f"{row['pairs']:>7.0f} {row['ms']:>9.1f}")
    text = "\n".join(out_lines)
    print(text)
    with open(os.path.join(RESULTS, "lab2_output.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(RESULTS, "lab2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)


if __name__ == "__main__":
    main()
