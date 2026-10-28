"""
CS605 Lab 1 - State-space search with BFS and DFS.

Problem A: Missionaries and Cannibals (3M, 3C, boat of capacity 2).
Problem B: Rabbit Leap (3 east-bound, 3 west-bound rabbits, 7 stones).

Run:  python lab1/lab1_bfs_dfs.py        (from the code/ directory)
"""
import json
import os
import sys
import tracemalloc
from math import factorial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.search import breadth_first_search, depth_first_search, count_reachable  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab1")


# ----------------------------------------------------------------------------
# Problem A: Missionaries and Cannibals
# ----------------------------------------------------------------------------
class MissionariesCannibals:
    """State = (M_left, C_left, B) where B = 1 if the boat is on the left bank."""

    BOAT_LOADS = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]   # (missionaries, cannibals) in boat

    def __init__(self, n=3, capacity=2):
        self.n = n
        self.initial_state = (n, n, 1)
        self.goal_state = (0, 0, 0)
        self.loads = [(m, c) for m in range(capacity + 1) for c in range(capacity + 1)
                      if 1 <= m + c <= capacity]

    def is_safe(self, m, c):
        n = self.n
        if not (0 <= m <= n and 0 <= c <= n):
            return False
        left_ok = (m == 0) or (m >= c)
        right_ok = (n - m == 0) or (n - m >= n - c)
        return left_ok and right_ok

    def is_goal(self, s):
        return s == self.goal_state

    def successors(self, s):
        m, c, b = s
        sign = -1 if b == 1 else 1          # boat on left carries people to the right
        for dm, dc in self.loads:
            m2, c2 = m + sign * dm, c + sign * dc
            if self.is_safe(m2, c2):
                side = "L->R" if b == 1 else "R->L"
                yield (f"{dm}M{dc}C {side}", (m2, c2, 1 - b), 1)

    def describe(self, s):
        m, c, b = s
        left = "M" * m + "C" * c
        right = "M" * (self.n - m) + "C" * (self.n - c)
        lb, rb = ("<B>", "   ") if b else ("   ", "<B>")
        return f"{s}  left={left or '-':<6} {lb} ~~~ {rb} right={right or '-'}"

    def state_space_counts(self):
        total = (self.n + 1) ** 2 * 2
        valid = sum(1 for m in range(self.n + 1) for c in range(self.n + 1) for b in (0, 1)
                    if self.is_safe(m, c))
        return total, valid


# ----------------------------------------------------------------------------
# Problem B: Rabbit Leap
# ----------------------------------------------------------------------------
class RabbitLeap:
    """State = string over {E, W, _} of length 2k+1.

    'E' rabbits face east and may only move right; 'W' rabbits move left.
    A rabbit may step one stone forward onto the empty stone, or jump over
    exactly one rabbit (of either kind) onto the empty stone two stones ahead.
    """

    def __init__(self, k=3):
        self.k = k
        self.initial_state = "E" * k + "_" + "W" * k
        self.goal_state = "W" * k + "_" + "E" * k

    def is_goal(self, s):
        return s == self.goal_state

    def successors(self, s):
        e = s.index("_")
        out = []
        # east-bound rabbits sit to the left of the gap
        for d in (1, 2):
            i = e - d
            if i >= 0 and s[i] == "E" and (d == 1 or s[i + 1] != "_"):
                out.append((f"E@{i}->{e} ({'step' if d == 1 else 'jump'})", self._swap(s, i, e), 1))
        # west-bound rabbits sit to the right of the gap
        for d in (1, 2):
            i = e + d
            if i < len(s) and s[i] == "W" and (d == 1 or s[i - 1] != "_"):
                out.append((f"W@{i}->{e} ({'step' if d == 1 else 'jump'})", self._swap(s, i, e), 1))
        return out

    @staticmethod
    def _swap(s, i, j):
        lst = list(s)
        lst[i], lst[j] = lst[j], lst[i]
        return "".join(lst)

    def describe(self, s):
        return " ".join(s)

    def state_space_counts(self):
        n = 2 * self.k + 1
        return factorial(n) // (factorial(self.k) ** 2), None


def dead_end_states(problem):
    """Reachable states (other than the goal) with no successors."""
    from collections import deque
    seen, q, dead = {problem.initial_state}, deque([problem.initial_state]), []
    while q:
        s = q.popleft()
        succ = list(problem.successors(s))
        if not succ and not problem.is_goal(s):
            dead.append(s)
        for _, s2, _ in succ:
            if s2 not in seen:
                seen.add(s2)
                q.append(s2)
    return dead


def run_with_memory(fn, problem):
    tracemalloc.start()
    res = fn(problem)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    res.extra["peak_kib"] = peak / 1024
    return res


def report(name, problem, res, lines):
    lines.append(f"\n--- {name}: {res.algorithm} ---")
    if not res.found:
        lines.append("no solution")
        return
    lines.append(f"solution length = {res.length} moves, expanded = {res.nodes_expanded}, "
                 f"generated = {res.nodes_generated}, max frontier = {res.max_frontier}, "
                 f"explored/reached = {res.max_explored}, time = {res.seconds*1e3:.3f} ms, "
                 f"peak memory = {res.extra['peak_kib']:.1f} KiB")
    for step, (action, state) in enumerate(res.solution.path()):
        lines.append(f"  {step:2d}. {action or 'start':<22} {problem.describe(state)}")


def all_solution_lengths(problem, limit=10**6):
    """Enumerate every loop-free path to the goal by DFS (small spaces only)."""
    lengths = []
    stack = [(problem.initial_state, (problem.initial_state,))]
    while stack and len(lengths) < limit:
        s, path = stack.pop()
        if problem.is_goal(s):
            lengths.append(len(path) - 1)
            continue
        for _, s2, _ in problem.successors(s):
            if s2 not in path:
                stack.append((s2, path + (s2,)))
    return lengths


def main():
    os.makedirs(RESULTS, exist_ok=True)
    lines, summary = [], {}
    for name, prob in [("Missionaries & Cannibals", MissionariesCannibals()),
                       ("Rabbit Leap", RabbitLeap(3))]:
        total, valid = prob.state_space_counts()
        reach = count_reachable(prob)
        dead = dead_end_states(prob)
        sols = all_solution_lengths(prob)
        lines.append("=" * 78)
        lines.append(f"{name}")
        lines.append(f"  raw state-space size  = {total}" + (f", valid (safe) states = {valid}" if valid else ""))
        lines.append(f"  reachable from start  = {reach}, dead-end states = {len(dead)}")
        lines.append(f"  loop-free solution paths: {len(sols)}; lengths = {sorted(set(sols))}")
        bfs = run_with_memory(breadth_first_search, prob)
        dfs = run_with_memory(depth_first_search, prob)
        report(name, prob, bfs, lines)
        report(name, prob, dfs, lines)
        summary[name] = {
            "raw_states": total, "valid_states": valid, "reachable": reach, "dead_ends": len(dead),
            "num_solution_paths": len(sols), "solution_lengths": sorted(set(sols)),
            **{alg.algorithm: {"length": alg.length, "expanded": alg.nodes_expanded,
                               "generated": alg.nodes_generated, "max_frontier": alg.max_frontier,
                               "explored": alg.max_explored, "ms": alg.seconds * 1e3,
                               "peak_kib": alg.extra["peak_kib"],
                               "path": [a for a, _ in alg.solution.path()[1:]]}
               for alg in (bfs, dfs)}}
    # scaling study for rabbit leap with k rabbits per side
    lines.append("\nRabbit-leap scaling (k rabbits on each side)")
    lines.append(f"{'k':>2} {'|S|':>8} {'reach':>7} {'len':>4} {'BFS exp':>8} {'BFS maxQ':>8} {'DFS exp':>8} {'DFS maxStk':>10}")
    scaling = []
    for k in range(1, 7):
        p = RabbitLeap(k)
        b, d = breadth_first_search(p), depth_first_search(p)
        row = dict(k=k, raw=p.state_space_counts()[0], reach=count_reachable(p), length=b.length,
                   bfs_exp=b.nodes_expanded, bfs_front=b.max_frontier,
                   dfs_exp=d.nodes_expanded, dfs_front=d.max_frontier, dfs_len=d.length)
        scaling.append(row)
        lines.append(f"{k:>2} {row['raw']:>8} {row['reach']:>7} {row['length']:>4} {row['bfs_exp']:>8} "
                     f"{row['bfs_front']:>8} {row['dfs_exp']:>8} {row['dfs_front']:>10}")
    summary["rabbit_scaling"] = scaling
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RESULTS, "lab1_output.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(RESULTS, "lab1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
