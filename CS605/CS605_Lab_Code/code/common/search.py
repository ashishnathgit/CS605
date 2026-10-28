"""
Generic graph-search agent used across Labs 1-3 (CS605, IIIT Vadodara).

The agent only needs a *problem* object exposing
    initial_state, is_goal(state), successors(state) -> iterable of (action, next_state, step_cost)
and optionally heuristic(state).  The same frontier/explored loop then runs
BFS, DFS, uniform-cost search, greedy best-first and A* by swapping the
frontier discipline (Russell & Norvig, AIMA 4e, Sec. 3.3-3.5).
"""
from __future__ import annotations

import heapq
import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Optional


@dataclass
class Node:
    state: Hashable
    parent: Optional["Node"] = None
    action: Any = None
    g: float = 0.0
    depth: int = 0

    def path(self):
        """Backtrack through parent pointers: returns [(action, state), ...] from the root."""
        node, out = self, []
        while node is not None:
            out.append((node.action, node.state))
            node = node.parent
        return out[::-1]


@dataclass
class SearchResult:
    solution: Optional[Node]
    nodes_expanded: int
    nodes_generated: int
    max_frontier: int
    max_explored: int
    seconds: float
    algorithm: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def found(self):
        return self.solution is not None

    @property
    def cost(self):
        return None if self.solution is None else self.solution.g

    @property
    def length(self):
        return None if self.solution is None else self.solution.depth


def breadth_first_search(problem) -> SearchResult:
    """BFS graph search with goal test on generation (AIMA Fig. 3.9)."""
    t0 = time.perf_counter()
    root = Node(problem.initial_state)
    if problem.is_goal(root.state):
        return SearchResult(root, 0, 1, 1, 0, time.perf_counter() - t0, "BFS")
    frontier = deque([root])
    reached = {root.state}            # hash table: states already generated
    expanded, generated = 0, 1
    max_front = 1
    while frontier:
        node = frontier.popleft()     # FIFO queue
        expanded += 1
        for action, s2, c in problem.successors(node.state):
            if s2 in reached:
                continue
            child = Node(s2, node, action, node.g + c, node.depth + 1)
            generated += 1
            if problem.is_goal(s2):
                return SearchResult(child, expanded, generated, max_front, len(reached),
                                    time.perf_counter() - t0, "BFS")
            reached.add(s2)
            frontier.append(child)
        max_front = max(max_front, len(frontier))
    return SearchResult(None, expanded, generated, max_front, len(reached), time.perf_counter() - t0, "BFS")


def depth_first_search(problem, depth_limit: Optional[int] = None) -> SearchResult:
    """DFS graph search with a LIFO stack and an explored set (avoids loops).

    Successors are pushed in reverse so that they are expanded in the order the
    problem generates them, making the trace easy to follow by hand.
    """
    t0 = time.perf_counter()
    root = Node(problem.initial_state)
    stack = [root]
    explored = set()
    on_stack = {root.state}
    expanded = 0
    generated = 1
    max_front = 1
    while stack:
        node = stack.pop()
        on_stack.discard(node.state)
        if problem.is_goal(node.state):
            return SearchResult(node, expanded, generated, max_front, len(explored),
                                time.perf_counter() - t0, "DFS")
        if node.state in explored:
            continue
        explored.add(node.state)
        expanded += 1
        if depth_limit is not None and node.depth >= depth_limit:
            continue
        children = []
        for action, s2, c in problem.successors(node.state):
            if s2 in explored or s2 in on_stack:
                continue
            children.append(Node(s2, node, action, node.g + c, node.depth + 1))
        generated += len(children)
        for ch in reversed(children):
            stack.append(ch)
            on_stack.add(ch.state)
        max_front = max(max_front, len(stack))
    return SearchResult(None, expanded, generated, max_front, len(explored), time.perf_counter() - t0, "DFS")


def best_first_search(problem, f: Callable[[Node], float], name="BestFirst",
                      max_expansions: Optional[int] = None) -> SearchResult:
    """Priority-queue graph search (AIMA Fig. 3.7). f = g -> UCS, f = h -> greedy, f = g + h -> A*."""
    t0 = time.perf_counter()
    counter = itertools.count()       # tie-breaker keeps heap entries comparable
    root = Node(problem.initial_state)
    frontier = [(f(root), next(counter), root)]
    reached = {root.state: root}      # best node found so far for each state
    expanded = 0
    generated = 1
    max_front = 1
    while frontier:
        _, _, node = heapq.heappop(frontier)
        if reached.get(node.state) is not node:
            continue                  # stale entry (a cheaper path was found later)
        if problem.is_goal(node.state):
            return SearchResult(node, expanded, generated, max_front, len(reached),
                                time.perf_counter() - t0, name)
        expanded += 1
        if max_expansions and expanded > max_expansions:
            break
        for action, s2, c in problem.successors(node.state):
            child = Node(s2, node, action, node.g + c, node.depth + 1)
            generated += 1
            old = reached.get(s2)
            if old is None or child.g < old.g:
                reached[s2] = child
                heapq.heappush(frontier, (f(child), next(counter), child))
        max_front = max(max_front, len(frontier))
    return SearchResult(None, expanded, generated, max_front, len(reached), time.perf_counter() - t0, name)


def uniform_cost_search(problem, **kw):
    return best_first_search(problem, lambda n: n.g, "UCS", **kw)


def greedy_best_first(problem, **kw):
    return best_first_search(problem, lambda n: problem.heuristic(n.state), "Greedy", **kw)


def astar_search(problem, **kw):
    return best_first_search(problem, lambda n: n.g + problem.heuristic(n.state), "A*", **kw)


def count_reachable(problem) -> int:
    """Exhaustive BFS over the reachable state space (no goal test)."""
    seen = {problem.initial_state}
    q = deque([problem.initial_state])
    while q:
        s = q.popleft()
        for _, s2, _ in problem.successors(s):
            if s2 not in seen:
                seen.add(s2)
                q.append(s2)
    return len(seen)
