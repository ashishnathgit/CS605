from collections import deque

CAPACITIES = (8, 5, 3)  # index 0 = 8L, 1 = 5L, 2 = 3L

def is_valid(state):
    for amount, capacity in zip(state, CAPACITIES):
        if amount < 0 or amount > capacity:
            return False
    return True

def get_successors(state):
    successors = []
    state = list(state)
    n = len(state)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if state[i] == 0 or state[j] == CAPACITIES[j]:
                continue  # nothing to pour, or destination already full
            pour_amount = min(state[i], CAPACITIES[j] - state[j])
            new_state = state.copy()
            new_state[i] -= pour_amount
            new_state[j] += pour_amount
            if is_valid(tuple(new_state)):
                successors.append(tuple(new_state))
    return successors

def bfs(start_state, goal_state):
    queue = deque([(start_state, [])])
    visited = set()
    while queue:
        (state, path) = queue.popleft()
        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        if state == goal_state:
            return path
        for successor in get_successors(state):
            queue.append((successor, path))
    return None

start_state = (8, 0, 0)

# ----------------------------------------------------
# CHANGE THIS to whatever goal state you want to test
# Must be a valid (jug1, jug2, jug3) combo within capacities (8, 5, 3)
goal_state = (5, 3, 0)
# ----------------------------------------------------

solution = bfs(start_state, goal_state)
if solution:
    print(f"Solution found in {len(solution)-1} moves:")
    for step in solution:
        print(step)
else:
    print("No solution found.")