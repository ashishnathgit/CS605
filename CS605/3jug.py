from collections import deque

def is_valid(state):
    jug1, jug2, jug3 = state
    if jug1>8 or jug1 < 0:
        return False
    if jug2 > 5 or jug2 < 0:
        return False
    if jug3 > 3 or jug3 < 0:
        return False
    return True

def get_successors(state):
    successors = []
    jug1, jug2, jug3 = state
    moves = [(1,-1,0),(1,0,-1),(-1,1,0),(-1,0,1),(0,1,-1),(0,-1,1)]
    for move in moves:
        new_state = (jug1+move[0], jug2+move[1], jug3+move[2])
        if is_valid(new_state):
            successors.append(new_state)
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
        if state == goal_state: #check_goal_state(state):
            return path
        for successor in get_successors(state):
            queue.append((successor, path))
    return None

start_state = (8, 0, 0)
goal_state = (5,3,0)

solution = bfs(start_state, goal_state)
if solution:
    print("Solution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")