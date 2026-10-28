import math
from queue import PriorityQueue

# ==========================================
# 1. KINEMATICS & UTILITY FUNCTIONS
# ==========================================

def tool_position(theta1, theta2):
    """
    Forward Kinematics: Calculates the (x, y) tool position.
    Link lengths are 1 unit each.
    """
    rad1 = math.radians(theta1)
    rad2 = math.radians(theta2)
    
    # End of Link 1 (Elbow)
    elbow_x = math.cos(rad1)
    elbow_y = math.sin(rad1)
    
    # End of Link 2 (Tool tip)
    tool_x = elbow_x + math.cos(rad1 + rad2)
    tool_y = elbow_y + math.sin(rad1 + rad2)
    
    return (tool_x, tool_y)

def inside_any(p, obstacles):
    """
    Checks if a point p=(x, y) is inside any rectangular obstacle.
    Obstacle format: (x_min, x_max, y_min, y_max)
    """
    px, py = p
    for (x_min, x_max, y_min, y_max) in obstacles:
        if x_min <= px <= x_max and y_min <= py <= y_max:
            return True
    return False

def ang_dist(a, b):
    """Calculates the absolute shortest angular distance handling 360 wrap."""
    d = abs(a - b) % 360
    return min(d, 360 - d)

def compute_h(c, goals):
    """
    Heuristic H(c): Minimum over all goal cells of the 
    maximum angular distance among both joints.
    """
    min_h = float('inf')
    c_t1, c_t2 = c
    for (g_t1, g_t2) in goals:
        dist1 = ang_dist(c_t1, g_t1)
        dist2 = ang_dist(c_t2, g_t2)
        h_val = max(dist1, dist2)
        if h_val < min_h:
            min_h = h_val
    return min_h

def get_neighbors(c, free_grid):
    """Generates valid, free 8-way adjacent neighbors with wrapping."""
    c_t1, c_t2 = c
    neighbors = []
    for d1 in [-1, 0, 1]:
        for d2 in [-1, 0, 1]:
            if d1 == 0 and d2 == 0:
                continue
            n_t1 = (c_t1 + d1) % 360
            n_t2 = (c_t2 + d2) % 360
            if free_grid[n_t1][n_t2]:
                neighbors.append((n_t1, n_t2))
    return neighbors

# ==========================================
# 2. PATH PLANNING CORE ALGORITHMS
# ==========================================

def a_star(starts, goals, free_grid):
    """
    A* algorithm implementation matching standard priority queue graph search.
    """
    # Priority Queue stores: (f_score, unique_counter, current_cell)
    # unique_counter breaks ties if f_scores match
    frontier = PriorityQueue()
    counter = 0
    
    g = {}
    parent = {}
    
    # Initialize all start nodes
    for s in starts:
        g[s] = 0
        parent[s] = None
        f_score = 0 + compute_h(s, goals)
        frontier.put((f_score, counter, s))
        counter += 1
        
    closed_set = set()
    
    while not frontier.empty():
        f, _, c = frontier.get()
        
        if c in closed_set:
            continue
        closed_set.add(c)
        
        # Goal test
        if c in goals:
            # Reconstruct path by following parents back to start
            path = []
            curr = c
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()  # Reverse to make it start -> goal
            return path
            
        for n in get_neighbors(c, free_grid):
            g_new = g[c] + 1  # Standard step cost = 1
            
            if n not in g or g_new < g[n]:
                g[n] = g_new
                parent[n] = c
                f_new = g_new + compute_h(n, goals)
                frontier.put((f_new, counter, n))
                counter += 1
                
    return "FAILURE"

def plan_arm_path(S, D, obstacles, epsilon=0.05):
    """Main function wrapper to execute configuration space translation and A*."""
    print("Step 1: Building Configuration Space Grid...")
    # Initialize 360x360 boolean grid map
    free_grid = [[True for _ in range(360)] for _ in range(360)]
    
    for t1 in range(360):
        for t2 in range(360):
            p = tool_position(t1, t2)
            if inside_any(p, obstacles):
                free_grid[t1][t2] = False
                
    print("Step 2: Finding valid Start and Goal angle configurations...")
    starts = set()
    goals = set()
    
    for t1 in range(360):
        for t2 in range(360):
            if free_grid[t1][t2]:
                p = tool_position(t1, t2)
                
                # Check distance to Source coordinate
                dist_S = math.sqrt((p[0] - S[0])**2 + (p[1] - S[1])**2)
                if dist_S < epsilon:
                    starts.add((t1, t2))
                    
                # Check distance to Destination coordinate
                dist_D = math.sqrt((p[0] - D[0])**2 + (p[1] - D[1])**2)
                if dist_D < epsilon:
                    goals.add((t1, t2))
                    
    if not starts or not goals:
        return "FAILURE: Source or Destination coordinates are inside obstacles or unreachable."
        
    print(f"Found {len(starts)} start nodes and {len(goals)} goal nodes within coordinate tolerance.")
    print("Step 3: Searching the configuration space grid with A*...")
    return a_star(starts, goals, free_grid)

# ==========================================
# 3. TEST EXECUTION RUN
# ==========================================
if __name__ == "__main__":
    # Define Source S and Destination D matching your assignment text
    source_point = (2.0, 0.0)
    dest_point = (-1.0, -1.0)
    
    # Define an obstacle blocking the path in the workspace
    # Format: (x_min, x_max, y_min, y_max)
    # This creates a vertical obstacle strip in the path
    # workspace_obstacles = [(0.2, 0.5, -1.5, 1.5)]
    # Change this line to an empty list for the zero-obstacles scenario
    # workspace_obstacles = []
    # This places a block right where the arm naturally wants to sweep downward
    workspace_obstacles = [(1.0, 2.2, -1.2, -0.2)]


    
    # Execute Path Planning
    generated_plan = plan_arm_path(source_point, dest_point, workspace_obstacles, epsilon=0.08)
    
    if isinstance(generated_plan, list):
        print(f"\nSUCCESS! Path found consisting of {len(generated_plan)} steps.")
        print(f"Start configuration angles: {generated_plan[0]}")
        print(f"Goal configuration angles: {generated_plan[-1]}")
        print("\nFirst 10 steps of motor angles sequence:")
        print(generated_plan[:10])
    else:
        print(f"\n{generated_plan}")
