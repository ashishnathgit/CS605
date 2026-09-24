from collections import deque

# Check if a state is safe and within the rules
def is_valid(m_left, c_left):
    m_right = 3 - m_left
    c_right = 3 - c_left

    # Check boundaries (can't have negative people or more than 3)
    if m_left < 0 or m_left > 3 or c_left < 0 or c_left > 3:
        return False
        
    # Check if cannibals outnumber missionaries on the left bank
    if m_left > 0 and c_left > m_left:
        return False
        
    # Check if cannibals outnumber missionaries on the right bank
    if m_right > 0 and c_right > m_right:
        return False
        
    return True

def bfs_solve():
    # Initial state: (3 Missionaries, 3 Cannibals, Boat on left bank=1)
    start_state = (3, 3, 1)
    goal_state = (0, 0, 0)
    
    # Standard queue and visited set (similar to std::queue and std::set in C++)
    queue = deque([start_state])
    visited = {start_state}
    
    # All possible moves for (Missionaries, Cannibals) in the boat
    possible_moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]
    
    step_count = 0
    
    while queue:
        # Loop through all nodes at the current tree level
        level_size = len(queue)
        print(f"--- Level {step_count} Processing ---")
        
        for _ in range(level_size):
            m, c, b = queue.popleft()
            print(f"Visiting State -> Left Bank: ({m}, {c}, {b})")
            
            if (m, c, b) == goal_state:
                print(f"\n🎉 Goal Reached at Level {step_count}!")
                return
                
            # Try all boat combinations
            for m_boat, c_boat in possible_moves:
                if b == 1: # Boat moving Left -> Right
                    next_m = m - m_boat
                    next_c = c - c_boat
                    next_b = 0
                else:      # Boat moving Right -> Left
                    next_m = m + m_boat
                    next_c = c + c_boat
                    next_b = 1
                    
                next_state = (next_m, next_c, next_b)
                
                # If valid and not visited, push to queue (std::queue::push)
                if is_valid(next_m, next_c) and (next_state not in visited):
                    visited.add(next_state)
                    queue.append(next_state)
                    
        step_count += 1

# Run the program
bfs_solve()
