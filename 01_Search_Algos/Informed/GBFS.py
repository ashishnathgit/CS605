import heapq

class Node:
    def __init__(self, name, heuristic):
        self.name = name
        self.h = heuristic  # Heuristic value: h(n)
        
    # Python needs this to compare nodes inside the priority queue (heapq)
    def __lt__(self, other):
        return self.h < other.h
        
    def __repr__(self):
        return f"{self.name}(h={self.h})"

def greedy_best_first_search(graph, start, goal):
    # open_list stores tuples: (heuristic_value, node_name)
    # Using heapq makes it a min-priority queue
    open_list = []
    heapq.heappush(open_list, (graph[start].h, start))
    
    # Track visited nodes to prevent infinite loops
    closed_list = set()
    
    # Track paths: child_node -> parent_node
    parent_map = {}
    
    print("--- Starting GBFS Search Steps ---")
    
    while open_list:
        # Dequeue the node with the lowest h(n) value
        current_h, current_name = heapq.heappop(open_list)
        print(f"Expanding Node: {current_name} | h(n) = {current_h}")
        
        # Check if goal is reached
        if current_name == goal:
            print("🎉 Goal found!")
            return reconstruct_path(parent_map, start, goal)
            
        closed_list.add(current_name)
        
        # Explore neighbors
        for neighbor in graph[current_name].neighbors:
            if neighbor in closed_list:
                continue
                
            # Check if neighbor is already queued (to avoid duplicates)
            if neighbor not in [item[1] for item in open_list]:
                parent_map[neighbor] = current_name
                neighbor_h = graph[neighbor].h
                heapq.heappush(open_list, (neighbor_h, neighbor))
                
    print("❌ Path not found.")
    return None

def reconstruct_path(parent_map, start, goal):
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = parent_map[current]
    path.append(start)
    path.reverse()
    return path

# =====================================================================
# Testing the Implementation
# =====================================================================
if __name__ == "__main__":
    # Define a simple graph structure with nodes and their heuristics to 'G'
    class GraphNode(Node):
        def __init__(self, name, heuristic):
            super().__init__(name, heuristic)
            self.neighbors = []

    # 1. Define nodes with their h(n) estimates to Goal 'G'
    nodes = {
        'A': GraphNode('A', heuristic=6),
        'B': GraphNode('B', heuristic=5),
        'C': GraphNode('C', heuristic=2),
        'D': GraphNode('D', heuristic=1),
        'G': GraphNode('G', heuristic=0)  # Goal node always has h=0
    }

    # 2. Define connections (edges)
    nodes['A'].neighbors = ['B', 'C']
    nodes['B'].neighbors = ['D']
    nodes['C'].neighbors = ['D', 'G']
    nodes['D'].neighbors = ['G']

    # 3. Run the search
    start_node = 'A'
    goal_node = 'G'
    final_path = greedy_best_first_search(nodes, start_node, goal_node)
    
    print("\n--- Final Results ---")
    print(f"Path taken from {start_node} to {goal_node}: {' -> '.join(final_path)}")
