import heapq

# Define the Node class to wrap states, paths, costs, and parent pointers
class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    # Helper method to allow Priority Queue to sort based on the evaluation function f
    def __lt__(self, other):
        # Tie-breaker handles evaluation ordering using an external priority metric
        return False 

# A wrapper around heapq to represent the ordered priority queue (frontier)
class PriorityQueue:
    def __init__(self, f):
        self.heap = []
        self.f = f  # The evaluation function f(node)
        self.counter = 0  # To ensure unique ordering and serve as a tie-breaker

    def add(self, node):
        # We push a tuple: (evaluation value, unique counter, node)
        heapq.heappush(self.heap, (self.f(node), self.counter, node))
        self.counter += 1

    def pop(self):
        _, _, node = heapq.heappop(self.heap)
        return node

    def is_empty(self):
        return len(self.heap) == 0

    def __contains__(self, node):
        return any(item[2].state == node.state for item in self.heap)


def best_first_search(problem, f):
    # node <- NODE(STATE=problem.INITIAL)
    node = Node(state=problem.initial)
    
    # frontier <- a priority queue ordered by f, with node as an element
    frontier = PriorityQueue(f)
    frontier.add(node)
    
    # reached <- a lookup table, with one entry with key problem.INITIAL and value node
    reached = {problem.initial: node}
    
    # while not IS-EMPTY(frontier) do
    while not frontier.is_empty():
        # node <- POP(frontier)
        node = frontier.pop()
        
        # if problem.IS-GOAL(node.STATE) then return node
        if problem.is_goal(node.state):
            return node
            
        # for each child in EXPAND(problem, node) do
        for child in expand(problem, node):
            s = child.state
            
            # if s is not in reached or child.PATH-COST < reached[s].PATH-COST then
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
                
    # return failure
    return None


def expand(problem, node):
    # s <- node.STATE
    s = node.state
    
    # for each action in problem.ACTIONS(s) do
    for action in problem.actions(s):
        # s' <- problem.RESULT(s, action)
        s_prime = problem.result(s, action)
        
        # cost <- node.PATH-COST + problem.ACTION-COST(s, action, s')
        cost = node.path_cost + problem.action_cost(s, action, s_prime)
        
        # yield NODE(STATE=s', PARENT=node, ACTION=action, PATH-COST=cost)
        yield Node(state=s_prime, parent=node, action=action, path_cost=cost)


class GraphProblem:
    def __init__(self, graph, initial, goal):
        self.graph = graph
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        return self.graph[state]

    def result(self, state, action):
        return action[0]

    def action_cost(self, state, action, result):
        return action[1]

    def is_goal(self, state):
        return state == self.goal


def path_from(node):
    path = []
    while node is not None:
        path.append(node.state)
        node = node.parent
    return list(reversed(path))


if __name__ == "__main__":
    graph = {
        "A": [("B", 1), ("C", 4)],
        "B": [("D", 2)],
        "C": [("D", 1)],
        "D": [("G", 3)],
        "G": [],
    }
    problem = GraphProblem(graph, initial="A", goal="G")
    result = best_first_search(problem, f=lambda node: node.path_cost)

    if result is None:
        print("No path found.")
    else:
        print("Path:", " -> ".join(path_from(result)))
        print("Cost:", result.path_cost)
