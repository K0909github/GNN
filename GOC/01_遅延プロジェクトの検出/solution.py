from collections import deque

def find_delayed_projects(n, edges):
    graph = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    # in_degree==0 のノードからBFS
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    visited = set()
    
    while queue:
        node = queue.popleft()
        visited.add(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # visitedされなかった = サイクル内
    return sorted([i for i in range(n) if i not in visited])

# Test
print(find_delayed_projects(4, [[0,1],[1,2],[2,0],[3,1]]))
# [0, 1, 2]
