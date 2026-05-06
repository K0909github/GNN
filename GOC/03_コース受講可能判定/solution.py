def canFinish(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    for a, b in prerequisites:
        graph[b].append(a)
    
    # 0: 未訪問, 1: 訪問中, 2: 訪問済
    state = [0] * numCourses
    
    def dfs(node):
        if state[node] == 1:  # サイクル検出
            return False
        if state[node] == 2:  # 処理済み
            return True
        
        state[node] = 1
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        state[node] = 2
        return True
    
    return all(dfs(i) for i in range(numCourses))

print(canFinish(2, [[1,0],[0,1]]))  # False
print(canFinish(2, [[1,0]]))        # True
