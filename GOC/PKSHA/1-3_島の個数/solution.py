from collections import deque
H, W = map(int, input().split())
grid = [input() for _ in range(H)]
visited = [[False]*W for _ in range(H)]
count = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#' and not visited[i][j]:
            count += 1
            q = deque([(i, j)])
            visited[i][j] = True
            while q:
                r, c = q.popleft()
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<H and 0<=nc<W and grid[nr][nc]=='#' and not visited[nr][nc]:
                        visited[nr][nc] = True
                        q.append((nr, nc))
print(count)
