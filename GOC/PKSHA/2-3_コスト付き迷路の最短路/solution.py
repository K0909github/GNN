import heapq
H, W = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(H)]
INF = float('inf')
dist = [[INF]*W for _ in range(H)]
dist[0][0] = A[0][0]
hq = [(A[0][0], 0, 0)]
while hq:
    d, r, c = heapq.heappop(hq)
    if d > dist[r][c]:
        continue
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0<=nr<H and 0<=nc<W:
            nd = d + A[nr][nc]
            if nd < dist[nr][nc]:
                dist[nr][nc] = nd
                heapq.heappush(hq, (nd, nr, nc))
print(dist[H-1][W-1])
