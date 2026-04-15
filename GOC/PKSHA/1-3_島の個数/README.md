# 島の個数

- ??: PKSHA
- ??ID: 1-3
- ???: C
- ????: 30〜40分
- ??: BFS, DFS, グラフ

## ???
H 行 W 列のグリッドが与えられます。各マスは '#'（陸地）または '.'（海）のいずれかです。上下左右に隣接する陸地のまとまりを「島」と定義します。グリッド中の島の個数を出力してください。

## ??
- 1 ≤ H, W ≤ 1000
- 各マスは '#' または '.'

## ?s
### ?1
```text
??:
4 5
##...
##...
...##
...##

??:
2
```
??: 左上と右下にそれぞれ1つ

### ?2
```text
??:
3 3
#.#
.#.
#.#

??:
5
```
??: 各 '#' は孤立

### ?3
```text
??:
2 2
##
##

??:
1
```
??: 全部繋がっている

## ???
未訪問の '#' を見つけるたびに BFS/DFS で繋がったマスを全て訪問済みにしていきます。

## ??
BFS を使う典型問題です。

【Python 例】
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

計算量: O(H×W)
