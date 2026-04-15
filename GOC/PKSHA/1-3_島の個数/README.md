# 島の個数

- 出典: PKSHA
- 問題ID: 1-3
- レベル: C
- 目安時間: 30〜40分
- タグ: BFS, DFS, グラフ

## 問題文
H 行 W 列のグリッドが与えられます。各マスは '#'（陸地）または '.'（海）のいずれかです。上下左右に隣接する陸地のまとまりを「島」と定義します。グリッド中の島の個数を出力してください。

## 制約
- 1 ≤ H, W ≤ 1000
- 各マスは '#' または '.'

## 入出力例
### 例1
```text
入力:
4 5
##...
##...
...##
...##

出力:
2
```
注釈: 左上と右下にそれぞれ1つ

### 例2
```text
入力:
3 3
#.#
.#.
#.#

出力:
5
```
注釈: 各 '#' は孤立

### 例3
```text
入力:
2 2
##
##

出力:
1
```
注釈: 全部繋がっている

## ヒント
未訪問の '#' を見つけるたびに BFS/DFS で繋がったマスを全て訪問済みにしていきます。

## 解説
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
