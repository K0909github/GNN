# コスト付き迷路の最短路

- 出典: PKSHA
- 問題ID: 2-3
- レベル: C
- 目安時間: 40〜50分
- タグ: ダイクストラ法, グラフ

## 問題文
H 行 W 列のグリッドがあり、マス (i, j) に入るコストは aᵢⱼ です。左上 (0, 0) から右下 (H-1, W-1) まで上下左右に移動するとき、通過するマスの入室コスト合計の最小値を出力してください（スタート地点のコストも含む）。

## 制約
- 1 ≤ H, W ≤ 1000
- 1 ≤ aᵢⱼ ≤ 10⁹

## 入出力例
### 例1
```text
入力:
3 3
1 3 1
1 5 1
4 2 1

出力:
7
```
注釈: (0,0)→(0,2)→(1,2)→(2,2): 1+1+1+1+1+1+1=7

### 例2
```text
入力:
1 1
5

出力:
5
```
注釈: スタートゴールが同じ

### 例3
```text
入力:
2 2
1 2
3 4

出力:
7
```
注釈: 1→2→4=7、1→3→4=8 より最小は7

## ヒント
dist[i][j] を (0,0) からマス (i,j) までの最小コストとして管理します。優先度付きキューが必要です。

## 解説
ダイクストラ法の適用です。

【Python 例】
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

計算量: O(HW log HW)
