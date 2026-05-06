# コース受講可能判定

- 出典: GOC
- 問題ID: 3
- カテゴリ: BFS/DFS
- 難易度: Medium
- サブタイトル: LeetCode 207

## 問題文
numCoursesコース（0〜numCourses-1）があり、prerequisites[i]=[a,b]は「aを受けるにはbが必要」を意味する。全コースを修了できるか判定せよ。

## 例
```text
入力: numCourses=2, prerequisites=[[1,0],[0,1]]
出力: False  # 循環依存あり
```

## ヒント
有向グラフのサイクル検出。DFSで各ノードに「未訪問/訪問中/訪問済」の3状態を持たせる。訪問中のノードに再訪したらサイクル。

## 計算量
時間: O(V+E)　空間: O(V+E)
