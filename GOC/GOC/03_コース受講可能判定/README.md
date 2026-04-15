# コース受講可能判定

- ??: GOC
- ??ID: 3
- ????: BFS/DFS
- ???: Medium
- ??????: LeetCode 207

## ???
numCoursesコース（0〜numCourses-1）があり、prerequisites[i]=[a,b]は「aを受けるにはbが必要」を意味する。全コースを修了できるか判定せよ。

## ?
```text
入力: numCourses=2, prerequisites=[[1,0],[0,1]]
出力: False  # 循環依存あり
```

## ???
有向グラフのサイクル検出。DFSで各ノードに「未訪問/訪問中/訪問済」の3状態を持たせる。訪問中のノードに再訪したらサイクル。

## ???
時間: O(V+E)　空間: O(V+E)
