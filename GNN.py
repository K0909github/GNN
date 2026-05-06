import numpy as np

# 簡単なグラフの例：5ノード
edges = [(0,1), (1,2), (2,3), (3,4), (4,0), (0,2)]
N = 5

A = np.zeros((N, N))
for i, j in edges:
    A[i,j] = A[j,i] = 1.0

D = np.diag(A.sum(axis=1))
L = D - A  # 非正規化ラプラシアン

# 正規化ラプラシアン
D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
L_norm = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt

print("隣接行列 A:\n", A)
print("正規化ラプラシアン固有値:", np.linalg.eigvalsh(L_norm).round(3))