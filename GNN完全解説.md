# GNN（グラフニューラルネットワーク）完全解説

> **対象**: GNNをゼロから本気で理解したい人  
> **前提知識**: 線形代数・基本的なDL（MLP, CNN程度）  
> **スタック**: PyTorch + PyTorch Geometric (PyG)

---

## 目次

1. [グラフとは何か（数学的基礎）](#1-グラフとは何か)
2. [なぜGNNが必要か](#2-なぜgnnが必要か)
3. [メッセージパッシングフレームワーク（MPNN）](#3-メッセージパッシングフレームワーク)
4. [GCN — Graph Convolutional Network](#4-gcn)
5. [GAT — Graph Attention Network](#5-gat)
6. [GraphSAGE — Inductive Learning](#6-graphsage)
7. [GIN — Graph Isomorphism Network](#7-gin)
8. [タスク種別と損失設計](#8-タスク種別と損失設計)
9. [PyTorch Geometric 完全実装](#9-pytorch-geometric-完全実装)
10. [過学習・表現力・限界](#10-過学習表現力限界)
11. [発展トピック](#11-発展トピック)

---

## 1. グラフとは何か

### 1-1. 定義

グラフ $\mathcal{G} = (\mathcal{V}, \mathcal{E})$：

- $\mathcal{V}$ : ノード集合（$|\mathcal{V}| = N$）
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ : エッジ集合

各ノード $v_i$ は特徴ベクトル $\mathbf{x}_i \in \mathbb{R}^d$ を持つ。  
まとめて **ノード特徴行列** $\mathbf{X} \in \mathbb{R}^{N \times d}$。

各エッジ $(v_i, v_j)$ も特徴 $\mathbf{e}_{ij}$ を持てる（エッジ属性）。

### 1-2. 隣接行列と度行列

**隣接行列** $\mathbf{A} \in \{0,1\}^{N \times N}$：

```math
A_{ij} = \begin{cases} 1 & \text{if } (v_i, v_j) \in \mathcal{E} \\ 0 & \text{otherwise} \end{cases}
```

**度行列** $\mathbf{D}$（対角行列）：

```math
D_{ii} = \sum_j A_{ij} \quad \text{（ノード } i \text{ の次数）}
```

### 1-3. グラフラプラシアン

**非正規化ラプラシアン**：
```math
\mathbf{L} = \mathbf{D} - \mathbf{A}
```

**正規化ラプラシアン**（GCNで使う）：
```math
\hat{\mathbf{L}} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}
```

$\hat{\mathbf{L}}$ の固有値 $0 \le \lambda_i \le 2$。これがスペクトルGNNの理論的根拠になる。

```python
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
```

### 1-4. 自己ループ付き隣接行列

GNNでは自ノードの情報も集約したいため、**自己ループ**を追加する：

```math
\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}
```

```math
\tilde{\mathbf{D}}_{ii} = \sum_j \tilde{A}_{ij}
```

---

## 2. なぜGNNが必要か

| データ形式 | モデル | 問題点 |
|-----------|--------|--------|
| 格子状画像 | CNN | ✅ 局所性＆並進不変性を活用 |
| 系列データ | RNN/Transformer | ✅ 位置情報を活用 |
| **グラフ** | **GNN** | ノード数可変・順序なし・不規則構造 |

### グラフが自然に現れる問題

- **分子設計**：原子=ノード、結合=エッジ → 薬剤活性予測
- **SNS**：ユーザー=ノード、友達関係=エッジ → スパム検出
- **知識グラフ**：エンティティ=ノード → 推薦システム
- **脳ネットワーク（EEG/fMRI）**：電極/ROI=ノード、機能的結合=エッジ
- **交通網**：交差点=ノード、道路=エッジ → 渋滞予測

### MLPではダメな理由

MLPにグラフを入力しようとすると：
- ノード順序を固定しなければならない（**順序不変性**が壊れる）
- ノード数が変わると使えない
- 構造情報（誰と繋がっているか）を自然に扱えない

---

## 3. メッセージパッシングフレームワーク

すべての主要GNNアーキテクチャは以下の **MPNN（Message Passing Neural Network）** フレームワークに統一できる。

### 3-1. 3ステップの直感

```
各ノードは「近所からメッセージを集めて → 自分の状態を更新する」
これを L 層繰り返すと、L ホップ先の情報が伝わる
```

### 3-2. 数式

第 $\ell$ 層における ノード $i$ の更新：

**Step 1: メッセージ生成**
```math
\mathbf{m}_{ij}^{(\ell)} = \phi_\text{msg}\!\left(\mathbf{h}_i^{(\ell)},\, \mathbf{h}_j^{(\ell)},\, \mathbf{e}_{ij}\right)
```

**Step 2: 集約（Aggregation）**
```math
\mathbf{m}_i^{(\ell)} = \bigoplus_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(\ell)}
```

$\bigoplus$ は **順序不変**な演算：Sum / Mean / Max など

**Step 3: 更新（Update）**
```math
\mathbf{h}_i^{(\ell+1)} = \phi_\text{upd}\!\left(\mathbf{h}_i^{(\ell)},\, \mathbf{m}_i^{(\ell)}\right)
```

初期値：$\mathbf{h}_i^{(0)} = \mathbf{x}_i$（入力特徴）

```
Layer 0        Layer 1         Layer 2
                   ↗ 近傍から集約
  x_i → h_i^(0) → h_i^(1) → h_i^(2)
                   ↘ 集約された情報で更新
```

### 3-3. 受容野（Receptive Field）

$L$ 層積み重ねると、**$L$ ホップ以内**のノード情報が伝わる。

```
L=1: 直接の隣人のみ
L=2: 隣人の隣人まで
L=3: その先まで...
```

**注意**: 層を深くしすぎると **Over-smoothing**（全ノードの表現が均一化）が起きる。→ 後述

---

## 4. GCN

### 4-1. 理論的導出

Kipf & Welling (2017) の **Graph Convolutional Network** はスペクトルグラフ理論から導出される。

1. グラフ信号 $\mathbf{x}$ のフーリエ変換：固有値分解 $\mathbf{L} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^\top$ を利用
2. スペクトルフィルタ $g_\theta(\boldsymbol{\Lambda})$ を1次チェビシェフ多項式で近似
3. 自己ループ付きで正規化 → GCNの更新式

### 4-2. GCNの更新式

```math
\mathbf{H}^{(\ell+1)} = \sigma\!\left( \hat{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \hat{\mathbf{D}}^{-1/2} \mathbf{H}^{(\ell)} \mathbf{W}^{(\ell)} \right)
```

ここで $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$、$\hat{\mathbf{D}}_{ii} = \sum_j \tilde{A}_{ij}$

**ノード単位で書くと**：

```math
\mathbf{h}_i^{(\ell+1)} = \sigma\!\left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\hat{d}_i \hat{d}_j}} \mathbf{W}^{(\ell)} \mathbf{h}_j^{(\ell)} \right)
```

**直感**：隣人の特徴を次数で正規化して平均し、線形変換 → 活性化

### 4-3. GCN の実装

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

# ────────────────────────────────────────────
# スクラッチ実装（原理理解用）
# ────────────────────────────────────────────
class GCNLayerScratch(nn.Module):
    """GCNの1層をスクラッチで実装"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : [N, in_dim]  ノード特徴
        adj : [N, N]       正規化済み隣接行列 (A_tilde_norm)
        """
        # A_hat_norm @ X @ W
        return self.W(adj @ x)


def normalize_adj(A: torch.Tensor) -> torch.Tensor:
    """A → D^{-1/2} (A+I) D^{-1/2}"""
    N = A.size(0)
    A_tilde = A + torch.eye(N, device=A.device)  # 自己ループ追加
    D_inv_sqrt = torch.diag(A_tilde.sum(dim=1).pow(-0.5))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt


class GCNScratch(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNLayerScratch(in_dim, hidden_dim)
        self.conv2 = GCNLayerScratch(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        h = F.relu(self.conv1(x, adj_norm))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, adj_norm)
        return h  # logits [N, num_classes]


# ────────────────────────────────────────────
# PyG実装（実用）
# ────────────────────────────────────────────
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # edge_index: [2, E]  (COO形式のエッジリスト)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # [N, out_channels]


# ────────────────────────────────────────────
# Cora データセットでの学習
# ────────────────────────────────────────────
def train_gcn_cora():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    # data.x         : [2708, 1433]  単語バッグ特徴
    # data.edge_index: [2, 10556]    COO形式エッジ
    # data.y         : [2708]        7クラスラベル
    # data.train_mask: 各ノードの学習/検証/テストフラグ

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
    ).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            acc = (pred[mask] == data.y[mask]).float().mean().item()
            accs.append(acc)
        return accs

    for epoch in range(1, 201):
        loss = train()
        if epoch % 20 == 0:
            train_acc, val_acc, test_acc = test()
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | Test: {test_acc:.3f}")

# train_gcn_cora()  # ← 実行するとCoraでGCN学習
```

---

## 5. GAT

### 5-1. 動機：注意機構の導入

GCNの問題：隣人を次数のみで重み付け → **全隣人を平等に扱う**

GATのアイデア：**隣人ごとに重要度（Attention）を学習**する

### 5-2. アテンション係数の計算

Veličković et al. (2018) の **Graph Attention Network**：

**生スコア**（非対称）：
```math
e_{ij} = \text{LeakyReLU}\!\left(\mathbf{a}^\top \left[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j\right]\right)
```

$\mathbf{a} \in \mathbb{R}^{2d'}$：学習可能なアテンションベクトル、$\|$：連結

**Softmaxで正規化**：
```math
\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}
```

**アテンション加重集約**：
```math
\mathbf{h}_i' = \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j\right)
```

**マルチヘッドアテンション**（安定化）：

```math
\mathbf{h}_i' = \|_{k=1}^{K} \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j\right)
```

最終層では連結の代わりに **平均**をとる。

### 5-3. GAT 実装

```python
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=8, dropout=0.6):
        super().__init__()
        # 第1層: 8ヘッド、出力 hidden_channels*heads
        self.conv1 = GATConv(
            in_channels, hidden_channels,
            heads=heads, dropout=dropout, concat=True
        )
        # 第2層: 1ヘッド（分類層）、出力 out_channels
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels,
            heads=1, dropout=dropout, concat=False
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ────────────────────────────────────────────
# アテンション係数を可視化する
# ────────────────────────────────────────────
class GATWithAttention(nn.Module):
    """アテンション係数を返すGAT"""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads,
                             return_attention_weights=False, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels,
                             heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        # return_attention_weights=True にすると (edge_index, alpha) が返る
        out, (ei, alpha) = self.conv2(x, edge_index,
                                      return_attention_weights=True)
        return out, alpha  # alpha: [E, heads] アテンション係数
```

---

## 6. GraphSAGE

### 6-1. 動機：帰納的学習（Inductive Learning）

GCN・GATの問題：**トランスダクティブ**（学習時に全グラフが必要）

- 新しいノードが追加されると再学習が必要
- 大規模グラフでは全グラフをメモリに乗せられない

GraphSAGE（Hamilton et al., 2017）の解決策：
- **サンプリング**：各ノードの隣人を固定数だけランダムサンプル
- **汎化可能な集約器**を学習 → 未知ノードにも適用可能

### 6-2. 更新式

```math
\mathbf{h}_{\mathcal{N}(i)}^{(\ell)} = \text{AGGREGATE}^{(\ell)}\!\left(\left\{\mathbf{h}_j^{(\ell-1)} : j \in \mathcal{N}_\text{sample}(i)\right\}\right)
```

```math
\mathbf{h}_i^{(\ell)} = \sigma\!\left(\mathbf{W}^{(\ell)} \cdot \left[\mathbf{h}_i^{(\ell-1)} \| \mathbf{h}_{\mathcal{N}(i)}^{(\ell)}\right]\right)
```

最後に $\ell_2$ 正規化。

**AGGREGATE の選択肢**：
| 集約器 | 式 | 特徴 |
|--------|-----|------|
| Mean   | 平均 | シンプル、GCNに近い |
| LSTM   | LSTMで処理 | 表現力高いが順序依存 |
| Max Pooling | 要素ごとmax | 代表的特徴を抽出 |

### 6-3. GraphSAGE 実装

```python
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ────────────────────────────────────────────
# ミニバッチ学習（大規模グラフ向け）
# ────────────────────────────────────────────
from torch_geometric.loader import NeighborLoader

def train_graphsage_minibatch(data, num_classes):
    """NeighborLoaderを使ったミニバッチ学習"""
    # 各層で最大 [25, 10] 隣人をサンプリング
    train_loader = NeighborLoader(
        data,
        num_neighbors=[25, 10],   # [layer1_neighbors, layer2_neighbors]
        batch_size=1024,
        input_nodes=data.train_mask,
        shuffle=True
    )

    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=256,
        out_channels=num_classes
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            # batch_size個のノードのみ損失計算
            loss = F.cross_entropy(out[:batch.batch_size],
                                   batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader):.4f}")
```

---

## 7. GIN

### 7-1. 理論的背景：表現力の限界

**Weisfeiler-Leman (WL) テスト**：グラフ同型性を判定するアルゴリズム。

GNNの表現力は **1-WL テスト以下**であることが証明されている（Xu et al., 2019）。

1-WL テストのアルゴリズム：
```
各ノードにラベルを付与
→ 隣人のラベルをマルチセットとして集約しハッシュ
→ 収束するまで繰り返す
```

**命題**：GNNが WL テストと同等の識別力を持つ条件：
1. 集約関数が **単射（Injective）** であること
2. 更新関数が **単射** であること

Sum集約 + MLP → 単射が近似可能 → **WL テストと同等の最大表現力**

### 7-2. GINの更新式

```math
\mathbf{h}_i^{(\ell)} = \text{MLP}^{(\ell)}\!\left((1 + \epsilon^{(\ell)}) \cdot \mathbf{h}_i^{(\ell-1)} + \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j^{(\ell-1)}\right)
```

$\epsilon$：学習可能パラメータ（または固定値0）

**なぜ Sum か**：Mean・Max は単射でない（多重度情報が失われる）

例：
- ノードA: 隣人=[1,1,1]  → Mean=1, Max=1  ← 区別できない！
- ノードB: 隣人=[1]      → Mean=1, Max=1
- Sum なら A=3, B=1 → 区別可能 ✅

### 7-3. GIN 実装

```python
from torch_geometric.nn import GINConv

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=5, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 各層のMLP（GINの核心）
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            mlp = nn.Sequential(
                nn.Linear(in_ch, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        # ノード表現の更新
        for conv, bn in zip(self.convs, self.bns):
            x = bn(F.relu(conv(x, edge_index)))

        # グラフ分類用: グローバルプーリング
        if batch is not None:
            from torch_geometric.nn import global_add_pool, global_mean_pool
            x = global_add_pool(x, batch)  # [num_graphs, hidden_channels]

        x = F.dropout(F.relu(self.lin1(x)), p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x
```

---

## 8. タスク種別と損失設計

GNNの出力は3つのレベルに対応できる。

### 8-1. ノード分類（Node Classification）

- **出力**: 各ノードのクラス確率 `[N, C]`
- **損失**: Cross-entropy（半教師あり学習が一般的）

```python
# セミサバイズド設定（一部ノードのみラベルあり）
out = model(data.x, data.edge_index)  # [N, C]
loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
```

### 8-2. リンク予測（Link Prediction）

- **出力**: エッジが存在するかのスコア
- **損失**: Binary cross-entropy + Negative Sampling

```python
from torch_geometric.utils import negative_sampling

def link_pred_loss(z, edge_index, num_nodes):
    """
    z          : [N, d] ノード埋め込み
    edge_index : [2, E] 正例エッジ
    """
    # 正例スコア
    pos_score = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)  # 内積

    # 負例をサンプリング（存在しないエッジ）
    neg_edge_index = negative_sampling(
        edge_index, num_nodes=num_nodes,
        num_neg_samples=edge_index.size(1)
    )
    neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

    # BCE Loss
    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
    return pos_loss + neg_loss
```

### 8-3. グラフ分類（Graph Classification）

- **出力**: グラフ単位のクラス確率 `[B, C]`
- **プーリング**: ノード表現をグラフ全体の表現に集約
- **損失**: Cross-entropy

```python
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

class GraphClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        from torch_geometric.nn import GINConv
        mlp1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(),
                              nn.Linear(hidden_channels, hidden_channels))
        mlp2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
                              nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # ノードレベルの特徴
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # グラフレベルに集約: batch=[N] でどのグラフに属するか
        x = global_add_pool(x, batch)  # [num_graphs, hidden_channels]
        return self.classifier(x)      # [num_graphs, out_channels]
```

### 8-4. グラフ分類のデータローダー

```python
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
# MUTAG: 188グラフ（分子）、2クラス（変異誘発性の有無）

loader = DataLoader(dataset[:150], batch_size=32, shuffle=True)

for batch in loader:
    # batch.x         : 全グラフのノード特徴を縦に結合
    # batch.edge_index: 全グラフのエッジ（ノードインデックスはシフト済み）
    # batch.batch     : 各ノードがどのグラフに属するか [total_N]
    # batch.y         : グラフラベル [32]
    print(batch.x.shape, batch.batch.shape, batch.y.shape)
    break
```

---

## 9. PyTorch Geometric 完全実装

### 9-1. インストール・環境構築

```bash
# PyTorch インストール後
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.x.0+cpu.html
```

### 9-2. Data オブジェクトの作り方

```python
from torch_geometric.data import Data
import torch

# グラフ作成
x = torch.tensor([
    [1.0, 2.0],  # ノード0の特徴
    [3.0, 4.0],  # ノード1
    [5.0, 6.0],  # ノード2
    [7.0, 8.0],  # ノード3
], dtype=torch.float)

# COO形式（双方向）
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3],  # 送信元
    [1, 0, 2, 1, 3, 2],  # 送信先
], dtype=torch.long)

# エッジ属性（オプション）
edge_attr = torch.tensor([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]])

# ノードラベル
y = torch.tensor([0, 1, 0, 1])

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

print(data)
# Data(x=[4, 2], edge_index=[2, 6], edge_attr=[6, 1], y=[4])
print(data.num_nodes, data.num_edges)
print(data.is_undirected())
```

### 9-3. カスタムGNNレイヤーの作り方

PyGの `MessagePassing` 基底クラスを使う：

```python
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class CustomGCNConv(MessagePassing):
    """GCNをMessagePassingで実装"""

    def __init__(self, in_channels, out_channels):
        # aggr='add': 集約関数をsumに設定
        # flow='source_to_target': メッセージの流れ方向
        super().__init__(aggr='add', flow='source_to_target')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        # Step 1: 自己ループ追加
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: 線形変換
        x = self.lin(x)  # [N, out_channels]

        # Step 3: 正規化係数を計算
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: メッセージパッシング（propagate内でmessage→aggregate→updateが呼ばれる）
        out = self.propagate(edge_index, x=x, norm=norm)

        return out + self.bias

    def message(self, x_j, norm):
        """
        x_j  : [E, out_channels] 送信元ノードの特徴
        norm : [E] 正規化係数
        """
        return norm.view(-1, 1) * x_j  # 正規化したメッセージ

    def update(self, aggr_out):
        """集約後の更新（ここでは恒等変換）"""
        return aggr_out
```

### 9-4. 完全なパイプライン（訓練・評価・可視化）

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# ────────────────────────────────────────────
# モデル定義
# ────────────────────────────────────────────
class UniversalGNN(nn.Module):
    """GCN / GAT / SAGE を選べるモデル"""
    def __init__(self, in_channels, hidden_channels, out_channels,
                 gnn_type='gcn', num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout

        ConvClass = {'gcn': GCNConv, 'sage': SAGEConv}[gnn_type]

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = hidden_channels if i < num_layers - 1 else out_channels
            self.convs.append(ConvClass(in_ch, out_ch))

    def forward(self, x, edge_index, return_embedding=False):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        if return_embedding:
            return x  # 最終層直前の埋め込み

        x = self.convs[-1](x, edge_index)
        return x


# ────────────────────────────────────────────
# 訓練ループ
# ────────────────────────────────────────────
def run_experiment(gnn_type='gcn', epochs=200):
    dataset = Planetoid(root=f'/tmp/Cora', name='Cora')
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UniversalGNN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        gnn_type=gnn_type,
    ).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_acc': [], 'test_acc': []}

    for epoch in range(1, epochs + 1):
        # 訓練
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 評価
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

            history['train_loss'].append(loss.item())
            history['val_acc'].append(val_acc)
            history['test_acc'].append(test_acc)
            print(f"[{gnn_type.upper()}] Epoch {epoch:03d} | "
                  f"Loss: {loss:.4f} | Val: {val_acc:.3f} | Test: {test_acc:.3f}")

    return model, data, history


# ────────────────────────────────────────────
# t-SNEで埋め込み可視化
# ────────────────────────────────────────────
def visualize_embeddings(model, data, title="GNN Embeddings"):
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index, return_embedding=True)
    embeddings = embeddings.cpu().numpy()
    labels = data.y.cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels,
                          cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('gnn_embeddings.png', dpi=150)
    plt.show()
```

---

## 10. 過学習・表現力・限界

### 10-1. Over-smoothing（過平滑化）

**問題**: 層を重ねると全ノードの表現が同一になる

**原因**: メッセージパッシングは平均化操作 → 繰り返すと収束

```python
import torch
import numpy as np

# Over-smoothingの数値デモ
def demo_oversmoothing(A_norm, x, num_layers=20):
    """正規化隣接行列を繰り返し適用するとどうなるか"""
    h = x.clone()
    dirichlet_energies = []

    for l in range(num_layers):
        h = A_norm @ h  # 隣接行列を適用（線形GCNの近似）
        # ディリクレエネルギー: 隣接ノード間の差分の大きさ
        energy = ((h[A_norm.nonzero()[:, 0]] - h[A_norm.nonzero()[:, 1]])**2).mean().item()
        dirichlet_energies.append(energy)
        if l % 5 == 0:
            print(f"Layer {l:2d}: Dirichlet energy = {energy:.6f}")

    return dirichlet_energies
```

**対策**：
- `DropEdge`：エッジをランダムにドロップ
- `PairNorm`：各層で正規化
- `DeeperGCN`：残差接続
- `APPNP`：伝播と変換を分離

```python
from torch_geometric.nn import APPNP as APPNPConv

class APPNPModel(nn.Module):
    """Approximate Personalized Propagation of Neural Predictions"""
    def __init__(self, in_channels, hidden_channels, out_channels,
                 K=10, alpha=0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        # K: 伝播ステップ数、alpha: テレポート確率
        self.prop = APPNPConv(K=K, alpha=alpha)

    def forward(self, x, edge_index):
        x = F.dropout(F.relu(self.lin1(x)), training=self.training)
        x = self.lin2(x)
        # 特徴変換後に伝播（Over-smoothingを防ぐ）
        x = self.prop(x, edge_index)
        return x
```

### 10-2. Over-squashing（過圧縮）

**問題**: 遠いノードの情報が指数的に「圧縮」されてボトルネックを通る

**原因**: ボトルネックノード（ブリッジ）に情報が集中

**対策**: グラフの再配線（`AddRandomWalkPE`、`GraphTransformer`）

### 10-3. 表現力の限界（1-WL の壁）

1-WLテストで区別できないグラフ：

```
グラフA: ノード数4の完全二部グラフ (K_{2,2})
グラフB: ノード数4の4サイクル (C_4)
→ 同じ次数列 [2,2,2,2] → GCN/GAT/GraphSAGEは区別不能！
```

**対策**（1-WLを超える手法）：
- **k-WL GNN**: 複数ノードのタプルを扱う（計算量大）
- **ランダム特徴注入**: ノードにランダムIDを付与
- **位置符号化**: ランダムウォーク特徴など
- **Subgraph GNN**: 部分グラフを考慮

---

## 11. 発展トピック

### 11-1. Graph Transformer

Transformerの自己注意をグラフに適用：

```python
from torch_geometric.nn import TransformerConv

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels // heads,
                                     heads=heads, dropout=0.1, concat=True)
        self.conv2 = TransformerConv(hidden_channels, out_channels,
                                     heads=1, concat=False)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        x = self.norm1(F.relu(self.conv1(x, edge_index)))
        x = self.norm2(self.conv2(x, edge_index))
        return x
```

### 11-2. 階層的プーリング（Hierarchical Pooling）

グラフの粗化（Coarsening）でグローバル構造を学習：

```python
from torch_geometric.nn import DiffPool

# DiffPool: 微分可能なソフトクラスタリング
# ノードをクラスタに割り当て → より粗いグラフを作成
# 複数の分子グラフで階層的特徴を学習可能
```

### 11-3. 時空間GNN（EEG・脳ネットワーク向け）

```python
from torch_geometric.nn import GCNConv

class SpatioTemporalGNN(nn.Module):
    """
    EEGなど時系列グラフデータ向け
    空間: GCN（電極間の関係）
    時間: GRU（時系列）
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_nodes, seq_len):
        super().__init__()
        self.gcn = GCNConv(in_channels, hidden_channels)
        self.gru = nn.GRU(hidden_channels * num_nodes, hidden_channels,
                          batch_first=True)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_seq, edge_index):
        """
        x_seq     : [B, T, N, F]  (batch, time, nodes, features)
        edge_index: [2, E]
        """
        B, T, N, F = x_seq.shape
        spatial_feats = []

        for t in range(T):
            x_t = x_seq[:, t].reshape(B * N, F)
            h_t = F.relu(self.gcn(x_t, edge_index))  # [B*N, H]
            h_t = h_t.reshape(B, N * h_t.size(-1))   # [B, N*H]
            spatial_feats.append(h_t)

        # [B, T, N*H]
        spatial_feats = torch.stack(spatial_feats, dim=1)

        # 時系列方向に GRU
        _, h_last = self.gru(spatial_feats)  # h_last: [1, B, H]
        return self.classifier(h_last.squeeze(0))  # [B, out_channels]
```

### 11-4. 自己教師あり学習（Self-Supervised）

ラベルなしグラフからの事前学習：

```python
# GraphCL (Graph Contrastive Learning) の概念実装
class GraphCL(nn.Module):
    """グラフ対照学習 - ラベルなしで表現を学習"""
    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        hidden_dim = 256  # encoderの出力次元に合わせる
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def augment(self, data):
        """グラフの拡張（データ拡張）"""
        import random
        from torch_geometric.utils import dropout_edge, mask_feature

        # エッジドロップ（20%）
        edge_index1, _ = dropout_edge(data.edge_index, p=0.2)
        # 特徴マスク（10%）
        x1, _ = mask_feature(data.x, p=0.1)

        edge_index2, _ = dropout_edge(data.edge_index, p=0.2)
        x2, _ = mask_feature(data.x, p=0.1)

        return (x1, edge_index1), (x2, edge_index2)

    def nt_xent_loss(self, z1, z2, temperature=0.5):
        """NT-Xent Loss（対照損失）"""
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N = z1.size(0)

        z = torch.cat([z1, z2], dim=0)  # [2N, D]
        sim = z @ z.T / temperature      # [2N, 2N]

        # 自分自身とのコサイン類似度を除外
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -1e9)

        # 正例ペア（i と i+N）のインデックス
        labels = torch.arange(N, device=z.device)
        labels = torch.cat([labels + N, labels])

        return F.cross_entropy(sim, labels)
```

---

## まとめ：アーキテクチャ選択ガイド

| タスク | 推奨モデル | 理由 |
|--------|-----------|------|
| ノード分類（小規模） | GCN, GAT | シンプル、精度高 |
| ノード分類（大規模） | GraphSAGE | ミニバッチ対応 |
| グラフ分類（高精度） | GIN | 最大表現力（WL同等） |
| 長距離依存 | Graph Transformer | グローバルな注意機構 |
| 時系列グラフ | ST-GNN | 空間+時間の両方を捉える |
| ラベルなし | GraphCL, GraphMAE | 自己教師あり事前学習 |

## 参考文献

- Kipf & Welling (2017): **Semi-Supervised Classification with GCN** [GCN]
- Veličković et al. (2018): **Graph Attention Networks** [GAT]
- Hamilton et al. (2017): **Inductive Representation Learning** [GraphSAGE]
- Xu et al. (2019): **How Powerful are GNNs?** [GIN]
- Vaswani et al. / Dwivedi et al.: **Graph Transformer**
- You et al. (2020): **Graph Contrastive Learning** [GraphCL]
- PyG Documentation: https://pytorch-geometric.readthedocs.io/
