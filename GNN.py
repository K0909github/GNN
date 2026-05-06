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

train_gcn_cora()  # ← 実行するとCoraでGCN学習