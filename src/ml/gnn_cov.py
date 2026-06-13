import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


def build_graph(ret_window, corr_threshold=0.3):
    """
    Build a stock correlation graph.
    - Nodes: stocks
    - Edges: pairs of stocks with |correlation| > threshold
    - Edge weight: correlation value

    Returns PyG Data object.
    """
    corr_matrix = ret_window.corr().fillna(0).values
    n = corr_matrix.shape[0]

    # Build edge list from correlation matrix
    edge_index = []
    edge_weight = []
    for i in range(n):
        for j in range(n):
            if i != j and abs(corr_matrix[i, j]) > corr_threshold:
                edge_index.append([i, j])
                edge_weight.append(corr_matrix[i, j])

    if len(edge_index) == 0:
        # Fallback: connect each node to itself
        edge_index = [[i, i] for i in range(n)]
        edge_weight = [1.0] * n

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

    # Node features: per-stock statistics (vol, mean return, skew)
    node_features = np.column_stack([
        ret_window.mean().values,
        ret_window.std().values,
        ret_window.skew().values,
    ])
    node_features = torch.tensor(node_features, dtype=torch.float32)
    node_features = F.normalize(node_features, dim=0)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)


class GATCovariance(nn.Module):
    """
    Graph Attention Network that learns a per-stock embedding.
    The outer product of embeddings gives a learned covariance-like matrix.
    """

    def __init__(self, in_channels=3, hidden=32, out_channels=16, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden, heads=heads, dropout=0.2, concat=True)
        self.gat2 = GATConv(hidden * heads, out_channels, heads=1, dropout=0.2, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden * heads)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.bn1(self.gat1(x, edge_index)))
        x = self.gat2(x, edge_index)
        return x  # node embeddings: (n_stocks, out_channels)


def embeddings_to_covariance(embeddings):
    """
    Convert GNN embeddings to a positive semi-definite covariance matrix.
    Method: C = Z @ Z.T where Z = normalised embeddings.
    This guarantees PSD (required for the MV optimiser).
    """
    Z = F.normalize(embeddings, dim=1)
    C = Z @ Z.T
    # Add small diagonal regularisation for numerical stability
    C = C + 0.01 * torch.eye(C.shape[0])
    return C


def train_gnn(ret_wide, epochs=100, lr=1e-3):
    """
    Train GNN to predict next-period stock returns from graph structure.
    Returns trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training GNN on: {device}")

    tickers = ret_wide.columns.tolist()
    dates = ret_wide.index

    model = GATCovariance().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    for epoch in range(epochs):
        total_loss = 0
        count = 0

        # Use rolling 252-day windows, stepping monthly
        for t in range(252, len(dates) - 21, 21):
            window = ret_wide.iloc[t - 252:t]
            target = ret_wide.iloc[t:t + 21].mean()  # 21-day forward return

            graph = build_graph(window).to(device)
            target_tensor = torch.tensor(
                target[tickers].fillna(0).values, dtype=torch.float32
            ).to(device)

            optimizer.zero_grad()
            embeddings = model(graph)
            # Predict returns from embeddings (simple readout)
            pred = embeddings.mean(dim=1)
            loss = criterion(pred, target_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "data/gnn_best.pt")

    model.load_state_dict(torch.load("data/gnn_best.pt"))
    print(f"GNN training done. Best loss: {best_loss:.6f}")
    return model


def get_gnn_covariance(model, ret_window):
    """
    Get GNN-learned covariance matrix for a return window.
    Returns numpy array, shape (n_stocks, n_stocks).
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        graph = build_graph(ret_window).to(device)
        embeddings = model(graph)
        cov_tensor = embeddings_to_covariance(embeddings)
    return cov_tensor.cpu().numpy()