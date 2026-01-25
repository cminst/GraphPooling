"""
A compact PyTorch Geometric benchmark runner for TU graph classification:
- 10-fold outer cross-validation (StratifiedKFold)
- Per-fold validation split (StratifiedShuffleSplit) for early stopping + light tuning
- One script supports many pooling/readout methods.

Models:
  TopKPool, SAGPool, ASAP, EdgePool, MinCutPool, DiffPool, Graclus,
  GMT (GraphMultisetTransformer), Global-Attention, SortPool, Set2Set, GCN, GIN

Datasets (PyG TUDataset):
  DD, PROTEINS, NCI1, NCI109, FRANKENSTEIN, MUTAG, REDDIT-BINARY,
  IMDB-BINARY, IMDB-MULTI, COLLAB

Usage example:
  python gnn_pooling_benchmark_cv.py --dataset PROTEINS --model TopKPool --tune
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Conv1d, Linear, ModuleList, ReLU, Sequential

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# -----------------------------
# PyG imports (version-tolerant)
# -----------------------------
try:
    from torch_geometric.datasets import TUDataset
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "This script requires PyTorch Geometric. "
        "Install instructions: https://pytorch-geometric.readthedocs.io/"
    ) from e

try:
    from torch_geometric.loader import DataLoader, DenseDataLoader
except Exception:  # pragma: no cover
    from torch_geometric.data import DataLoader  # type: ignore
    from torch_geometric.loader import DenseDataLoader  # type: ignore

from torch_geometric.data import Data

import torch_geometric.transforms as T
from torch_geometric.utils import degree, subgraph

from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GraphConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

# Pooling layers
from torch_geometric.nn import TopKPooling, SAGPooling, ASAPooling, EdgePooling

# Dense pooling + dense convs
try:
    from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, dense_mincut_pool
except Exception:  # pragma: no cover
    from torch_geometric.nn.dense import DenseSAGEConv  # type: ignore
    from torch_geometric.nn.dense import dense_diff_pool, dense_mincut_pool  # type: ignore

# Deterministic pooling (Graclus)
try:
    from torch_geometric.nn import graclus, max_pool
except Exception:  # pragma: no cover
    from torch_geometric.nn.pool import graclus, max_pool  # type: ignore

# Global attention pooling
try:
    from torch_geometric.nn import GlobalAttention
except Exception:  # pragma: no cover
    from torch_geometric.nn.glob import GlobalAttention  # type: ignore

# Aggregations (Set2Set, SortPool, GMT)
try:
    from torch_geometric.nn.aggr import Set2Set, SortAggregation, GraphMultisetTransformer
except Exception:  # pragma: no cover
    # Older versions exposed Set2Set at torch_geometric.nn (and did not have SortAggregation/GMT).
    from torch_geometric.nn import Set2Set  # type: ignore
    SortAggregation = None  # type: ignore
    GraphMultisetTransformer = None  # type: ignore


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Small utilities
# -----------------------------
def accuracy(logits: Tensor, y: Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).sum().item() / y.numel())


def ensure_y_flat_long(y: Tensor) -> Tensor:
    return y.view(-1).long()


def maybe_get_num_features(dataset) -> int:
    # TUDataset exposes num_node_features; but it can be 0 while x exists with shape [N, 0].
    data0 = dataset[0]
    if getattr(data0, "x", None) is None:
        return 0
    if data0.x is None:
        return 0
    return int(data0.x.size(-1))


def compute_max_degree(dataset, cap: int = 100) -> int:
    max_deg = 0
    for data in dataset:
        if data.edge_index is None:
            continue
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        max_deg = max(max_deg, int(deg.max().item()))
    return int(min(max_deg, cap))


def compute_max_num_nodes(dataset, cap: Optional[int] = None) -> int:
    max_nodes = 0
    for data in dataset:
        if data.num_nodes is None:
            continue
        max_nodes = max(max_nodes, int(data.num_nodes))
    if cap is not None:
        return int(min(max_nodes, cap))
    return int(max_nodes)


class TruncateToMaxNodes:
    def __init__(self, max_nodes: int, seed: int = 0):
        self.max_nodes = int(max_nodes)
        self.seed = int(seed)

    def __call__(self, data: Data) -> Data:
        if data.num_nodes is None or data.num_nodes <= self.max_nodes:
            return data

        g = torch.Generator()
        g.manual_seed(self.seed + int(data.num_nodes))
        perm = torch.randperm(data.num_nodes, generator=g)[: self.max_nodes]
        perm, _ = perm.sort()

        edge_index, edge_attr = subgraph(
            perm, data.edge_index, data.edge_attr, relabel_nodes=True
        )
        x = data.x[perm] if data.x is not None else None

        out = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=data.y,
            num_nodes=int(perm.numel()),
        )
        return out


class SafeOneHotDegree:
    """
    One-hot encode node degree up to max_degree, and bucket all larger degrees into the last bin.
    This avoids crashes when a dataset has degrees larger than the cap.

    Output feature dim = max_degree + 1.
    """
    def __init__(self, max_degree: int, cat: bool = False):
        self.max_degree = int(max_degree)
        self.cat = bool(cat)

    def __call__(self, data: Data) -> Data:
        if data.edge_index is None:
            deg = torch.zeros((data.num_nodes,), dtype=torch.long)
        else:
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes).to(torch.long)

        deg = deg.clamp(max=self.max_degree)
        x = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)

        if getattr(data, "x", None) is not None and data.x is not None and self.cat:
            data.x = torch.cat([data.x, x], dim=-1)
        else:
            data.x = x
        return data


# -----------------------------
# Configs
# -----------------------------
@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    epochs: int = 300
    patience: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    val_ratio: float = 0.1
    num_workers: int = 0
    tune: bool = False
    max_trials: int = 16
    log_every: int = 25
    save_dir: str = "./runs"
    save_json: bool = True


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    num_conv_layers: int = 3
    dropout: float = 0.0

    # Optional per-model LR override (used in tuning grids)
    lr: Optional[float] = None

    # Common pooling params
    pool_ratio: float = 0.5  # used by TopK/SAG/ASAP
    num_pools: int = 3

    # SortPool params
    sort_k: int = 30
    sort_conv1d_out: int = 32
    sort_conv1d_kernel: int = 5

    # Set2Set params
    set2set_steps: int = 3
    set2set_layers: int = 1

    # GMT params
    gmt_k: int = 30
    gmt_heads: int = 4
    gmt_encoder_blocks: int = 1
    gmt_dropout: float = 0.0
    gmt_layer_norm: bool = False

    # DiffPool / MinCut params
    # 2-layer architecture defaults: first pool to ~25% nodes, then to ~25% again (relative)
    dense_pool_layers: int = 2
    diffpool_ratio_1: float = 0.25
    diffpool_ratio_2: float = 0.25
    mincut_ratio_1: float = 0.5
    mincut_ratio_2: float = 0.5
    diffpool_link_loss_weight: float = 1.0
    diffpool_entropy_loss_weight: float = 1.0
    mincut_loss_weight: float = 1.0
    mincut_ortho_loss_weight: float = 1.0

    # Dense padding control
    dense_max_nodes: Optional[int] = None
    dense_policy: str = "error"  # {"error", "drop", "truncate"}

    # Feature engineering for featureless graphs
    degree_as_feature: bool = True
    degree_onehot_cap: int = 100


# A compact place to adjust defaults per method.
MODEL_DEFAULTS: Dict[str, ModelConfig] = {
    # Hierarchical pooling
    "TopKPool": ModelConfig(hidden_dim=64, pool_ratio=0.5, num_pools=3, dropout=0.0),
    "SAGPool": ModelConfig(hidden_dim=64, pool_ratio=0.5, num_pools=3, dropout=0.0),
    "ASAP": ModelConfig(hidden_dim=64, pool_ratio=0.5, num_pools=3, dropout=0.0),
    "EdgePool": ModelConfig(hidden_dim=64, num_pools=3, dropout=0.2),
    "Graclus": ModelConfig(hidden_dim=64, num_pools=3, dropout=0.0),

    # Dense pooling (defaults include a safe node cap to avoid OOM on large TU graphs)
    "DiffPool": ModelConfig(
        hidden_dim=64,
        dense_pool_layers=2,
        diffpool_ratio_1=0.25,
        diffpool_ratio_2=0.25,
        dense_max_nodes=750,
        dense_policy="truncate",
    ),
    "MinCutPool": ModelConfig(
        hidden_dim=32,
        dense_pool_layers=2,
        mincut_ratio_1=0.5,
        mincut_ratio_2=0.5,
        dense_max_nodes=750,
        dense_policy="truncate",
    ),

    # Readout pooling
    "GMT": ModelConfig(hidden_dim=64, gmt_k=30, gmt_heads=4),
    "Global-Attention": ModelConfig(hidden_dim=64),
    "SortPool": ModelConfig(hidden_dim=64, sort_k=30, sort_conv1d_out=32, sort_conv1d_kernel=5),
    "Set2Set": ModelConfig(hidden_dim=64, set2set_steps=3, set2set_layers=1),

    # Baselines
    "GCN": ModelConfig(hidden_dim=64, num_conv_layers=3, dropout=0.5),
    "GIN": ModelConfig(hidden_dim=64, num_conv_layers=5, dropout=0.5),
}

# Very small, method-specific “light” tuning spaces.
# Each entry is a list of *partial* overrides, merged onto MODEL_DEFAULTS[model].
MODEL_TUNE_GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "TopKPool": [{"pool_ratio": r} for r in [0.4, 0.5, 0.6]],
    "SAGPool": [{"pool_ratio": r} for r in [0.4, 0.5, 0.6]],
    "ASAP": [{"pool_ratio": r} for r in [0.4, 0.5, 0.6]],
    "EdgePool": [{"dropout": d} for d in [0.0, 0.2]],
    "Graclus": [{"dropout": d} for d in [0.0, 0.2]],
    "DiffPool": [
        {"diffpool_ratio_1": 0.25, "diffpool_ratio_2": 0.25},
        {"diffpool_ratio_1": 0.10, "diffpool_ratio_2": 0.25},
    ],
    "MinCutPool": [
        {"mincut_ratio_1": 0.5, "mincut_ratio_2": 0.5},
        {"mincut_ratio_1": 0.3, "mincut_ratio_2": 0.5},
    ],
    "GMT": [{"gmt_k": k, "gmt_heads": h} for k in [16, 30, 50] for h in [1, 4]],
    "Global-Attention": [{"dropout": d} for d in [0.0, 0.5]],
    "SortPool": [{"sort_k": k} for k in [20, 30, 50]],
    "Set2Set": [{"set2set_steps": s} for s in [3, 6]],
    "GCN": [{"lr": lr} for lr in [1e-3, 5e-4]],
    "GIN": [{"lr": lr} for lr in [1e-3, 5e-4]],
}


def merge_cfg(base: ModelConfig, override: Dict[str, Any]) -> ModelConfig:
    cfg = copy.deepcopy(base)
    for k, v in override.items():
        if not hasattr(cfg, k):
            raise KeyError(f"Unknown ModelConfig key: {k}")
        setattr(cfg, k, v)
    return cfg


# -----------------------------
# Model implementations
# -----------------------------
class MLP(Sequential):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__(
            Linear(in_dim, hidden_dim),
            ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_dim, out_dim),
        )


class GCNGraphClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin = MLP(hidden_dim, hidden_dim, out_dim, dropout=dropout)
        self.dropout = float(dropout)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.lin(x), x.new_zeros(())


class GINGraphClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.convs = ModuleList()
        for i in range(num_layers):
            mlp = Sequential(
                Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
        self.lin = MLP(hidden_dim, hidden_dim, out_dim, dropout=dropout)
        self.dropout = float(dropout)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        return self.lin(x), x.new_zeros(())


class Set2SetClassifier(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        processing_steps: int,
        set2set_layers: int,
    ):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = float(dropout)
        self.set2set = Set2Set(hidden_dim, processing_steps=processing_steps, num_layers=set2set_layers)
        self.lin = MLP(2 * hidden_dim, hidden_dim, out_dim, dropout=dropout)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.set2set(x, batch)
        return self.lin(x), x.new_zeros(())


class GlobalAttentionClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = float(dropout)
        gate_nn = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, 1))
        self.att = GlobalAttention(gate_nn=gate_nn)
        self.lin = MLP(hidden_dim, hidden_dim, out_dim, dropout=dropout)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.att(x, batch)
        return self.lin(x), x.new_zeros(())


class GMTClassifier(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        k: int,
        heads: int,
        num_encoder_blocks: int,
        layer_norm: bool,
        gmt_dropout: float,
    ):
        super().__init__()
        if GraphMultisetTransformer is None:
            raise RuntimeError("GraphMultisetTransformer is not available in this torch_geometric version.")
        self.convs = ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = float(dropout)
        self.gmt = GraphMultisetTransformer(
            channels=hidden_dim,
            k=int(k),
            num_encoder_blocks=int(num_encoder_blocks),
            heads=int(heads),
            layer_norm=bool(layer_norm),
            dropout=float(gmt_dropout),
        )
        self.lin = MLP(hidden_dim, hidden_dim, out_dim, dropout=dropout)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gmt(x, batch)
        return self.lin(x), x.new_zeros(())


class SortPoolClassifier(torch.nn.Module):
    """
    SortPool + 1D conv head (DGCNN-style readout, simplified).
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        k: int,
        conv1d_out: int,
        conv1d_kernel: int,
    ):
        super().__init__()
        if SortAggregation is None:
            raise RuntimeError("SortAggregation is not available in this torch_geometric version.")
        self.convs = ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = float(dropout)
        self.k = int(k)
        self.sort = SortAggregation(k=self.k)
        self.conv1d = Conv1d(hidden_dim, int(conv1d_out), int(conv1d_kernel))
        out_len = max(1, self.k - int(conv1d_kernel) + 1)
        self.lin1 = Linear(int(conv1d_out) * out_len, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.sort(x, batch)  # [B, k*hidden]
        B = x.size(0)
        x = x.view(B, self.k, -1).transpose(1, 2).contiguous()  # [B, hidden, k]
        x = F.relu(self.conv1d(x))
        x = x.view(B, -1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x, x.new_zeros(())


class HierarchicalPoolClassifier(torch.nn.Module):
    """
    A common 3-block "conv -> pool" architecture used in several pooling baselines:
      - TopKPool
      - SAGPool
      - ASAP
      - EdgePool
      - Graclus

    Graph-level embedding = sum over blocks of [mean_pool || max_pool] (concat).
    """
    def __init__(
        self,
        pool_type: str,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_pools: int,
        pool_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.pool_type = pool_type
        self.dropout = float(dropout)
        self.num_pools = int(num_pools)

        self.convs = ModuleList()
        self.pools = ModuleList()

        for i in range(self.num_pools):
            self.convs.append(GraphConv(in_dim if i == 0 else hidden_dim, hidden_dim))

            if pool_type == "TopKPool":
                self.pools.append(TopKPooling(hidden_dim, ratio=float(pool_ratio)))
            elif pool_type == "SAGPool":
                self.pools.append(SAGPooling(hidden_dim, ratio=float(pool_ratio), GNN=GraphConv))
            elif pool_type == "ASAP":
                self.pools.append(ASAPooling(hidden_dim, ratio=float(pool_ratio)))
            elif pool_type == "EdgePool":
                self.pools.append(EdgePooling(hidden_dim, dropout=float(dropout)))
            elif pool_type == "Graclus":
                self.pools.append(torch.nn.Identity())
            else:
                raise ValueError(f"Unknown pool_type: {pool_type}")

        self.lin1 = Linear(2 * hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        xs: List[Tensor] = []
        for i in range(self.num_pools):
            x = F.relu(self.convs[i](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.pool_type in ("TopKPool", "SAGPool"):
                x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, None, batch)
            elif self.pool_type == "ASAP":
                x, edge_index, _, batch, _ = self.pools[i](x, edge_index, None, batch)
            elif self.pool_type == "EdgePool":
                x, edge_index, batch, _ = self.pools[i](x, edge_index, batch)
            elif self.pool_type == "Graclus":
                cluster = graclus(edge_index, num_nodes=x.size(0))
                pooled = max_pool(cluster, Data(x=x, edge_index=edge_index, batch=batch))
                x, edge_index, batch = pooled.x, pooled.edge_index, pooled.batch
            else:
                raise RuntimeError("Unhandled pool_type")

            xs.append(torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1))

        x = torch.stack(xs, dim=0).sum(dim=0)  # [B, 2*hidden]
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x, x.new_zeros(())


class DenseDiffPoolClassifier(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        max_nodes: int,
        ratio_1: float,
        ratio_2: float,
        link_w: float,
        ent_w: float,
    ):
        super().__init__()
        self.max_nodes = int(max_nodes)
        self.dropout = float(dropout)
        self.link_w = float(link_w)
        self.ent_w = float(ent_w)

        c1 = max(2, int(np.ceil(self.max_nodes * float(ratio_1))))
        c2 = max(2, int(np.ceil(c1 * float(ratio_2))))

        # Separate GNNs for embedding and assignment; DenseSAGEConv ~ GraphSAGE mean.
        self.embed1 = DenseSAGEConv(in_dim, hidden_dim)
        self.pool1 = DenseSAGEConv(in_dim, c1)

        self.embed2 = DenseSAGEConv(hidden_dim, hidden_dim)
        self.pool2 = DenseSAGEConv(hidden_dim, c2)

        self.embed3 = DenseSAGEConv(hidden_dim, hidden_dim)

        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)

        self.lin = MLP(hidden_dim, hidden_dim, out_dim, dropout=dropout)

    def _bn(self, x: Tensor, bn: BatchNorm1d) -> Tensor:
        B, N, C = x.size()
        x = x.view(B * N, C)
        x = bn(x)
        return x.view(B, N, C)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        x, adj, mask = data.x, data.adj, data.mask  # x:[B,N,F], adj:[B,N,N], mask:[B,N]

        # Layer 1
        z1 = F.relu(self.embed1(x, adj, mask))
        z1 = self._bn(z1, self.bn1)
        z1 = F.normalize(z1, p=2.0, dim=-1)
        s1 = self.pool1(x, adj, mask)
        x, adj, link1, ent1 = dense_diff_pool(z1, adj, s1, mask)

        # Layer 2
        z2 = F.relu(self.embed2(x, adj))
        z2 = self._bn(z2, self.bn2)
        z2 = F.normalize(z2, p=2.0, dim=-1)
        s2 = self.pool2(x, adj)
        x, adj, link2, ent2 = dense_diff_pool(z2, adj, s2)

        # Final embedding
        x = F.relu(self.embed3(x, adj))
        x = self._bn(x, self.bn3)
        x = F.normalize(x, p=2.0, dim=-1)

        g = x.mean(dim=1)
        logits = self.lin(g)

        reg = self.link_w * (link1 + link2) + self.ent_w * (ent1 + ent2)
        return logits, reg


class DenseMinCutPoolClassifier(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        max_nodes: int,
        ratio_1: float,
        ratio_2: float,
        mincut_w: float,
        ortho_w: float,
    ):
        super().__init__()
        self.max_nodes = int(max_nodes)
        self.dropout = float(dropout)
        self.mincut_w = float(mincut_w)
        self.ortho_w = float(ortho_w)

        c1 = max(2, int(np.ceil(self.max_nodes * float(ratio_1))))
        c2 = max(2, int(np.ceil(c1 * float(ratio_2))))

        self.embed1 = DenseSAGEConv(in_dim, hidden_dim)
        self.pool1 = DenseSAGEConv(in_dim, c1)

        self.embed2 = DenseSAGEConv(hidden_dim, hidden_dim)
        self.pool2 = DenseSAGEConv(hidden_dim, c2)

        self.embed3 = DenseSAGEConv(hidden_dim, hidden_dim)

        self.lin = MLP(hidden_dim, hidden_dim, out_dim, dropout=dropout)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        x, adj, mask = data.x, data.adj, data.mask

        z1 = F.relu(self.embed1(x, adj, mask))
        s1 = self.pool1(x, adj, mask)
        x, adj, mc1, o1 = dense_mincut_pool(z1, adj, s1, mask)

        z2 = F.relu(self.embed2(x, adj))
        s2 = self.pool2(x, adj)
        x, adj, mc2, o2 = dense_mincut_pool(z2, adj, s2)

        x = F.relu(self.embed3(x, adj))
        g = x.mean(dim=1)
        logits = self.lin(g)

        reg = self.mincut_w * (mc1 + mc2) + self.ortho_w * (o1 + o2)
        return logits, reg


def build_model(model_name: str, in_dim: int, out_dim: int, cfg: ModelConfig, max_nodes: Optional[int]) -> torch.nn.Module:
    if model_name == "GCN":
        return GCNGraphClassifier(in_dim, cfg.hidden_dim, out_dim, cfg.num_conv_layers, cfg.dropout)
    if model_name == "GIN":
        return GINGraphClassifier(in_dim, cfg.hidden_dim, out_dim, cfg.num_conv_layers, cfg.dropout)
    if model_name == "Set2Set":
        return Set2SetClassifier(
            in_dim, cfg.hidden_dim, out_dim, cfg.num_conv_layers, cfg.dropout,
            processing_steps=cfg.set2set_steps, set2set_layers=cfg.set2set_layers
        )
    if model_name == "Global-Attention":
        return GlobalAttentionClassifier(in_dim, cfg.hidden_dim, out_dim, cfg.num_conv_layers, cfg.dropout)
    if model_name == "GMT":
        return GMTClassifier(
            in_dim, cfg.hidden_dim, out_dim, cfg.num_conv_layers, cfg.dropout,
            k=cfg.gmt_k, heads=cfg.gmt_heads, num_encoder_blocks=cfg.gmt_encoder_blocks,
            layer_norm=cfg.gmt_layer_norm, gmt_dropout=cfg.gmt_dropout,
        )
    if model_name == "SortPool":
        return SortPoolClassifier(
            in_dim, cfg.hidden_dim, out_dim, cfg.num_conv_layers, cfg.dropout,
            k=cfg.sort_k, conv1d_out=cfg.sort_conv1d_out, conv1d_kernel=cfg.sort_conv1d_kernel,
        )
    if model_name in ("TopKPool", "SAGPool", "ASAP", "EdgePool", "Graclus"):
        return HierarchicalPoolClassifier(
            model_name, in_dim, cfg.hidden_dim, out_dim,
            num_pools=cfg.num_pools, pool_ratio=cfg.pool_ratio, dropout=cfg.dropout,
        )
    if model_name == "DiffPool":
        if max_nodes is None:
            raise ValueError("DiffPool requires max_nodes (dense padding).")
        return DenseDiffPoolClassifier(
            in_dim=in_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=out_dim,
            dropout=cfg.dropout,
            max_nodes=max_nodes,
            ratio_1=cfg.diffpool_ratio_1,
            ratio_2=cfg.diffpool_ratio_2,
            link_w=cfg.diffpool_link_loss_weight,
            ent_w=cfg.diffpool_entropy_loss_weight,
        )
    if model_name == "MinCutPool":
        if max_nodes is None:
            raise ValueError("MinCutPool requires max_nodes (dense padding).")
        return DenseMinCutPoolClassifier(
            in_dim=in_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=out_dim,
            dropout=cfg.dropout,
            max_nodes=max_nodes,
            ratio_1=cfg.mincut_ratio_1,
            ratio_2=cfg.mincut_ratio_2,
            mincut_w=cfg.mincut_loss_weight,
            ortho_w=cfg.mincut_ortho_loss_weight,
        )
    raise ValueError(f"Unknown model: {model_name}")


def is_dense_model(model_name: str) -> bool:
    return model_name in ("DiffPool", "MinCutPool")


# -----------------------------
# Dataset handling
# -----------------------------
def load_dataset(
    root: str,
    name: str,
    model_name: str,
    cfg: ModelConfig,
    seed: int,
) -> Tuple[Any, int, int, Optional[int]]:
    """
    Returns:
      dataset, in_dim, num_classes, max_nodes (for dense models, else None)
    """
    # Be permissive on TU datasets: keep node + edge attributes when available.
    dataset = None
    for kwargs in (
        dict(use_node_attr=True, use_edge_attr=True),
        dict(use_node_attr=True),
        dict(),
    ):
        try:
            dataset = TUDataset(root=root, name=name, **kwargs)
            break
        except TypeError:
            continue
    if dataset is None:  # pragma: no cover
        dataset = TUDataset(root=root, name=name)

    # If featureless, add one-hot degree features (common baseline on TU).
    data0 = dataset[0]
    has_x = (getattr(data0, "x", None) is not None) and (data0.x is not None) and (data0.x.size(-1) > 0)

    transforms: List[Any] = []
    if (not has_x) and cfg.degree_as_feature:
        # Fixed dim with safe bucketing:
        transforms.append(SafeOneHotDegree(cfg.degree_onehot_cap, cat=False))

    max_nodes = None
    if is_dense_model(model_name):
        dataset_max_nodes = compute_max_num_nodes(dataset)
        max_nodes = int(cfg.dense_max_nodes) if cfg.dense_max_nodes is not None else dataset_max_nodes

        if dataset_max_nodes > max_nodes:
            if cfg.dense_policy == "error":
                raise RuntimeError(
                    f"{model_name} needs dense adjacency. "
                    f"Dataset max_nodes={dataset_max_nodes} exceeds dense_max_nodes={max_nodes}. "
                    f"Set --dense_max_nodes >= {dataset_max_nodes}, or choose --dense_policy drop|truncate."
                )
            elif cfg.dense_policy == "drop":
                kept = [i for i, d in enumerate(dataset) if d.num_nodes <= max_nodes]
                dataset = dataset[kept]
            elif cfg.dense_policy == "truncate":
                transforms.append(TruncateToMaxNodes(max_nodes=max_nodes, seed=seed))
            else:
                raise ValueError(f"Unknown dense_policy: {cfg.dense_policy}")

        transforms.append(T.ToDense(num_nodes=max_nodes))

    if len(transforms) > 0:
        dataset.transform = T.Compose(transforms)

    # After transform, infer feature dim
    in_dim = maybe_get_num_features(dataset)
    if in_dim == 0:
        dataset.transform = (
            T.Compose([dataset.transform, T.Constant(value=1.0)]) if dataset.transform else T.Constant(value=1.0)
        )
        in_dim = maybe_get_num_features(dataset)

    num_classes = int(dataset.num_classes)
    return dataset, in_dim, num_classes, max_nodes


def make_loaders(
    dataset,
    indices: Sequence[int],
    batch_size: int,
    dense: bool,
    shuffle: bool,
    num_workers: int,
):
    subset = dataset[indices]
    if dense:
        return DenseDataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# -----------------------------
# Train/eval
# -----------------------------
@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_graphs = 0

    for data in loader:
        data = data.to(device)
        logits, reg = model(data)
        y = ensure_y_flat_long(data.y)
        loss = F.cross_entropy(logits, y) + reg
        bs = y.numel()
        total_loss += float(loss.item()) * bs
        total_acc += accuracy(logits, y) * bs
        total_graphs += bs

    return total_loss / max(1, total_graphs), total_acc / max(1, total_graphs)


def train_one_setting(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    log_every: int = 25,
) -> Tuple[float, float, Dict[str, Any]]:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_state = None
    best_epoch = 0
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            opt.zero_grad(set_to_none=True)
            logits, reg = model(data)
            y = ensure_y_flat_long(data.y)
            loss = F.cross_entropy(logits, y) + reg
            loss.backward()
            opt.step()

        val_loss, val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        if log_every > 0 and epoch % log_every == 0:
            print(f"  epoch {epoch:4d} | val_acc={val_acc:.4f} | best={best_val_acc:.4f} @ {best_epoch}")

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_loss, final_val_acc = evaluate(model, val_loader, device)
    info = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_val_acc": final_val_acc,
        "final_val_loss": final_val_loss,
    }
    return final_val_loss, final_val_acc, info


def run_cv(
    dataset,
    y_all: np.ndarray,
    model_name: str,
    in_dim: int,
    num_classes: int,
    max_nodes: Optional[int],
    train_cfg: TrainConfig,
    base_model_cfg: ModelConfig,
) -> Dict[str, Any]:
    device = torch.device(train_cfg.device if torch.cuda.is_available() and train_cfg.device.startswith("cuda") else "cpu")
    print(f"[device] {device}")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=train_cfg.seed)

    fold_results: List[Dict[str, Any]] = []
    for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros_like(y_all), y_all), start=1):
        print(f"\n[fold {fold}/10] trainval={len(trainval_idx)} test={len(test_idx)}")

        # Inner split for validation:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=train_cfg.val_ratio, random_state=train_cfg.seed + fold)
        tr_idx_rel, val_idx_rel = next(sss.split(np.zeros_like(trainval_idx), y_all[trainval_idx]))
        train_idx = trainval_idx[tr_idx_rel]
        val_idx = trainval_idx[val_idx_rel]

        dense = is_dense_model(model_name)
        train_loader = make_loaders(dataset, train_idx, train_cfg.batch_size, dense=dense, shuffle=True, num_workers=train_cfg.num_workers)
        val_loader = make_loaders(dataset, val_idx, train_cfg.batch_size, dense=dense, shuffle=False, num_workers=train_cfg.num_workers)
        test_loader = make_loaders(dataset, test_idx, train_cfg.batch_size, dense=dense, shuffle=False, num_workers=train_cfg.num_workers)

        candidates: List[Tuple[ModelConfig, Dict[str, Any]]] = []
        if train_cfg.tune:
            grid = MODEL_TUNE_GRIDS.get(model_name, []) or [dict()]
            grid = grid[: train_cfg.max_trials]
            for override in grid:
                cfg = merge_cfg(base_model_cfg, override)
                candidates.append((cfg, override))
        else:
            candidates.append((base_model_cfg, {}))

        best = None
        for trial, (cfg, override) in enumerate(candidates, start=1):
            print(f"  [trial {trial}/{len(candidates)}] override={override}")
            model = build_model(model_name, in_dim, num_classes, cfg, max_nodes=max_nodes)
            lr = float(cfg.lr) if (hasattr(cfg, "lr") and cfg.lr is not None) else train_cfg.lr

            _, val_acc, info = train_one_setting(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=train_cfg.epochs,
                patience=train_cfg.patience,
                lr=lr,
                weight_decay=train_cfg.weight_decay,
                log_every=train_cfg.log_every,
            )
            entry = {
                "cfg": cfg.__dict__,
                "override": override,
                "val_acc": val_acc,
                "train_info": info,
                "state_dict": copy.deepcopy(model.state_dict()),
            }
            if best is None or entry["val_acc"] > best["val_acc"]:
                best = entry

        assert best is not None

        # Evaluate best config on test:
        best_cfg = ModelConfig(**{k: v for k, v in best["cfg"].items() if k in ModelConfig.__annotations__})
        model = build_model(model_name, in_dim, num_classes, best_cfg, max_nodes=max_nodes).to(device)
        model.load_state_dict(best["state_dict"])
        test_loss, test_acc = evaluate(model, test_loader, device)

        fold_results.append({
            "fold": fold,
            "val_acc": best["val_acc"],
            "test_acc": test_acc,
            "test_loss": test_loss,
            "best_override": best["override"],
            "best_cfg": best_cfg.__dict__,
            "train_info": best["train_info"],
        })
        print(f"[fold {fold}] val_acc={best['val_acc']:.4f} test_acc={test_acc:.4f}")

    test_accs = np.array([fr["test_acc"] for fr in fold_results], dtype=float)
    return {
        "model": model_name,
        "mean_test_acc": float(test_accs.mean()),
        "std_test_acc": float(test_accs.std(ddof=1)),
        "folds": fold_results,
        "train_cfg": train_cfg.__dict__,
        "base_model_cfg": base_model_cfg.__dict__,
    }


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True,
                   help="TU dataset name, e.g., DD, PROTEINS, NCI1, NCI109, FRANKENSTEIN, MUTAG, REDDIT-BINARY, IMDB-BINARY, IMDB-MULTI, COLLAB")
    p.add_argument("--model", type=str, required=True,
                   help="Model name: TopKPool, SAGPool, ASAP, EdgePool, MinCutPool, DiffPool, Graclus, GMT, Global-Attention, SortPool, Set2Set, GCN, GIN")
    p.add_argument("--root", type=str, default="./data/TUDataset", help="Dataset root directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--val_ratio", type=float, default=0.1, help="Validation fraction inside each training fold")
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--tune", action="store_true", help="Enable light grid search on the validation split (per-fold)")
    p.add_argument("--max_trials", type=int, default=16)

    # quick overrides for common per-method knobs (optional)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--pool_ratio", type=float, default=None)
    p.add_argument("--num_pools", type=int, default=None)
    p.add_argument("--sort_k", type=int, default=None)
    p.add_argument("--gmt_k", type=int, default=None)
    p.add_argument("--gmt_heads", type=int, default=None)
    p.add_argument("--set2set_steps", type=int, default=None)

    # dense pooling safety knobs
    p.add_argument("--dense_max_nodes", type=int, default=None,
                   help="(DiffPool/MinCutPool) Pad (or truncate/drop) graphs to this many nodes for dense adjacency.")
    p.add_argument("--dense_policy", type=str, default=None, choices=["error", "drop", "truncate"],
                   help="What to do if a graph exceeds dense_max_nodes (DiffPool/MinCutPool).")

    p.add_argument("--save_dir", type=str, default="./runs")
    p.add_argument("--no_save_json", action="store_true")
    p.add_argument("--log_every", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model_name = args.model
    if model_name not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown model '{model_name}'. Choices: {sorted(MODEL_DEFAULTS.keys())}")

    train_cfg = TrainConfig(
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        tune=args.tune,
        max_trials=args.max_trials,
        log_every=args.log_every,
        save_dir=args.save_dir,
        save_json=(not args.no_save_json),
    )

    base_cfg = copy.deepcopy(MODEL_DEFAULTS[model_name])
    if args.hidden_dim is not None:
        base_cfg.hidden_dim = int(args.hidden_dim)
    if args.dropout is not None:
        base_cfg.dropout = float(args.dropout)
    if args.pool_ratio is not None:
        base_cfg.pool_ratio = float(args.pool_ratio)
    if args.num_pools is not None:
        base_cfg.num_pools = int(args.num_pools)
    if args.sort_k is not None:
        base_cfg.sort_k = int(args.sort_k)
    if args.gmt_k is not None:
        base_cfg.gmt_k = int(args.gmt_k)
    if args.gmt_heads is not None:
        base_cfg.gmt_heads = int(args.gmt_heads)
    if args.set2set_steps is not None:
        base_cfg.set2set_steps = int(args.set2set_steps)
    if args.dense_max_nodes is not None:
        base_cfg.dense_max_nodes = int(args.dense_max_nodes)
    if args.dense_policy is not None:
        base_cfg.dense_policy = str(args.dense_policy)

    dataset, in_dim, num_classes, max_nodes = load_dataset(
        root=args.root,
        name=args.dataset,
        model_name=model_name,
        cfg=base_cfg,
        seed=args.seed,
    )

    y_all = np.array([int(dataset[i].y.item()) for i in range(len(dataset))], dtype=int)

    print(f"[dataset] {args.dataset} | graphs={len(dataset)} | in_dim={in_dim} | classes={num_classes}")
    if is_dense_model(model_name):
        print(f"[dense] max_nodes={max_nodes} (ToDense padding); policy={base_cfg.dense_policy}")

    results = run_cv(
        dataset=dataset,
        y_all=y_all,
        model_name=model_name,
        in_dim=in_dim,
        num_classes=num_classes,
        max_nodes=max_nodes,
        train_cfg=train_cfg,
        base_model_cfg=base_cfg,
    )

    os.makedirs(train_cfg.save_dir, exist_ok=True)
    out_path = os.path.join(train_cfg.save_dir, f"{args.dataset}__{model_name}__seed{args.seed}.json")
    if train_cfg.save_json:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[saved] {out_path}")

    print(f"\n[summary] {args.dataset} / {model_name} | mean_test_acc={results['mean_test_acc']:.4f} ± {results['std_test_acc']:.4f}")


if __name__ == "__main__":
    main()
