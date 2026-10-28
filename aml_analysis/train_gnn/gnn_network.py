import torch
from torch.nn import Linear, LayerNorm, Dropout
from torch_geometric.nn import PNAConv, GCNConv, SAGEConv, GATv2Conv, TransformerConv
import torch.nn.functional as F
from torch_geometric.utils import degree


class GPNA(torch.nn.Module):
    """Residual GPNA with a direct node-feature classifier path."""

    def __init__(self, config, dataset, training=False):
        super().__init__()
        hidden = int(config.hidden_dim)
        layers = int(getattr(config, 'gnn_layers', 2))
        if layers not in (1, 2, 3):
            raise ValueError("The residual GNN layer count must be 1, 2 or 3")
        self.dropout = float(config.dropout)
        self.feature_dropout = Dropout(float(getattr(config, "feature_dropout", 0.2)))
        self.embedding_dropout = Dropout(float(getattr(config, "embedding_dropout", 0.2)))

        # Bidirectional edges already provide each node's incoming degree.
        src, dst = dataset.edge_index
        train_mask = getattr(dataset, "train_mask", None)
        if train_mask is not None:
            train_mask = train_mask.bool()
        if train_mask is not None and train_mask.any():
            mask = train_mask[src] & train_mask[dst]
            deg = degree(dst[mask], num_nodes=dataset.num_nodes, dtype=torch.long)[train_mask]
        else:
            deg = degree(dst, num_nodes=dataset.num_nodes, dtype=torch.long)
        histogram = torch.bincount(deg, minlength=1)

        torch.manual_seed(config.SEED)

        self.encoder = torch.nn.Sequential(
            Linear(config.gene_dim, hidden), LayerNorm(hidden), torch.nn.GELU(),
        )
        self.convs = torch.nn.ModuleList([
            PNAConv(
                hidden, hidden, ["mean", "min", "max", "std"],
                ["identity", "amplification", "attenuation"], histogram,
            ) for _ in range(layers)
        ])
        self.norms = torch.nn.ModuleList([LayerNorm(hidden) for _ in range(layers)])
        # Original features bypass message passing, preserving rare-node signals.
        self.classifier = torch.nn.Sequential(
            Linear(config.gene_dim + hidden * (layers + 1), hidden * 2),
            LayerNorm(hidden * 2), torch.nn.GELU(), Dropout(self.dropout),
            Linear(hidden * 2, config.num_classes),
        )
        self.train(training)

    def forward(self, x, edge_index):
        features = self.feature_dropout(x)
        h = self.encoder(features)
        states = [h]
        for conv, norm in zip(self.convs, self.norms):
            update = F.dropout(F.gelu(conv(h, edge_index)), p=self.dropout, training=self.training)
            h = norm(h + update)
            states.append(h)
        embedding = torch.cat([features] + states, dim=1)
        return self.classifier(self.embedding_dropout(embedding)), h, embedding



class GCN(torch.nn.Module):
    """Residual GCN with a direct node-feature classifier path."""

    def __init__(self, config, training=False):
        super().__init__()
        hidden = int(config.hidden_dim)
        layers = int(getattr(config, 'gnn_layers', 2))
        if layers not in (1, 2, 3):
            raise ValueError("The residual GNN layer count must be 1, 2 or 3")
        self.dropout = float(config.dropout)
        self.feature_dropout = Dropout(float(getattr(config, "feature_dropout", 0.2)))
        self.embedding_dropout = Dropout(float(getattr(config, "embedding_dropout", 0.2)))

        torch.manual_seed(config.SEED)

        self.encoder = torch.nn.Sequential(
            Linear(config.gene_dim, hidden), LayerNorm(hidden), torch.nn.GELU(),
        )
        self.convs = torch.nn.ModuleList([
            GCNConv(hidden, hidden) for _ in range(layers)
        ])
        self.norms = torch.nn.ModuleList([LayerNorm(hidden) for _ in range(layers)])
        # Original features bypass message passing, preserving rare-node signals.
        self.classifier = torch.nn.Sequential(
            Linear(config.gene_dim + hidden * (layers + 1), hidden * 2),
            LayerNorm(hidden * 2), torch.nn.GELU(), Dropout(self.dropout),
            Linear(hidden * 2, config.num_classes),
        )
        self.train(training)

    def forward(self, x, edge_index):
        features = self.feature_dropout(x)
        h = self.encoder(features)
        states = [h]
        for conv, norm in zip(self.convs, self.norms):
            update = F.dropout(F.gelu(conv(h, edge_index)), p=self.dropout, training=self.training)
            h = norm(h + update)
            states.append(h)
        embedding = torch.cat([features] + states, dim=1)
        return self.classifier(self.embedding_dropout(embedding)), h, embedding



class GraphSAGE(torch.nn.Module):
    """Residual GraphSAGE with a direct node-feature classifier path."""

    def __init__(self, config, training=False):
        super().__init__()
        hidden = int(config.hidden_dim)
        layers = int(getattr(config, 'gnn_layers', 2))
        if layers not in (1, 2, 3):
            raise ValueError("The residual GNN layer count must be 1, 2 or 3")
        self.dropout = float(config.dropout)
        self.feature_dropout = Dropout(float(getattr(config, "feature_dropout", 0.2)))
        self.embedding_dropout = Dropout(float(getattr(config, "embedding_dropout", 0.2)))

        torch.manual_seed(config.SEED)

        self.encoder = torch.nn.Sequential(
            Linear(config.gene_dim, hidden), LayerNorm(hidden), torch.nn.GELU(),
        )
        self.convs = torch.nn.ModuleList([
            SAGEConv(hidden, hidden) for _ in range(layers)
        ])
        self.norms = torch.nn.ModuleList([LayerNorm(hidden) for _ in range(layers)])
        # Original features bypass message passing, preserving rare-node signals.
        self.classifier = torch.nn.Sequential(
            Linear(config.gene_dim + hidden * (layers + 1), hidden * 2),
            LayerNorm(hidden * 2), torch.nn.GELU(), Dropout(self.dropout),
            Linear(hidden * 2, config.num_classes),
        )
        self.train(training)

    def forward(self, x, edge_index):
        features = self.feature_dropout(x)
        h = self.encoder(features)
        states = [h]
        for conv, norm in zip(self.convs, self.norms):
            update = F.dropout(F.gelu(conv(h, edge_index)), p=self.dropout, training=self.training)
            h = norm(h + update)
            states.append(h)
        embedding = torch.cat([features] + states, dim=1)
        return self.classifier(self.embedding_dropout(embedding)), h, embedding



class GATv2(torch.nn.Module):
    """Residual GATv2 with a direct node-feature classifier path."""

    def __init__(self, config, training=False):
        super().__init__()
        hidden = int(config.hidden_dim)
        layers = int(getattr(config, 'gnn_layers', 2))
        if layers not in (1, 2, 3):
            raise ValueError("The residual GNN layer count must be 1, 2 or 3")
        self.dropout = float(config.dropout)
        self.feature_dropout = Dropout(float(getattr(config, "feature_dropout", 0.2)))
        self.embedding_dropout = Dropout(float(getattr(config, "embedding_dropout", 0.2)))
        heads = int(getattr(config, "heads", 4))
        if heads < 1 or hidden % heads:
            raise ValueError("hidden_dim must be divisible by a positive heads value")

        torch.manual_seed(config.SEED)

        self.encoder = torch.nn.Sequential(
            Linear(config.gene_dim, hidden), LayerNorm(hidden), torch.nn.GELU(),
        )
        self.convs = torch.nn.ModuleList([
            GATv2Conv(hidden, hidden // heads, heads=heads, dropout=self.dropout) for _ in range(layers)
        ])
        self.norms = torch.nn.ModuleList([LayerNorm(hidden) for _ in range(layers)])
        # Original features bypass message passing, preserving rare-node signals.
        self.classifier = torch.nn.Sequential(
            Linear(config.gene_dim + hidden * (layers + 1), hidden * 2),
            LayerNorm(hidden * 2), torch.nn.GELU(), Dropout(self.dropout),
            Linear(hidden * 2, config.num_classes),
        )
        self.train(training)

    def forward(self, x, edge_index):
        features = self.feature_dropout(x)
        h = self.encoder(features)
        states = [h]
        for conv, norm in zip(self.convs, self.norms):
            update = F.dropout(F.gelu(conv(h, edge_index)), p=self.dropout, training=self.training)
            h = norm(h + update)
            states.append(h)
        embedding = torch.cat([features] + states, dim=1)
        return self.classifier(self.embedding_dropout(embedding)), h, embedding



class GraphTransformer(torch.nn.Module):
    """Residual GraphTransformer with a direct node-feature classifier path."""

    def __init__(self, config, training=False):
        super().__init__()
        hidden = int(config.hidden_dim)
        layers = int(getattr(config, 'gnn_layers', 2))
        if layers not in (1, 2, 3):
            raise ValueError("The residual GNN layer count must be 1, 2 or 3")
        self.dropout = float(config.dropout)
        self.feature_dropout = Dropout(float(getattr(config, "feature_dropout", 0.2)))
        self.embedding_dropout = Dropout(float(getattr(config, "embedding_dropout", 0.2)))
        heads = int(getattr(config, "heads", 4))
        if heads < 1 or hidden % heads:
            raise ValueError("hidden_dim must be divisible by a positive heads value")

        torch.manual_seed(config.SEED)

        self.encoder = torch.nn.Sequential(
            Linear(config.gene_dim, hidden), LayerNorm(hidden), torch.nn.GELU(),
        )
        self.convs = torch.nn.ModuleList([
            TransformerConv(hidden, hidden // heads, heads=heads, dropout=self.dropout) for _ in range(layers)
        ])
        self.norms = torch.nn.ModuleList([LayerNorm(hidden) for _ in range(layers)])
        # Original features bypass message passing, preserving rare-node signals.
        self.classifier = torch.nn.Sequential(
            Linear(config.gene_dim + hidden * (layers + 1), hidden * 2),
            LayerNorm(hidden * 2), torch.nn.GELU(), Dropout(self.dropout),
            Linear(hidden * 2, config.num_classes),
        )
        self.train(training)

    def forward(self, x, edge_index):
        features = self.feature_dropout(x)
        h = self.encoder(features)
        states = [h]
        for conv, norm in zip(self.convs, self.norms):
            update = F.dropout(F.gelu(conv(h, edge_index)), p=self.dropout, training=self.training)
            h = norm(h + update)
            states.append(h)
        embedding = torch.cat([features] + states, dim=1)
        return self.classifier(self.embedding_dropout(embedding)), h, embedding



class ResidualPNA(GPNA):
    """Compatibility name for the residual PNA architecture."""
