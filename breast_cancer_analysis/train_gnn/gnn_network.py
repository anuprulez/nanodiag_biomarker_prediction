import torch
from torch_geometric.data import Data
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import (
    GCNConv,
    PNAConv,
    TransformerConv,
    APPNP,
    GCN2Conv,
    GINConv,
    GATv2Conv,
    SAGEConv,
)
import torch.nn.functional as F
from torch_geometric.utils import degree

import pandas as pd
import numpy as np


class GPNA(torch.nn.Module):
    def __find_deg(self, train_dataset):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_degree = -1
        d = degree(
            train_dataset.edge_index[1],
            num_nodes=train_dataset.num_nodes,
            dtype=torch.long,
        )
        max_degree = max(max_degree, int(d.max()))
        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        deg = deg.to(device)
        d = degree(
            train_dataset.edge_index[1],
            num_nodes=train_dataset.num_nodes,
            dtype=torch.long,
        )
        d = d.to(device)
        deg += torch.bincount(d, minlength=deg.numel())
        return deg

    def __init__(self, config, train_dataset):
        super().__init__()
        num_classes = config["num_classes"]
        gene_dim = config["gene_dim"]
        hidden_dim = config["hidden_dim"]
        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        p_drop = config.get("dropout", 0.2)
        deg = self.__find_deg(train_dataset)
        SEED = config["SEED"]
        torch.manual_seed(SEED)
        self.conv1 = PNAConv(gene_dim, hidden_dim, aggregators, scalers, deg)
        self.conv2 = PNAConv(hidden_dim, 2 * hidden_dim, aggregators, scalers, deg)
        self.conv3 = PNAConv(2 * hidden_dim, hidden_dim, aggregators, scalers, deg)
        self.conv4 = PNAConv(hidden_dim, hidden_dim // 2, aggregators, scalers, deg)
        self.classifier = Linear(hidden_dim // 2, num_classes)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(2 * hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)
        self.bn4 = BatchNorm1d(hidden_dim // 2)
        self.p_drop = p_drop

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.bn1(h)
        h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = self.bn2(h)
        h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.conv3(h, edge_index))
        h = self.bn3(h)
        h = F.dropout(h, p=self.p_drop, training=self.training)
        out_pnaconv4 = F.relu(self.conv4(h, edge_index))
        out_batch_norm4 = self.bn4(out_pnaconv4)
        out_batch_norm4 = F.dropout(out_batch_norm4, p=self.p_drop, training=self.training)
        return self.classifier(out_batch_norm4), out_pnaconv4, out_batch_norm4


class GCN(torch.nn.Module):
    """
    Neural network with graph convolution network (GCN)
    """
    def __init__(self, config):
        super().__init__()
        num_classes = config.num_classes
        gene_dim = config.gene_dim
        hidden_dim = config.hidden_dim
        SEED = config.SEED
        p_drop = config["dropout"]
        torch.manual_seed(SEED)
        self.conv1 = GCNConv(gene_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2 * hidden_dim)
        self.conv3 = GCNConv(2 * hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim // 2)
        self.classifier = Linear(hidden_dim // 2, num_classes)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(2 * hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)
        self.bn4 = BatchNorm1d(hidden_dim // 2)
        self.p_drop = p_drop

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.bn1(h)
        h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = self.bn2(h)
        h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.conv3(h, edge_index))
        h = self.bn3(h)
        h = F.dropout(h, p=self.p_drop, training=self.training)
        out_pnaconv4 = F.relu(self.conv4(h, edge_index))
        out_batch_norm4 = self.bn4(out_pnaconv4)
        out_batch_norm4 = F.dropout(out_batch_norm4, p=self.p_drop, training=self.training)
        return self.classifier(out_batch_norm4), out_pnaconv4, out_batch_norm4


class GraphSAGE(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        num_classes = config["num_classes"]
        gene_dim = config["gene_dim"]
        hidden_dim = config["hidden_dim"]
        p_drop = config["dropout"]
        torch.manual_seed(config["SEED"])

        self.conv1 = SAGEConv(gene_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, 2 * hidden_dim)
        self.conv3 = SAGEConv(2 * hidden_dim, hidden_dim)
        self.conv4 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(2 * hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)
        self.bn4 = BatchNorm1d(hidden_dim // 2)
        self.classifier = Linear(hidden_dim // 2, num_classes)
        self.p_drop = p_drop

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.bn1(h)
        h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = self.bn2(h)
        h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.conv3(h, edge_index))
        h = self.bn3(h)
        h = F.dropout(h, p=self.p_drop, training=self.training)
        out_pnaconv4 = F.relu(self.conv4(h, edge_index))
        out_batch_norm4 = self.bn4(out_pnaconv4)
        out_batch_norm4 = F.dropout(out_batch_norm4, p=self.p_drop, training=self.training)
        return self.classifier(out_batch_norm4), out_pnaconv4, out_batch_norm4


class GATv2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        num_classes = config["num_classes"]
        gene_dim = config["gene_dim"]
        hidden_dim = config["hidden_dim"]
        heads = config["heads"]
        p_drop = config["dropout"]
        torch.manual_seed(config["SEED"])

        self.conv1 = GATv2Conv(
            gene_dim, hidden_dim // heads, heads=heads, dropout=p_drop
        )
        self.conv2 = GATv2Conv(
            hidden_dim, hidden_dim // heads, heads=heads, dropout=p_drop
        )
        self.conv3 = GATv2Conv(
            hidden_dim, hidden_dim // heads, heads=heads, dropout=p_drop
        )
        self.conv4 = GATv2Conv(
            hidden_dim,
            (hidden_dim // 2) // heads,
            heads=heads,
            dropout=p_drop,
            concat=True,
        )
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)
        self.bn4 = BatchNorm1d(hidden_dim // 2)
        self.classifier = Linear(hidden_dim // 2, num_classes)
        self.p_drop = p_drop

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        h = self.bn1(h)
        h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.elu(self.conv2(h, edge_index))
        h = self.bn2(h)
        h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.elu(self.conv3(h, edge_index))
        h = self.bn3(h)
        h = F.dropout(h, p=self.p_drop, training=self.training)
        out_pnaconv4 = F.elu(self.conv4(h, edge_index))
        out_batch_norm4 = self.bn4(out_pnaconv4)
        out_batch_norm4 = F.dropout(out_batch_norm4, p=self.p_drop, training=self.training)
        return self.classifier(out_batch_norm4), out_pnaconv4, out_batch_norm4
    

class GraphTransformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        num_classes = config["num_classes"]
        gene_dim = config["gene_dim"]
        hidden_dim = config["hidden_dim"]
        heads = config["heads"]
        p_drop = config["dropout"]
        torch.manual_seed(config["SEED"])

        self.conv1 = TransformerConv(
            gene_dim, hidden_dim // heads, heads=heads, dropout=p_drop
        )
        self.conv2 = TransformerConv(
            hidden_dim, hidden_dim // heads, heads=heads, dropout=p_drop
        )
        self.conv3 = TransformerConv(
            hidden_dim, hidden_dim // heads, heads=heads, dropout=p_drop
        )
        self.conv4 = TransformerConv(
            hidden_dim, (hidden_dim // 2) // heads, heads=heads, dropout=p_drop
        )
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)
        self.bn4 = BatchNorm1d(hidden_dim // 2)
        self.classifier = Linear(hidden_dim // 2, num_classes)
        self.p_drop = p_drop

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.bn1(h)
        h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = self.bn2(h)
        h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.conv3(h, edge_index))
        h = self.bn3(h)
        h = F.dropout(h, p=self.p_drop, training=self.training)
        h_pnaconv4 = F.relu(self.conv4(h, edge_index))
        h_batch_norm4 = self.bn4(h_pnaconv4)
        h_batch_norm4 = F.dropout(h_batch_norm4, p=self.p_drop, training=self.training)
        return self.classifier(h_batch_norm4), h_pnaconv4, h_batch_norm4
