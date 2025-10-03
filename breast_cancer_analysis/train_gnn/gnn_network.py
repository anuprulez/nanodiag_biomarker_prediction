import torch
from torch_geometric.data import Data
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, PNAConv, TransformerConv, APPNP, GCN2Conv, GINConv, GATv2Conv, SAGEConv
import torch.nn.functional as F
from torch_geometric.utils import degree

import pandas as pd
import numpy as np


class GPNA(torch.nn.Module):
    
    def __find_deg(self, train_dataset):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        max_degree = -1
        d = degree(train_dataset.edge_index[1], num_nodes=train_dataset.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))
        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        deg = deg.to(device)
        d = degree(train_dataset.edge_index[1], num_nodes=train_dataset.num_nodes, dtype=torch.long)
        d = d.to(device)
        deg += torch.bincount(d, minlength=deg.numel())
        return deg

    def __init__(self, config, train_dataset):
        super().__init__()
        num_classes = config["num_classes"]
        gene_dim = config["gene_dim"]
        hidden_dim = config["hidden_dim"]
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        deg = self.__find_deg(train_dataset)
        SEED = config["SEED"]
        torch.manual_seed(SEED)
        self.pnaconv1 = PNAConv(gene_dim, hidden_dim, aggregators, scalers, deg)
        self.pnaconv2 = PNAConv(hidden_dim, 2 * hidden_dim, aggregators, scalers, deg)
        self.pnaconv3 = PNAConv(2 * hidden_dim, hidden_dim, aggregators, scalers, deg)
        self.pnaconv4 = PNAConv(hidden_dim, hidden_dim // 2, aggregators, scalers, deg)
        self.classifier = Linear(hidden_dim // 2, num_classes)
        self.batch_norm1 = BatchNorm1d(hidden_dim)
        self.batch_norm2 = BatchNorm1d(2 * hidden_dim)
        self.batch_norm3 = BatchNorm1d(hidden_dim)
        self.batch_norm4 = BatchNorm1d(hidden_dim // 2)
        self.model_activation = dict()

    def forward(self, x, edge_index):
        h = self.pnaconv1(x, edge_index)
        h = self.batch_norm1(F.relu(h))
        h = self.pnaconv2(h, edge_index)
        h = self.batch_norm2(F.relu(h))
        h = self.pnaconv3(h, edge_index)
        h = self.batch_norm3(F.relu(h))
        out_pnaconv4 = self.pnaconv4(h, edge_index)
        out_batch_norm4 = self.batch_norm4(F.relu(out_pnaconv4))
        out = self.classifier(out_batch_norm4)
        return out, out_pnaconv4, out_batch_norm4


class GCN(torch.nn.Module):
    '''
    Neural network with graph convolution network (GCN)
    '''
    def __init__(self, config):
        super().__init__()
        num_classes = config["num_classes"]
        gene_dim = config["gene_dim"]
        hidden_dim = config["hidden_dim"]
        SEED = config["SEED"]
        torch.manual_seed(SEED)
        self.conv1 = GCNConv(gene_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2 * hidden_dim)
        self.conv3 = GCNConv(2 * hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim // 2)
        self.classifier = Linear(hidden_dim // 2, num_classes)
        self.batch_norm1 = BatchNorm1d(hidden_dim)
        self.batch_norm2 = BatchNorm1d(2 * hidden_dim)
        self.batch_norm3 = BatchNorm1d(hidden_dim)
        self.batch_norm4 = BatchNorm1d(hidden_dim // 2)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.batch_norm1(F.relu(h))
        h = self.conv2(h, edge_index)
        h = self.batch_norm2(F.relu(h))
        h = self.conv3(h, edge_index)
        h = self.batch_norm3(F.relu(h))
        h = self.conv4(h, edge_index)
        h = self.batch_norm4(F.relu(h))
        out = self.classifier(h)
        return out


class GraphSAGE(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        num_classes = config["num_classes"]
        gene_dim    = config["gene_dim"]
        hidden_dim  = config["hidden_dim"]
        p_drop      = config.get("dropout", 0.2)
        torch.manual_seed(config["SEED"])

        self.conv1 = SAGEConv(gene_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, 2 * hidden_dim)
        self.conv3 = SAGEConv(2 * hidden_dim, hidden_dim)
        self.conv4 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.bn1   = BatchNorm1d(hidden_dim)
        self.bn2   = BatchNorm1d(2 * hidden_dim)
        self.bn3   = BatchNorm1d(hidden_dim)
        self.bn4   = BatchNorm1d(hidden_dim // 2)
        self.cls   = Linear(hidden_dim // 2, num_classes)
        self.p_drop = p_drop

    def forward(self, x, edge_index):
        h = F.relu(self.bn1(self.conv1(x, edge_index))); h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index))); h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.bn3(self.conv3(h, edge_index))); h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.bn4(self.conv4(h, edge_index))); h = F.dropout(h, p=self.p_drop, training=self.training)
        return self.cls(h)


def _mlp(in_dim, out_dim):
    return Sequential(Linear(in_dim, out_dim), ReLU(), Linear(out_dim, out_dim))


class GIN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        num_classes = config["num_classes"]
        gene_dim    = config["gene_dim"]
        hidden_dim  = config["hidden_dim"]
        p_drop      = config.get("dropout", 0.2)
        torch.manual_seed(config["SEED"])

        self.conv1 = GINConv(_mlp(gene_dim, hidden_dim))
        self.conv2 = GINConv(_mlp(hidden_dim, 2 * hidden_dim))
        self.conv3 = GINConv(_mlp(2 * hidden_dim, hidden_dim))
        self.conv4 = GINConv(_mlp(hidden_dim, hidden_dim // 2))
        self.bn1   = BatchNorm1d(hidden_dim)
        self.bn2   = BatchNorm1d(2 * hidden_dim)
        self.bn3   = BatchNorm1d(hidden_dim)
        self.bn4   = BatchNorm1d(hidden_dim // 2)
        self.cls   = Linear(hidden_dim // 2, num_classes)
        self.p_drop = p_drop

    def forward(self, x, edge_index):
        h = F.relu(self.bn1(self.conv1(x, edge_index))); h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index))); h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.bn3(self.conv3(h, edge_index))); h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.bn4(self.conv4(h, edge_index))); h = F.dropout(h, p=self.p_drop, training=self.training)
        return self.cls(h)


class GATv2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        num_classes = config["num_classes"]
        gene_dim    = config["gene_dim"]
        hidden_dim  = config["hidden_dim"]
        heads       = config.get("heads", 4)
        p_drop      = config.get("dropout", 0.3)
        torch.manual_seed(config["SEED"])

        self.conv1 = GATv2Conv(gene_dim, hidden_dim // heads, heads=heads, dropout=p_drop)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=p_drop)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=p_drop)
        self.conv4 = GATv2Conv(hidden_dim, (hidden_dim // 2) // heads, heads=heads, dropout=p_drop, concat=True)
        self.bn1   = BatchNorm1d(hidden_dim)
        self.bn2   = BatchNorm1d(hidden_dim)
        self.bn3   = BatchNorm1d(hidden_dim)
        self.bn4   = BatchNorm1d(hidden_dim // 2)
        self.cls   = Linear(hidden_dim // 2, num_classes)
        self.p_drop = p_drop

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index)); h = self.bn1(h); h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.elu(self.conv2(h, edge_index)); h = self.bn2(h); h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.elu(self.conv3(h, edge_index)); h = self.bn3(h); h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.elu(self.conv4(h, edge_index)); h = self.bn4(h); h = F.dropout(h, p=self.p_drop, training=self.training)
        return self.cls(h)


class GCNII(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        num_classes = config["num_classes"]
        gene_dim    = config["gene_dim"]
        hidden_dim  = config["hidden_dim"]
        num_layers  = config.get("num_layers", 8)
        alpha       = config.get("alpha", 0.1)   # initial residual weight
        theta       = config.get("theta", 0.5)   # identity mapping strength
        p_drop      = config.get("dropout", 0.5)
        torch.manual_seed(config["SEED"])

        self.lin_in  = Linear(gene_dim, hidden_dim)
        self.convs   = torch.nn.ModuleList([GCN2Conv(hidden_dim, alpha=alpha, theta=theta, layer=i+1) 
                                            for i in range(num_layers)])
        self.bn      = torch.nn.ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.lin_out = Linear(hidden_dim, num_classes)
        self.p_drop  = p_drop

    def forward(self, x, edge_index):
        x0 = F.dropout(F.relu(self.lin_in(x)), p=self.p_drop, training=self.training)
        h  = x0
        for i, conv in enumerate(self.convs):
            h = F.dropout(h, p=self.p_drop, training=self.training)
            h = conv(h, x0, edge_index)  # uses initial features x0
            h = self.bn[i](F.relu(h))
        h = F.dropout(h, p=self.p_drop, training=self.training)
        return self.lin_out(h)


class APPNPNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        num_classes = config["num_classes"]
        gene_dim    = config["gene_dim"]
        hidden_dim  = config["hidden_dim"]
        K           = config.get("K", 10)         # propagation steps
        alpha       = config.get("alpha", 0.1)    # teleport proba
        p_drop      = config.get("dropout", 0.5)
        torch.manual_seed(config["SEED"])

        self.lin1 = Linear(gene_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, hidden_dim // 2)
        self.lin3 = Linear(hidden_dim // 2, num_classes)
        self.bn1  = BatchNorm1d(hidden_dim)
        self.bn2  = BatchNorm1d(hidden_dim // 2)
        self.prop = APPNP(K=K, alpha=alpha, dropout=p_drop)
        self.p_drop = p_drop

    def forward(self, x, edge_index):
        h0 = F.relu(self.bn1(self.lin1(x))); h0 = F.dropout(h0, p=self.p_drop, training=self.training)
        h0 = F.relu(self.bn2(self.lin2(h0))); h0 = F.dropout(h0, p=self.p_drop, training=self.training)
        logits = self.lin3(h0)
        return self.prop(logits, edge_index)


class GraphTransformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        num_classes = config["num_classes"]
        gene_dim    = config["gene_dim"]
        hidden_dim  = config["hidden_dim"]
        heads       = config.get("heads", 4)
        p_drop      = config.get("dropout", 0.2)
        torch.manual_seed(config["SEED"])

        self.conv1 = TransformerConv(gene_dim, hidden_dim // heads, heads=heads, dropout=p_drop)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=p_drop)
        self.conv3 = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=p_drop)
        self.conv4 = TransformerConv(hidden_dim, (hidden_dim // 2) // heads, heads=heads, dropout=p_drop)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)
        self.bn4 = BatchNorm1d(hidden_dim // 2)
        self.cls = Linear(hidden_dim // 2, num_classes)
        self.p_drop = p_drop

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index)); h = self.bn1(h); h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.conv2(h, edge_index)); h = self.bn2(h); h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.conv3(h, edge_index)); h = self.bn3(h); h = F.dropout(h, p=self.p_drop, training=self.training)
        h = F.relu(self.conv4(h, edge_index)); h = self.bn4(h); h = F.dropout(h, p=self.p_drop, training=self.training)
        return self.cls(h)

