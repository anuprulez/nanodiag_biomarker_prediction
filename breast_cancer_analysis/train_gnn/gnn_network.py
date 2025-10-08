import torch
from torch.nn import Linear, BatchNorm1d, LayerNorm
from torch_geometric.nn import (
    PNAConv,
    GCNConv,
    SAGEConv,
    GATv2Conv,
    TransformerConv
)
import torch.nn.functional as F
from torch_geometric.utils import degree, subgraph
from torch_geometric.nn.norm import GraphNorm


class GPNA(torch.nn.Module):

    def pna_deg_from_train_nodes(self, data, undirected=True, device=None):
        if device is None:
            device = data.edge_index.device

        # Induce subgraph containing only training nodes
        train_n = data.train_mask.nonzero(as_tuple=False).view(-1)
        # obtain subgraph edges
        e_sub, _ = subgraph(train_n, data.edge_index,
                            relabel_nodes=True,
                            num_nodes=data.num_nodes)

        # If undirected (and edges are stored as one direction), count both ends:
        dst = torch.cat([e_sub[0], e_sub[1]], dim=0)

        d = degree(dst, num_nodes=train_n.size(0), dtype=torch.long)
        d = d.to(device)
        max_deg = int(d.max().item()) if d.numel() > 0 else 0
        deg_hist = torch.zeros(max_deg + 1, dtype=torch.long, device=device)
        deg_hist[:d.numel()] += torch.bincount(d, minlength=deg_hist.numel())

        return deg_hist

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
        num_classes = config.num_classes
        gene_dim = config.gene_dim
        hidden_dim = config.hidden_dim
        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        p_drop = config.dropout
        #deg = self.__find_deg(train_dataset)
        deg = self.pna_deg_from_train_nodes(train_dataset)
        torch.manual_seed(config.SEED)
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
        h1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        h1 = F.dropout(h1, p=self.p_drop, training=self.training)

        h2 = F.elu(self.bn2(self.conv2(h1, edge_index)))
        h2 = F.dropout(h2, p=self.p_drop, training=self.training)

        h3 = F.elu(self.bn3(self.conv3(h2, edge_index)))
        h3 = F.dropout(h3, p=self.p_drop, training=self.training)

        h4_in = h3 + h1

        out_pnaconv4 = self.conv4(h4_in, edge_index)
        out_batch_norm4 = self.bn4(out_pnaconv4)
        h4 = F.elu(out_batch_norm4)
        h4 = F.dropout(h4, p=self.p_drop, training=self.training)
        return self.classifier(h4), out_pnaconv4, out_batch_norm4


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
        h1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        h1 = F.dropout(h1, p=self.p_drop, training=self.training)

        h2 = F.elu(self.bn2(self.conv2(h1, edge_index)))
        h2 = F.dropout(h2, p=self.p_drop, training=self.training)

        h3 = F.elu(self.bn3(self.conv3(h2, edge_index)))
        h3 = F.dropout(h3, p=self.p_drop, training=self.training)

        h4_in = h3 + h1

        out_pnaconv4 = self.conv4(h4_in, edge_index)
        out_batch_norm4 = self.bn4(out_pnaconv4)
        h4 = F.elu(out_batch_norm4)
        h4 = F.dropout(h4, p=self.p_drop, training=self.training)
        return self.classifier(h4), out_pnaconv4, out_batch_norm4


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
        h1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        h1 = F.dropout(h1, p=self.p_drop, training=self.training)

        h2 = F.elu(self.bn2(self.conv2(h1, edge_index)))
        h2 = F.dropout(h2, p=self.p_drop, training=self.training)

        h3 = F.elu(self.bn3(self.conv3(h2, edge_index)))
        h3 = F.dropout(h3, p=self.p_drop, training=self.training)

        h4_in = h3 + h1

        out_pnaconv4 = self.conv4(h4_in, edge_index)
        out_batch_norm4 = self.bn4(out_pnaconv4)
        h4 = F.elu(out_batch_norm4)
        h4 = F.dropout(h4, p=self.p_drop, training=self.training)
        return self.classifier(h4), out_pnaconv4, out_batch_norm4


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
        h1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        h1 = F.dropout(h1, p=self.p_drop, training=self.training)

        h2 = F.elu(self.bn2(self.conv2(h1, edge_index)))
        h2 = F.dropout(h2, p=self.p_drop, training=self.training)

        h3 = F.elu(self.bn3(self.conv3(h2, edge_index)))
        h3 = F.dropout(h3, p=self.p_drop, training=self.training)

        h4_in = h3 + h1

        out_pnaconv4 = self.conv4(h4_in, edge_index)
        out_batch_norm4 = self.bn4(out_pnaconv4)
        h4 = F.elu(out_batch_norm4)
        h4 = F.dropout(h4, p=self.p_drop, training=self.training)
        return self.classifier(h4), out_pnaconv4, out_batch_norm4
    

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
        h1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        h1 = F.dropout(h1, p=self.p_drop, training=self.training)

        h2 = F.elu(self.bn2(self.conv2(h1, edge_index)))
        h2 = F.dropout(h2, p=self.p_drop, training=self.training)

        h3 = F.elu(self.bn3(self.conv3(h2, edge_index)))
        h3 = F.dropout(h3, p=self.p_drop, training=self.training)

        h4_in = h3 + h1

        out_pnaconv4 = self.conv4(h4_in, edge_index)
        out_batch_norm4 = self.bn4(out_pnaconv4)
        h4 = F.elu(out_batch_norm4)
        h4 = F.dropout(h4, p=self.p_drop, training=self.training)
        return self.classifier(h4), out_pnaconv4, out_batch_norm4