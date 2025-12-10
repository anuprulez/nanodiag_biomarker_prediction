import torch
from torch.nn import Linear, BatchNorm1d, LayerNorm, Dropout
from torch_geometric.nn import (
    PNAConv,
    GCNConv,
    SAGEConv,
    GATv2Conv,
    TransformerConv,
)
import torch.nn.functional as F
from torch_geometric.utils import degree, subgraph
from torch_geometric.nn.norm import GraphNorm


class GPNA(torch.nn.Module):

    def __find_deg_train_nodes(self, data, undirected=True, device=None):
        # take only training nodes when mask is provided, otherwise fall back to full graph
        train_mask = getattr(data, "train_mask", None)
        if train_mask is None:
            return None
        if train_mask.dtype != torch.bool:
            train_mask = train_mask.bool()
        train_nodes = train_mask.nonzero(as_tuple=False).view(-1)
        if train_nodes.numel() == 0:
            return None

        if device is None:
            device = data.edge_index.device

        # induce subgraph on training nodes; relabel to 0..n-1
        e_sub, _ = subgraph(
            train_nodes,
            data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes
        )

        n = int(train_nodes.numel())

        # choose which edge endpoints to count
        if undirected:
            dst = torch.cat([e_sub[0], e_sub[1]], dim=0)
        else:
            dst = e_sub[1]

        # per-node degrees (length = n)
        d = degree(dst, num_nodes=n, dtype=torch.long)

        # histogram over degree values (length = max(d)+1)
        max_deg = int(d.max().item()) if d.numel() > 0 else 0
        deg_hist = torch.bincount(d, minlength=max_deg + 1)

        # move to model device
        deg_hist = deg_hist.to(device)

        return deg_hist

    def __find_deg(self, dataset):
        # take entire dataset
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_degree = -1
        d = degree(
            dataset.edge_index[1],
            num_nodes=dataset.num_nodes,
            dtype=torch.long,
        )
        max_degree = max(max_degree, int(d.max()))
        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        deg = deg.to(device)
        d = degree(
            dataset.edge_index[1],
            num_nodes=dataset.num_nodes,
            dtype=torch.long,
        )
        d = d.to(device)
        deg += torch.bincount(d, minlength=deg.numel())
        return deg

    def __init__(self, config, dataset, training=False):
        super().__init__()
        num_classes = config.num_classes
        gene_dim = config.gene_dim
        hidden_dim = config.hidden_dim
        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"] # TODO: test these "linear", "inverse_linear".
        deg = self.__find_deg_train_nodes(dataset)
        if deg is None:
            deg = self.__find_deg(dataset)

        torch.manual_seed(config.SEED)
        self.conv1 = PNAConv(gene_dim, hidden_dim, aggregators, scalers, deg)
        self.conv2 = PNAConv(hidden_dim, 2 * hidden_dim, aggregators, scalers, deg)
        self.conv3 = PNAConv(2 * hidden_dim, hidden_dim, aggregators, scalers, deg)
        self.conv4 = PNAConv(hidden_dim, hidden_dim // 2, aggregators, scalers, deg)
        self.classifier = Linear(hidden_dim // 2, num_classes)
        self.bn1 = LayerNorm(hidden_dim)
        self.bn2 = LayerNorm(2 * hidden_dim)
        self.bn3 = LayerNorm(hidden_dim)
        self.bn4 = LayerNorm(hidden_dim // 2)
        self.dropout1 = Dropout(p=config.dropout)
        self.dropout2 = Dropout(p=config.dropout)
        self.dropout3 = Dropout(p=config.dropout)
        self.dropout4 = Dropout(p=config.dropout)
        self.training = training

    def forward(self, x, edge_index):
        h1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        h1 = self.dropout1(h1)

        h2 = F.elu(self.bn2(self.conv2(h1, edge_index)))
        h2 = self.dropout2(h2)

        h3 = F.elu(self.bn3(self.conv3(h2, edge_index)))
        h3 = self.dropout3(h3)

        h4_in = h3 + h1

        out_pnaconv = self.conv4(h4_in, edge_index)
        out_batch_norm = self.bn4(out_pnaconv)
        h = F.elu(out_batch_norm)
        h = self.dropout4(h)
        return self.classifier(h), out_pnaconv, out_batch_norm


class GCN(torch.nn.Module):
    """
    Neural network with graph convolution network (GCN)
    """
    def __init__(self, config, training=False):
        super().__init__()
        num_classes = config.num_classes
        gene_dim = config.gene_dim
        hidden_dim = config.hidden_dim
        SEED = config.SEED
        p_drop = config.dropout
        torch.manual_seed(SEED)
        self.conv1 = GCNConv(gene_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2 * hidden_dim)
        self.conv3 = GCNConv(2 * hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim // 2)
        self.classifier = Linear(hidden_dim // 2, num_classes)
        self.bn1 = LayerNorm(hidden_dim)
        self.bn2 = LayerNorm(2 * hidden_dim)
        self.bn3 = LayerNorm(hidden_dim)
        self.bn4 = LayerNorm(hidden_dim // 2)
        self.dropout1 = Dropout(p=config.dropout)
        self.dropout2 = Dropout(p=config.dropout)
        self.dropout3 = Dropout(p=config.dropout)
        self.dropout4 = Dropout(p=config.dropout)
        self.p_drop = p_drop
        self.training = training

    def forward(self, x, edge_index):
        h1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        h1 = self.dropout1(h1)

        h2 = F.elu(self.bn2(self.conv2(h1, edge_index)))
        h2 = self.dropout2(h2)

        h3 = F.elu(self.bn3(self.conv3(h2, edge_index)))
        h3 = self.dropout3(h3)

        h4_in = h3 + h1

        out_pnaconv = self.conv4(h4_in, edge_index)
        out_batch_norm = self.bn4(out_pnaconv)
        h = F.elu(out_batch_norm)
        h = self.dropout4(h)
        return self.classifier(h), out_pnaconv, out_batch_norm


class GraphSAGE(torch.nn.Module):
    def __init__(self, config, training=False):
        super().__init__()
        num_classes = config.num_classes
        gene_dim = config.gene_dim
        hidden_dim = config.hidden_dim
        SEED = config.SEED
        p_drop = config.dropout
        torch.manual_seed(SEED)
        self.conv1 = SAGEConv(gene_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, 2 * hidden_dim)
        self.conv3 = SAGEConv(2 * hidden_dim, hidden_dim)
        self.conv4 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.bn1 = LayerNorm(hidden_dim)
        self.bn2 = LayerNorm(2 * hidden_dim)
        self.bn3 = LayerNorm(hidden_dim)
        self.bn4 = LayerNorm(hidden_dim // 2)
        self.classifier = Linear(hidden_dim // 2, num_classes)
        self.dropout1 = Dropout(p=config.dropout)
        self.dropout2 = Dropout(p=config.dropout)
        self.dropout3 = Dropout(p=config.dropout)
        self.dropout4 = Dropout(p=config.dropout)
        self.p_drop = p_drop
        self.training = training

    def forward(self, x, edge_index):
        h1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        h1 = self.dropout1(h1)

        h2 = F.elu(self.bn2(self.conv2(h1, edge_index)))
        h2 = self.dropout2(h2)

        h3 = F.elu(self.bn3(self.conv3(h2, edge_index)))
        h3 = self.dropout3(h3)

        h4_in = h3 + h1

        out_pnaconv = self.conv4(h4_in, edge_index)
        out_batch_norm = self.bn4(out_pnaconv)
        h = F.elu(out_batch_norm)
        h = self.dropout4(h)
        return self.classifier(h), out_pnaconv, out_batch_norm


class GATv2(torch.nn.Module):
    def __init__(self, config, training=False):
        super().__init__()
        num_classes = config.num_classes
        gene_dim = config.gene_dim
        hidden_dim = config.hidden_dim
        SEED = config.SEED
        p_drop = config.dropout
        heads = config.heads
        torch.manual_seed(SEED)
        self.conv1 = GATv2Conv(
            gene_dim, hidden_dim // heads, heads=heads, dropout=p_drop
        )
        self.conv2 = GATv2Conv(
            hidden_dim, (2 * hidden_dim) // heads, heads=heads, dropout=p_drop
        )
        self.conv3 = GATv2Conv(
            (2 * hidden_dim), hidden_dim // heads, heads=heads, dropout=p_drop
        )
        self.conv4 = GATv2Conv(
            hidden_dim,
            (hidden_dim // 2) // heads,
            heads=heads,
            dropout=p_drop
        )
        self.bn1 = LayerNorm(hidden_dim)
        self.bn2 = LayerNorm(2 * hidden_dim)
        self.bn3 = LayerNorm(hidden_dim)
        self.bn4 = LayerNorm(hidden_dim // 2)
        self.classifier = Linear(hidden_dim // 2, num_classes)
        self.dropout1 = Dropout(p=config.dropout)
        self.dropout2 = Dropout(p=config.dropout)
        self.dropout3 = Dropout(p=config.dropout)
        self.dropout4 = Dropout(p=config.dropout)
        self.p_drop = p_drop
        self.training = training

    def forward(self, x, edge_index):
        h1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        h1 = self.dropout1(h1)

        h2 = F.elu(self.bn2(self.conv2(h1, edge_index)))
        h2 = self.dropout2(h2)

        h3 = F.elu(self.bn3(self.conv3(h2, edge_index)))
        h3 = self.dropout3(h3)

        h4_in = h3 + h1

        out_pnaconv = self.conv4(h4_in, edge_index)
        out_batch_norm = self.bn4(out_pnaconv)
        h = F.elu(out_batch_norm)
        h = self.dropout4(h)
        return self.classifier(h), out_pnaconv, out_batch_norm
    

class GraphTransformer(torch.nn.Module):
    def __init__(self, config, training=False):
        super().__init__()
        num_classes = config.num_classes
        gene_dim = config.gene_dim
        hidden_dim = config.hidden_dim
        SEED = config.SEED
        p_drop = config.dropout
        heads = config.heads
        torch.manual_seed(SEED)
        self.conv1 = TransformerConv(
            gene_dim, hidden_dim // heads, heads=heads
        )
        self.conv2 = TransformerConv(
            hidden_dim, (2 * hidden_dim) // heads, heads=heads
        )
        self.conv3 = TransformerConv(
            2 * hidden_dim, hidden_dim // heads, heads=heads
        )
        self.conv4 = TransformerConv(
            hidden_dim, (hidden_dim // 2) // heads, heads=heads
        )
        self.bn1 = LayerNorm(hidden_dim)
        self.bn2 = LayerNorm(2 * hidden_dim)
        self.bn3 = LayerNorm(hidden_dim)
        self.bn4 = LayerNorm(hidden_dim // 2)
        self.classifier = Linear(hidden_dim // 2, num_classes)
        self.dropout1 = Dropout(p=config.dropout)
        self.dropout2 = Dropout(p=config.dropout)
        self.dropout3 = Dropout(p=config.dropout)
        self.dropout4 = Dropout(p=config.dropout)
        self.p_drop = p_drop
        self.training = training

    def forward(self, x, edge_index):
        h1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        h1 = self.dropout1(h1)

        h2 = F.elu(self.bn2(self.conv2(h1, edge_index)))
        h2 = self.dropout2(h2)

        h3 = F.elu(self.bn3(self.conv3(h2, edge_index)))
        h3 = self.dropout3(h3)

        h4_in = h3 + h1

        out_pnaconv = self.conv4(h4_in, edge_index)
        out_batch_norm = self.bn4(out_pnaconv)
        h = F.elu(out_batch_norm)
        h = self.dropout4(h)
        return self.classifier(h), out_pnaconv, out_batch_norm
