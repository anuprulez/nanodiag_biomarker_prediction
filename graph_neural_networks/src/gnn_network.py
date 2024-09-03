import torch
from torch_geometric.data import Data
from torch.nn import Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, PNAConv
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
        h = self.pnaconv4(h, edge_index)
        h = self.batch_norm4(F.relu(h))
        out = self.classifier(h)
        return out


class GCN(torch.nn.Module):
    '''
    Neural network with graph convolution network (GCN)
    '''
    def __init__(self, config):
        super().__init__()
        num_classes = config["num_classes"]
        gene_dim = config["gene_dim"]
        hidden_dim = config["hidden_dim"]
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        SEED = config["SEED"]
        torch.manual_seed(SEED)
        '''self.conv1 = GCNConv(gene_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2 * hidden_dim)
        self.conv3 = GCNConv(2 * hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim // 2)'''
        self.conv1 = PNAConv(gene_dim, hidden_dim)
        self.conv2 = PNAConv(hidden_dim, 2 * hidden_dim)
        self.conv3 = PNAConv(2 * hidden_dim, hidden_dim)
        self.conv4 = PNAConv(hidden_dim, hidden_dim // 2)
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
