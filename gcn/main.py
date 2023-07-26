import torch

from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class GNN(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.gcn_1 = GCNConv(features,16)
        self.gcn_1 = GCNConv(features,16)