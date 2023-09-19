from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, GATConv,GraphConv


class ImprovedGCN(GCNConv):
    def message(self, x_i, x_j, edge_weight):
        return x_j


dataset = TUDataset(root='/dataset', name='ENZYMES')
data = dataset[0]

print(data)
# add_self_loops=False 168 同输入
# add_self_loops=True  168+37=205
layer = GCNConv(3, 3)
out = layer(data.x, data.edge_index)
