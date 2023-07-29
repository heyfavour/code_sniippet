import torch
from torch_geometric.utils import one_hot
from torch_geometric.utils import coalesce, one_hot, remove_self_loops

def get_one_hot():
    m = torch.tensor([[1], [2], [3], [4], [2], [1], [3]], dtype=torch.long)
    m = m - 1
    print(m)
    m = m.view(-1)
    print(m)
    m = one_hot(m)
    print(m)

def rsp():
    edge_index = torch.tensor([[1, 2, 3,1,1,3],
                               [0, 0, 0,1,2,3], ], dtype=torch.long)
    edge_index,edge_attr = remove_self_loops(edge_index)
    print(edge_index)
    num_nodes = 4
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)
    print(edge_index)

if __name__ == '__main__':
    #get_one_hot()
    rsp()
