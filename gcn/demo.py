"""
邻接矩阵
特征矩阵

A 连接矩阵
D 度矩阵
A=D^(-1)*S,其中D是度矩阵，S是邻接矩阵。
D^(-1/2)*A*D^(-1/2)HW

pip install torch-geometric
"""
import numpy as np
import torch
from torch_geometric.data import Data

"""
data.x
data.edge_index [2 num_edges]
data.edge_attr  [num_edges,num_edge_features]
data.y
data.pos
"""
"""
# 边数据
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
# 节点特征
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)
# print(data)
# [num_edges 2]
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1], ], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
# print(edge_index.t())
# print(torch.transpose(edge_index, dim0=0, dim1=1))
data = Data(x=x, edge_index=edge_index.t().contiguous())  # contiguous 让存储地址连续
# print(data)
"""
"""
消息传递方案
messagepass
aggr = add/mean/max
"""
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add")
        self.layer = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x->[N in_channel]
        # edge_index [2,E]
        # 1.聚合
        #自连接
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))#x.size() [4,1]
        """
        tensor([[1, 2, 3],      tensor([[1, 2, 3, 0, 1, 2, 3],
             [0, 0, 0]])                [0, 0, 0, 0, 1, 2, 3]])
        """
        # 线性变换
        x = self.layer(x)
        print(x)
        # 归一化
        _from,_to = edge_index
        print(_from)
        print(_to)
        _degree = degree(_to, x.size(0), dtype=x.dtype)#计算度 [4*1]
        print(_degree)
        degree_inv_sqrt = _degree.pow(-0.5)#D^(-1/2)
        print(degree_inv_sqrt)
        degree_inv_sqrt[degree_inv_sqrt == float("inf")] = 0
        print(degree_inv_sqrt)
        #
        norm = degree_inv_sqrt[_from] * degree_inv_sqrt[_to]
        print(norm)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        print("============================")
        print(norm)
        # x_j [E out_channel] [7*x]
        print(x_j)
        return norm.view(-1, 1) * x_j


edge_index = torch.tensor([[1, 2, 3],
                           [0, 0, 0], ], dtype=torch.long)
x = torch.tensor([[1], [1], [1], [1]], dtype=torch.float)

conv = GCNConv(1,2)
output = conv(x,edge_index)
print(output)

