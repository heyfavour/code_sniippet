"""
1. 如何用PyG表示一张图 (torch_geometric.data.Data)
2. 如何用PyG 表示多张图(torch_geometric.data.Batch)
3.如何用PyG表示一系列的图(torch_geometric.data.Dataset)
4.如何用PyG加载一个Batch 的图片(torch_geometric.data.DataLoader)、
"""
import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter


def graph_data():
    # x: 用于存储每个节点的特征，形状是[num_nodes, num_node_features]。
    # edge_index: 用于存储节点之间的边，形状是[2, num_edges]。
    # pos: 存储节点的坐标，形状是[num_nodes, num_dimensions]。
    # y: 存储样本标签。如果是每个节点都有标签，那么形状是[num_nodes, *]；如果是整张图只有一个标签，那么形状是[1, *]。
    # edge_attr: 存储边的特征。形状是[num_edges, num_edge_features]。
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    # x 用于存储节点的属性
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    # pos N x 3 的矩阵，用于表示上述图形当中的所有的Point的坐标
    pos = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    data = Data(x=x, edge_index=edge_index, pos=pos)
    print(data)
    # Data(x=[3, 1], edge_index=[2, 4], pos=[3, 3])


def graph_batch():
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[1], [2], [3]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, )
    batch_list = [data, data, data]
    graph_batch = Batch.from_data_list(batch_list)
    print(graph_batch)
    print(graph_batch[1])
    print(graph_batch.batch)#表示节点但属于哪个图
    x = scatter(graph_batch.x, graph_batch.batch, dim=0, reduce='add')
    print(x)
    # DataBatch(x=[9, 1], edge_index=[2, 12], batch=[9], ptr=[4])
    batch_list = graph_batch.to_data_list()
    print(batch_list)



def graph_dataloader():
    dataset = TUDataset(root='./', name='data', use_node_attr=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for idex, batch in enumerate(loader):
        pass


if __name__ == '__main__':
    graph_data()
    graph_batch()
