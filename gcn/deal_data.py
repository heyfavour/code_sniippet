"""
1. 如何用PyG表示一张图 (torch_geometric.data.Data)
2. 如何用PyG 表示多张图(torch_geometric.data.Batch)
3.如何用PyG表示一系列的图(torch_geometric.data.Dataset)
4.如何用PyG加载一个Batch 的图片(torch_geometric.data.DataLoader)、
"""
import torch
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader

from dataset import GCNDataset


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
    batch_list = []
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[0], [0], [0]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, )
    batch_list = [data, data, data]
    graph_batch = Batch.from_data_list(batch_list)
    print(graph_batch)
    # ptr每个数据的范围
    print(graph_batch.ptr)  # [0, 3, 6, 9]
    print(graph_batch.batch)  #
    print(graph_batch[1])
    # DataBatch(x=[9, 1], edge_index=[2, 12], batch=[9], ptr=[4])
    batch_list = graph_batch.to_data_list()
    print(batch_list)


def graph_dataset():
    pass


def graph_dataloader():

    dataset = GCNDataset(root='./')
    # print(dataset)  # [600]
    # print(dataset.data)  # Data(x=[19580, 21], edge_index=[2, 74564], y=[600])
    show_graph(dataset[0])
    # print(dataset[0])  # Data(edge_index=[2, 168], x=[37, 21], y=[1])
    # print(dataset.num_classes)#6
    # print(dataset.num_features)
    # print(dataset.num_node_labels)
    # print("--------------------------------------------------")
    # loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # for idx, batch in enumerate(loader):  #
    #     print(batch)
    #     # print(batch.ptr)
    #     # print(batch.batch)
    #     print(batch[0])
    #     break


def show_graph(data):
    print("show graph ===============================")
    print(data)
    G = to_networkx(data)
    nx.draw(G, with_labels=True)
    # nx.draw(G, with_labels=True, node_color=data.y)
    plt.show()


if __name__ == '__main__':
    # graph_data()
    # graph_batch()
    graph_dataloader()
