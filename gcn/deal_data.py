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
from torch_geometric.data import InMemoryDataset, download_url

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


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        download_url(url, self.raw_dir)
        ...

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def graph_batch():
    print("=========================================================")
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[0], [0], [0]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, )
    batch_list = [data, data, data]
    graph_batch = Batch.from_data_list(batch_list)
    print(graph_batch)
    print(graph_batch[1])
    # DataBatch(x=[9, 1], edge_index=[2, 12], batch=[9], ptr=[4])
    batch_list = graph_batch.to_data_list()
    print(batch_list)


def graph_dataloader():
    print("=========================================================")
    dataset = TUDataset(root='./', name='ENZYMES', use_node_attr=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for idx, batch in enumerate(loader):
        print(batch)


if __name__ == '__main__':
    graph_data()
    graph_batch()
    graph_dataloader()
