"""
InMemoryDataset: 当需要将数据集完整地存入内存的时候使用
Dataset:
transform：用于在神经网络计算前对数据进行变换，例如数据增强
pre_transform：用于在将数据集保存至硬盘之前进行预处理，所以一半将比较重的预处理任务放在这个阶段
pre_filter：和pre_transform作用于同一阶段，用于数据的预过滤。
tree:
    process_dir:存放处理后的数据，一般是 pt 格式 ( 由我们重写process()方法实现)。
    raw_dir:存放原始数据的路径，一般是 csv、mat 等格式
        A:邻接矩阵 [74564,2]
        graph_indicator:每一行表示一个节点属于哪个图 [19580 1]
        graph_labels:图的标签 y 每一行表示一个图属于哪个类别
        node_attributes:19580个节点的属性向量 [19580 21] F
        node_label:[19580 1] 节点所属的原子类型
"""
import os.path as osp
from typing import List

import torch

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_tu_data
from torch_geometric.loader import DataLoader


class GCNDataset(InMemoryDataset):
    def __init__(self, root: str, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # _download
        # _process -> torch.save self.processed_paths[0]
        out = torch.load(self.processed_paths[0])
        self.data, self.slices, self.sizes = out

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, '../data', "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, '../data', "processed")

    @property
    def num_node_labels(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, "ENZYMES")  # ENZYMES prefix
        # read_tu_data
        # name ['A', 'graph_indicator', 'graph_labels', 'node_attributes', 'node_labels']
        # edge_attributes edge_labels
        # edge_index [74564 2] 转置 [2 74564]
        # graph_indicator [19580] 节点归属图 batch [19580] node_info
        # node_attributes [19580 18]
        # node_labels [19580] unsqueeze [19580 1] node_labels = node_labels - 1  one_hot(node_labels) [19580 3]
        # x = cat([node_attributes, node_labels])
        # edge_attr = cat([edge_attributes, edge_labels]) None
        # graph_attributes 回归
        # graph_labels 分类
        # num_nodes = 19580 x.size(0)
        # remove_self_loops -> coalesce -> Data Data(x=[19580, 21], edge_index=[2, 74564], y=[600])
        # data, slices, sizes
        torch.save((self._data, self.slices, sizes), self.processed_paths[0])
        # .\data\processed\data.pt

    # def download(self):
    #     pass


if __name__ == '__main__':
    dataset = GCNDataset(root='./')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for idx, batch in enumerate(dataloader):
        print(idx, batch)
        print(batch.batch)
        print(batch.num_graphs)
