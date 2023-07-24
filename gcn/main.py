"""
邻接矩阵
特征矩阵

连接矩阵
度矩阵

D^(-1/2)*A*D^(-1/2)HW

pip install torch-geometric
"""
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv