import datetime

from model_block import *

from typing import Callable, Union

import torch
from torch import Tensor

from torch_geometric.nn import radius_graph
from torch_geometric.nn.resolver import activation_resolver  # 激活函数
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter


class DimeNet(torch.nn.Module):
    r"""
    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        num_bilinear (int): Size of the bilinear layer tensor.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act (str or Callable, optional): The activation function.
            (default: :obj:`"swish"`)
    """

    url = ('https://github.com/klicperajo/dimenet/raw/master/pretrained/dimenet')

    def __init__(
            self,
            hidden_channels: int,
            out_channels: int,
            num_blocks: int,
            num_bilinear: int,
            num_spherical: int,
            num_radial: int,
            cutoff: float = 5.0,
            max_num_neighbors: int = 32,
            envelope_exponent: int = 5,
            num_before_skip: int = 1,
            num_after_skip: int = 2,
            num_output_layers: int = 3,
            act: Union[str, Callable] = 'swish',
    ):
        super().__init__()

        if num_spherical < 2:  # 球面谐波>1
            raise ValueError("'num_spherical' should be greater than 1")

        act = activation_resolver(act)  # swish 激活函数 Swish 比 ReLU 在更深层次的模型上工作得更好

        self.cutoff = cutoff  # 截断 5.0
        self.max_num_neighbors = max_num_neighbors  # 32 cutoff最大邻居数
        self.num_blocks = num_blocks  # block层数
        # 球贝塞尔基(径向基函数个数 6 ,截断 5,平滑切割形状 5)
        # self.envelope(dist) = (1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2) * (x < 1.0)
        # self.envelope(dist) * (W(envelope_exponent) * dist).sin()
        # sin(平滑距离*liner(dist,6)->[边,6]
        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        # 球谐函数 (球面谐波 7 径向基函数个数 6 ,截断 5,平滑切割形状 5)
        #  self.sph_funcs n个球谐函数
        #  self.bessel_funcs n*k个 贝塞尔函数
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)#->[ijk 42]

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.output_blocks = torch.nn.ModuleList([
            OutputBlock(num_radial, hidden_channels, out_channels, num_output_layers, act) for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionBlock(
                hidden_channels,  # 隐藏层
                num_bilinear,  # 双线性
                num_spherical,  # 球谐函数
                num_radial,  # 径向基
                num_before_skip,
                num_after_skip,
                act,
            ) for _ in range(num_blocks)
        ])

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def forward(self, z: Tensor, pos: Tensor, batch: OptTensor = None) -> Tensor:
        start_time = datetime.datetime.now()
        # pos cutoff 32  ->[2 N]
        # 按照batch新edge_index 取最近的32个
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        # i 原子  j 原子  idx_i 三体的i idx_j 三体的j idx_k 三体的k  idx_kj kj的边索引 ji的边索引
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(edge_index, num_nodes=z.size(0))#

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)#向量点积求和
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)#叉积求模
        angle = torch.atan2(b, a)#每个ijk的弧度
        rbf = self.rbf(dist)#ij距离->[边,6] 每条边的距离
        sbf = self.sbf(dist, angle, idx_kj)#ij距离 ijk角度 jk边索引->[ijk n*k]
        #1.3s 0.75s

        # Embedding block.
        x = self.emb(z, rbf, i, j)#原子 [边,6] row col
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))#[单体 1] 1含有 ijk的信息
        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))
        #1.3s 0.55
        if batch is None:
            return P.sum(dim=0)
        else:
            return scatter(P, batch, dim=0, reduce='sum')


class DimeNetPlusPlus(DimeNet):
    r"""The DimeNet++ from the `"Fast and Uncertainty-Aware
    Directional Message Passing for Non-Equilibrium Molecules"
    <https://arxiv.org/abs/2011.14115>`_ paper.

    :class:`DimeNetPlusPlus` is an upgrade to the :class:`DimeNet` model with
    8x faster and 10% more accurate than :class:`DimeNet`.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Size of embedding in the interaction block.
        basis_emb_size (int): Size of basis embedding in the interaction block.
        out_emb_channels (int): Size of embedding in the output block.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (str or Callable, optional): The activation funtion.
            (default: :obj:`"swish"`)
    """

    url = ('https://raw.githubusercontent.com/gasteigerjo/dimenet/'
           'master/pretrained/dimenet_pp')

    def __init__(
            self,
            hidden_channels: int,
            out_channels: int,
            num_blocks: int,
            int_emb_size: int,
            basis_emb_size: int,
            out_emb_channels: int,
            num_spherical: int,
            num_radial: int,
            cutoff: float = 5.0,
            max_num_neighbors: int = 32,
            envelope_exponent: int = 5,
            num_before_skip: int = 1,
            num_after_skip: int = 2,
            num_output_layers: int = 3,
            act: Union[str, Callable] = 'swish',
    ):
        act = activation_resolver(act)

        super().__init__(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            num_bilinear=1,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
        )

        # We are re-using the RBF, SBF and embedding layers of `DimeNet` and
        # redefine output_block and interaction_block in DimeNet++.
        # Hence, it is to be noted that in the above initalization, the
        # variable `num_bilinear` does not have any purpose as it is used
        # solely in the `OutputBlock` of DimeNet:
        self.output_blocks = torch.nn.ModuleList([
            OutputPPBlock(
                num_radial,
                hidden_channels,
                out_emb_channels,
                out_channels,
                num_output_layers,
                act,
            ) for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionPPBlock(
                hidden_channels,
                int_emb_size,
                basis_emb_size,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                act,
            ) for _ in range(num_blocks)
        ])

        self.reset_parameters()


if __name__ == '__main__':
    model = DimeNet(hidden_channels=128, out_channels=1, num_blocks=6, num_bilinear=8, num_spherical=7, num_radial=6)
