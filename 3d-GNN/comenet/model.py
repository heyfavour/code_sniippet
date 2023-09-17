import torch
import math
import sympy as sym

from torch import nn
from torch_cluster import radius_graph
from torch_scatter import scatter, scatter_min

from model_block import angle_emb, torsion_emb, swish, EmbeddingBlock, SimpleInteractionBlock, Linear


class ComENet(nn.Module):
    r"""
        Args:
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`8.0`)
            num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`256`)
            middle_channels (int, optional): Middle embedding size for the two layer linear block. (default: :obj:`256`)
            out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`3`)
            num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`2`)
            num_output_layers (int, optional): Number of linear layers for the output blocks. (default: :obj:`3`)
    """

    def __init__(
            self,
            cutoff=8.0,
            num_layers=4,
            hidden_channels=256,
            middle_channels=64,
            out_channels=1,
            num_radial=3,
            num_spherical=2,
            num_output_layers=3,
    ):
        super(ComENet, self).__init__()
        self.out_channels = out_channels
        self.cutoff = cutoff
        self.num_layers = num_layers

        if sym is None:
            raise ImportError("Package `sympy` could not be found.")

        act = swish
        self.act = act

        self.feature1 = torsion_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature2 = angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        self.emb = EmbeddingBlock(hidden_channels, act)  # 256 swish

        self.interaction_blocks = torch.nn.ModuleList(
            [
                SimpleInteractionBlock(
                    hidden_channels,
                    middle_channels,
                    num_radial,
                    num_spherical,
                    num_output_layers,
                    hidden_channels,
                    act,
                )
                for _ in range(num_layers)
            ]
        )

        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lin_out = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        self.lin_out.reset_parameters()

    def _forward(self, data):
        z, pos, batch = data.z.long(), data.pos, data.batch  # 21
        num_nodes = z.size(0)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        j, i = edge_index  # 402
        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)

        # Embedding block.
        x = self.emb(z)

        # Calculate distances.
        ##############################################################################
        # 中心原子的第一近邻 第二近邻
        _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)  # 返回每个中心节点最近的边索引
        argmin0[argmin0 >= len(i)] = 0  # 逻辑上不存在吧?
        n0 = j[argmin0]  # 根据边索引找到中心原子第一近邻居节点j n0是原子索引
        # 给最近的原子加上cutoff让去找第二近原子
        add = torch.zeros_like(dist).to(dist.device)
        add[argmin0] = self.cutoff
        dist1 = dist + add

        _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
        argmin1[argmin1 >= len(i)] = 0
        n1 = j[argmin1]  # 根据索引找到第二近原子索引
        # --------------------------------------------------------
        # 邻居原子的第一近邻 第二近邻 为什么要找j？因为i的第一近邻可以是j但是j的第一近邻不一定是i
        _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
        argmin0_j[argmin0_j >= len(j)] = 0
        n0_j = i[argmin0_j]

        add_j = torch.zeros_like(dist).to(dist.device)
        add_j[argmin0_j] = self.cutoff
        dist1_j = dist + add_j

        # i[argmin] = range(0, num_nodes)
        _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
        argmin1_j[argmin1_j >= len(j)] = 0
        n1_j = i[argmin1_j]
        #########################################################################
        # ----------------------------------------------------------
        # 单体到边
        # n0, n1 for i
        n0 = n0[i]
        n1 = n1[i]
        # n0, n1 for j
        n0_j = n0_j[j]
        n1_j = n1_j[j]
        # ----------------------------------------------------------
        # tau: (iref, i, j, jref)
        # when compute tau, do not use n0, n0_j as ref for i and j,
        # because if n0 = j, or n0_j = i, the computed tau is zero
        # so if n0 = j, we choose iref = n1
        # if n0_j = i, we choose jref = n1_j
        # 由于需要计算时j需要考虑i的第一近邻做参考系，可能出现j本身就是i的第一近邻 需要做mask
        mask_iref = n0 == j  # 选出中心原子第一近邻是j的索引
        iref = torch.clone(n0)
        iref[mask_iref] = n1[mask_iref]  # 每个两体 n0第一近邻是j的换成第二近邻 原子索引
        idx_iref = argmin0[i]  # 每个两体  i最近的边的索引
        idx_iref[mask_iref] = argmin1[i][mask_iref]  # 边索引
        # iref 每个两体j-i i最近原子索引 （哪个原子） idx_iref每个两体j-i i最近的边的索引(j-i的索引位置)

        mask_jref = n0_j == i  # 选出邻居原子第一近邻是i的索引
        jref = torch.clone(n0_j)
        jref[mask_jref] = n1_j[mask_jref]  # 两体 j第一近邻是i的换成第二近邻
        idx_jref = argmin0_j[j]
        idx_jref[mask_jref] = argmin1_j[j][mask_jref]  # 同理更新单体数据
        # jref 每个两体j-i j最近原子索引 （哪个原子） idx_jref每个两体j-i j最近的边的索引(j-i的索引位置)

        pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
            vecs, vecs[argmin0][i], vecs[argmin1][i], vecs[idx_iref], vecs[idx_jref]
        )
        # pos_ji j-i 的距离向量
        # pos_in0 j-i 距离向量 j-i  i 最近的向量 i-k 边最近的向量
        # pos_in1 j-i 距离向量 j-i  j 最近的向量 j-k 边最近的向量
        # pos_iref j-i 距离向量 每条j-i i最近的原子的边向量
        # pos_jref j-i 距离向量 每条j-i j最近的原子的边向量
        # pos_in0 - pos_iref 区别 pos_iref 替换了最近是j的距离向量
        # pos_in1 - pos_jref 区别 pos_jref 替换了最近是i的距离向量



        # Calculate angles.
        a = ((-pos_ji) * pos_in0).sum(dim=-1)
        b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta[theta < 0] = theta[theta < 0] + math.pi

        # Calculate torsions.
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(-pos_ji, pos_in0)
        plane2 = torch.cross(-pos_ji, pos_in1)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        phi = torch.atan2(b, a)
        phi[phi < 0] = phi[phi < 0] + math.pi

        # Calculate right torsions.
        plane1 = torch.cross(pos_ji, pos_jref_j)
        plane2 = torch.cross(pos_ji, pos_iref)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        tau = torch.atan2(b, a)
        tau[tau < 0] = tau[tau < 0] + math.pi

        feature1 = self.feature1(dist, theta, phi)
        feature2 = self.feature2(dist, tau)

        # Interaction blocks.
        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, feature1, feature2, edge_index, batch)

        for lin in self.lins:
            x = self.act(lin(x))
        x = self.lin_out(x)

        energy = scatter(x, batch, dim=0)
        return energy

    def forward(self, batch_data):
        return self._forward(batch_data)


if __name__ == '__main__':
    model = ComENet()
