from math import pi as PI
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear
from torch_scatter import scatter
from torch_geometric.nn import radius_graph


class update_e(torch.nn.Module):
    #hidden_channels, num_filters, num_gaussians, cutoff
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super(update_e, self).__init__()
        self.cutoff = cutoff
        self.lin = Linear(hidden_channels, num_filters, bias=False)
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)

    def forward(self, v, dist, dist_emb, edge_index):#原子数nn.emb展开 距离 距离的高展开 边
        j, _ = edge_index#end  start
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)# 距离/cutoff * [0-1]的cos下降
        W = self.mlp(dist_emb) * C.view(-1, 1)#线性层(距离映射)*距离 ege_index_num , out_channel
        v = self.lin(v)#[原子数 特征数]
        print("====================")
        print(W.shape)
        print(v.shape)
        print("===============")
        e = v[j] * W#考虑了起点+距离的矩阵表达
        print(e.shape)
        return e


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters):
        super(update_v, self).__init__()
        self.act = ShiftedSoftplus()
        self.lin1 = Linear(num_filters, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, e, edge_index):#原子特征 , 出发点边特征 egge_indx
        _, i = edge_index#start  end
        out = scatter(e, i, dim=0)#将所有归属到A点的点特征sum求和-> [原子数  新特征]
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)
        return v + out #原特征+新特征


class update_u(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(update_u, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, batch):
        v = self.lin1(v)
        v = self.act(v)
        v = self.lin2(v)
        u = scatter(v, batch, dim=0)
        return u


class emb(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(emb, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))#e^(-x^2)


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()#0.6931471824645996 Ln2

    def forward(self, x):
        return F.softplus(x) - self.shift#ln(1+e^x) - ln2


class SchNet(torch.nn.Module):
    r"""
        The re-implementation for SchNet from the `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper
        under the 3DGN gramework from `"Spherical Message Passing for 3D Molecular Graphs" <https://openreview.net/forum?id=givsRXsOt9r>`_ paper.

        Args:
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            num_layers (int, optional): The number of layers. (default: :obj:`6`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            out_channels (int, optional): Output embedding size. (default: :obj:`1`)
            num_filters (int, optional): The number of filters to use. (default: :obj:`128`)
            num_gaussians (int, optional): The number of gaussians :math:`\mu`. (default: :obj:`50`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`10.0`).
    """

    def __init__(self, energy_and_force=False, cutoff=10.0, num_layers=6, hidden_channels=128, out_channels=1,
                 num_filters=128, num_gaussians=50):
        super(SchNet, self).__init__()

        self.energy_and_force = energy_and_force
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        #
        self.init_v = Embedding(100, hidden_channels)
        self.dist_emb = emb(0.0, cutoff, num_gaussians)#高斯展开

        self.update_vs = torch.nn.ModuleList([update_v(hidden_channels, num_filters) for _ in range(num_layers)])#双线性层
        #interaction  message passing
        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, num_filters, num_gaussians, cutoff) for _ in range(num_layers)])

        self.update_u = update_u(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.init_v.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()
        self.update_u.reset_parameters()

    def forward(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        if self.energy_and_force:
            pos.requires_grad_()
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)#pos,10,batch x: Tensor, r: float, batch ->新的edge_indx
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)#距离 矩阵A各项元素的绝对值平方的总和再开根 [edge_index_num ,3]->[edge_index_num]
        dist_emb = self.dist_emb(dist)#距离向量 高斯展开 #edge_index_num guass_num
        v = self.init_v(z)#[node_num emb_num]
        #卷几次
        for update_e, update_v in zip(self.update_es, self.update_vs):#先过update_e 再过update_v
            e = update_e(v, dist, dist_emb, edge_index)#原子数 距离 距离映射 边 ->#考虑了起点+距离的矩阵表达
            v = update_v(v, e, edge_index)#原子特征 出发点边特征 egge_indx -> node_num feature
        u = self.update_u(v, batch)#预测

        return u

if __name__ == '__main__':
    from torch_geometric.datasets import QM9
    dataset = QM9(root='./data')
    model = SchNet()
    y = model(dataset[0])