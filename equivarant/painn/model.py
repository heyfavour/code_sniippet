from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import properties as properties
import nn as snn
import schnetpack as spk
import schnetpack.transform as trn
import torchmetrics

from schnetpack.datasets import QM9


class PaiNNInteraction(nn.Module):
    r"""PaiNN interaction block for modeling equivariant interactions of atomistic systems."""

    def __init__(self, n_atom_basis: int, activation: Callable):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNNInteraction, self).__init__()
        self.n_atom_basis = n_atom_basis

        self.interatomic_context_net = nn.Sequential(
            snn.Dense(n_atom_basis, n_atom_basis, activation=activation),
            snn.Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

    def forward(self, q: torch.Tensor, mu: torch.Tensor, Wij: torch.Tensor, dir_ij: torch.Tensor, idx_i: torch.Tensor,
                idx_j: torch.Tensor, n_atoms: int):
        """Compute interaction output.

        Args:
            q: scalar input values
            mu: vector input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        # [单体 1 128]  [单体 3 128] [两体 1 384] [两体 1] idx_i idx_j n_atoms
        # q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms
        # inter-atomic
        print("=============================================================")
        x = self.interatomic_context_net(q)
        print(x.shape)
        xj = x[idx_j]
        x = Wij * xj

        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)  # [两体 1 128]
        dq = snn.scatter_add(dq, idx_i, dim_size=n_atoms)  # [单体 1 128]

        muj = mu[idx_j]
        # [370, 1, 128]*[370, 3, 1]->[370, 3, 128] +  [370, 1, 128]*[370, 3, 128]->[370, 3, 128]  ->[370, 3, 128]
        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu = snn.scatter_add(dmu, idx_i, dim_size=n_atoms)  # [单体 3 128]

        q = q + dq  # [单体 1 128]
        mu = mu + dmu  # [单体 3 128]
        return q, mu


class PaiNNMixing(nn.Module):
    r"""PaiNN interaction block for mixing on atom features."""

    def __init__(self, n_atom_basis: int, activation: Callable, epsilon: float = 1e-8):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNNMixing, self).__init__()
        self.n_atom_basis = n_atom_basis

        self.intraatomic_context_net = nn.Sequential(
            snn.Dense(2 * n_atom_basis, n_atom_basis, activation=activation),
            snn.Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )
        self.mu_channel_mix = snn.Dense(
            n_atom_basis, 2 * n_atom_basis, activation=None, bias=False
        )
        self.epsilon = epsilon

    def forward(self, q: torch.Tensor, mu: torch.Tensor):
        """Compute intraatomic mixing.

        Args:
            q: scalar input values
            mu: vector input values

        Returns:
            atom features after interaction
        """
        ## intra-atomic
        mu_mix = self.mu_channel_mix(mu)#[21, 3, 128]->[21, 3, 256]

        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)#[21, 3, 256] -> [21, 3, 128] [21, 3, 128]
        mu_Vn = torch.sqrt(torch.sum(mu_V ** 2, dim=-2, keepdim=True) + self.epsilon)#[21, 3, 128]->[21, 1, 128]

        ctx = torch.cat([q, mu_Vn], dim=-1)#[21, 1, 128] cat [21, 1, 128] = [21, 1, 256]
        x = self.intraatomic_context_net(ctx)#[21, 1, 256]->[21, 1, 384]
        print(x.shape)

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)#[21, 1, 128]
        dmu_intra = dmu_intra * mu_W
        print( torch.sum(mu_V * mu_W, dim=1, keepdim=True).shape)
        print("===================rugu")
        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        q = q + dq_intra + dqmu_intra
        mu = mu + dmu_intra
        return q, mu


class PaiNN(nn.Module):
    """PaiNN - polarizable interaction neural network

    References:

    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

    """

    def __init__(
            self,
            n_atom_basis: int,
            n_interactions: int,
            activation: Optional[Callable] = F.silu,
            max_z: int = 100,
            shared_interactions: bool = False,
            shared_filters: bool = False,
            epsilon: float = 1e-8,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            activation: activation function
            shared_interactions: if True, share the weights across
                interaction blocks.
            shared_interactions: if True, share the weights across
                filter-generating networks.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNN, self).__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cal_rij = spk.atomistic.PairwiseDistances()

        cutoff_fn = spk.nn.cutoff.CosineCutoff(5.0)
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=5.0)

        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)  # 100 128

        self.share_filters = shared_filters
        self.filter_net = snn.Dense(self.radial_basis.n_rbf, self.n_interactions * n_atom_basis * 3,
                                    activation=None)  # 20->3*128*3

        self.interactions = snn.replicate_module(
            lambda: PaiNNInteraction(
                n_atom_basis=self.n_atom_basis, activation=activation
            ),
            self.n_interactions,
            shared_interactions,
        )
        self.mixing = snn.replicate_module(
            lambda: PaiNNMixing(
                n_atom_basis=self.n_atom_basis, activation=activation, epsilon=epsilon
            ),
            self.n_interactions,
            shared_interactions,
        )

        self.outnet = spk.nn.build_mlp(
            n_in=128,
            n_out=1,
            n_hidden=128,
            n_layers=3,
            activation=activation,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        # get tensors from input dictionary
        inputs = self.cal_rij(inputs)  # 计算Rij
        atomic_numbers = inputs[properties.Z]  # 每个原子的Z
        r_ij = inputs[properties.Rij]  # j-i 矢量
        idx_i = inputs[properties.idx_i]  # i
        idx_j = inputs[properties.idx_j]  # j
        n_atoms = atomic_numbers.shape[0]  # 总原子数
        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)  # 距离

        phi_ij = self.radial_basis(d_ij)  # [两体 1 20]  高斯展开
        fcut = self.cutoff_fn(d_ij)[..., None]  # 0.5 * (torch.cos(input * math.pi / cutoff) + 1.0) 余弦优化  [两体 1 1]
        filters = self.filter_net(phi_ij) * fcut  # [两体 1 3*128*3]*[两体 1 1]->[两体 1 3*128*3]
        filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)  # [两体 1 3*128*3] -> 分成 3个 两体 1 3*128

        dir_ij = r_ij / d_ij  # rij归一化
        q = self.embedding(atomic_numbers)[:, None]  # [单体] -> [单体 1 128]
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)  # [单体 3 128] 0

        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            # [单体 1 128]  [单体 3 128] [两体 1 128*3] [两体 1] idx_i idx_j n_atoms
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)

        q = q.squeeze(1)

        y = self.outnet(q)
        idx_m = inputs[properties.idx_m]
        y = snn.scatter_add(y, idx_m, dim_size=inputs["batch_num"])
        return y, mu


if __name__ == '__main__':
    n_atom_basis = 30
    pairwise_distance = spk.atomistic.PairwiseDistances()  # calculates pairwise distances between atoms
    schnet = spk.representation.PaiNN(
        n_atom_basis=128, n_interactions=3,
        radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=5.0),
        cutoff_fn=spk.nn.CosineCutoff(5.0)
    )
    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        postprocessors=[trn.CastTo64(), trn.AddOffsets(QM9.U0, add_mean=True, add_atomrefs=True)]
    )
