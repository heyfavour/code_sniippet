import random
import torch
import e3nn

import numpy as np


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seed(99)

    pos = torch.randn((2, 3))
    # [[x0,y0,z0],[x1,y1,z1]
    print(pos)
    node = torch.randn((2, 7))
    # [
    #   [m0,v0x,v0y,v0z,a0x,a0y,a0z],
    #   [m1,v1x,v1y,v10,a1x,a1y,a1z],
    # ]
    print(node)

    scalar = e3nn.o3.Irrep("0e")  # L1=0 even p
    vector = e3nn.o3.Irrep("1o")  # L=1 odd p
    irreps = 1 * scalar + 1 * vector + 1 * vector
    print(irreps)




