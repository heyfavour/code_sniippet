import torch
from torch_geometric.utils import one_hot


def get_one_hot():
    m = torch.tensor([[1], [2], [3], [4], [2], [1], [3]], dtype=torch.long)
    m = m - 1
    print(m)
    m = m.view(-1)
    print(m)
    m = one_hot(m)
    print(m)


if __name__ == '__main__':
    get_one_hot()
