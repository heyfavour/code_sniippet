import torch

from torch import nn

import torch.optim as optim
import torch.nn.functional as F

from model import GCN
from dataset import GCNDataset,get_loader

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN().to(device)
    dataset = GCNDataset(root='./')
    dataloader = get_loader(dataset)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    # for epoch in range(10):
    #     for idx, batch in dataloader:
    #         batch = batch.to(device)
    #         optimizer.zero_grad()
    #
    #         out = model(batch)
    #         loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #         loss.backward()
    #         optimizer.step()
    #         print(idx,loss)