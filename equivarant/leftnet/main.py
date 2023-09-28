import os
import sys

import torch
import random
import numpy as np

from torch import optim
from torch.optim.lr_scheduler import StepLR, OneCycleLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
import time

from model import LEFTNet
from load_data import QM9_dataloader
from schedules import LinearWarmupExponentialDecay


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, info = QM9_dataloader()

    model = LEFTNet(
        pos_require_grad=False,
        cutoff=8.0,
        num_layers=6,
        hidden_channels=256,
        num_radial=96,
        y_mean=info["mean"],
        y_std=info["std"],
    ).to(device)
    loss_func = torch.nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=0.0005, amsgrad=True, weight_decay=0)
    scheduler_lr = LinearWarmupExponentialDecay(optimizer, warmup_steps=3000, decay_rate=0.01, decay_steps=20000)  #

    model = model.to(device)
    print('参数总量:', sum(p.numel() for p in model.parameters()))

    steps = 0

    for epoch in range(0, 1000):
        model.train()
        loss_accum = 0
        for idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)
            loss = loss_func(out, data.y[:,4].unsqueeze(1))
            loss.backward()
            optimizer.step()
            scheduler_lr.step()
            loss_accum += loss.item() * data.num_graphs
            steps += 1
        print(f"[EPOCH]:{epoch} loss:{loss_accum / info['train_count']}] lr:{optimizer.param_groups[0]['lr']}")
        model.eval()
        _out = torch.Tensor([]).to(device)
        _y = torch.Tensor([]).to(device)
        with torch.no_grad():
            for idx, data in enumerate(valid_loader):
                data.to(device)
                out = model(data)  # [bs,1]
                y = torch.reshape(data.y[:, 4], (-1, 1))
                _out = torch.cat([_out, out.detach_()], dim=0)
                _y = torch.cat([_y, data.y.unsqueeze(1)], dim=0)

        loss = torch.mean(torch.abs(_out - _y)).cpu().item()
        print(f"[VALID] loss:{loss}]")
