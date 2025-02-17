import sys, os
import math
import time
import random
import numpy as np
import torch
import torchvision

from torch.optim import AdamW

from model.unet import UNet
from model.sde import VESDE
from model.diffusion import GaussianDiffusion
from load_data import get_dataloader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seed(96)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 2e-4
    batch_size = 256
    unet = UNet(ch=16, ch_mult=[1, 2, 4], attn=[2], num_res_blocks=2, dropout=0.15).to(device)
    sde = VESDE(sigma_min=0.01, sigma_max=50, N=1000).to(device)
    diffusion = GaussianDiffusion(
        unet,
        sde=sde,
        batch=batch_size,
        device=device,
    )

    diffusion = diffusion.to(device)
    print("参数总量:", sum(p.numel() for p in diffusion.parameters()))
    ################################################################################################
    dataloader, info = get_dataloader(batch_size=batch_size)
    optimizer = AdamW(diffusion.parameters(), lr=lr)
    batch_count = math.ceil(info['count'] / (batch_size))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=batch_count, T_mult=2)
    ################################################################################################ train
    epoch_num = 32
    max_clip = 1.0
    for epoch in range(epoch_num):
        epoch_loss = 0
        diffusion.train()
        for idx, (data, label) in enumerate(dataloader):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            loss = diffusion(data, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_clip)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            if idx % 50 == 0:
                print(f"[IDX] {idx} [LOSS] {loss.item()}")
        print(f"[EPOCH] {epoch:<3d} {epoch_loss:.4f}")
        time.sleep(0.01)
        # sample
        if not math.log2(epoch+1).is_integer(): continue
        diffusion.eval()
        images = diffusion.sample(10, labels=torch.arange(10).to(device))
        os.makedirs(f"./results/{epoch+1}", exist_ok=True)
        for i in range(10):
            torchvision.utils.save_image(images[i], f'./results/{epoch+1}/{i}.jpg')
        torch.save(diffusion.state_dict(), "./model.pth")
