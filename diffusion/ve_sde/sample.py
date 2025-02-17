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

    lr = 1e-3
    batch_size = 256
    unet = UNet(ch=16, ch_mult=[1, 2, 4], attn=[2], num_res_blocks=2, dropout=0.15).to(device)
    sde = VESDE(sigma_min=0.01, sigma_max=50, N=1000, device=device)
    diffusion = GaussianDiffusion(
        unet,
        sde=sde,
        batch=batch_size,
        device=device,
    )
    diffusion.load_state_dict(torch.load("./model.pth",weights_only=True))
    diffusion = diffusion.to(device)
    print("参数总量:", sum(p.numel() for p in diffusion.parameters()))
    ################################################################################################
    dataloader, info = get_dataloader(batch_size=batch_size)
    optimizer = AdamW(diffusion.parameters(), lr=lr)
    batch_count = math.ceil(info['count'] / (batch_size))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=batch_count, T_mult=2)
    ################################################################################################ train
    diffusion.to(device)
    diffusion.eval()
    images = diffusion.sample(10, labels=torch.arange(10).to(device))
    os.makedirs(f"./results/sample", exist_ok=True)
    for i in range(10):
        torchvision.utils.save_image(images[i], f'./results/sample/{i}.jpg')
