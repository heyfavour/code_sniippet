import sys, os
import math
import time
import random
import numpy as np
import torch

from torch.optim import AdamW
from torch.cuda import amp

from model.unet import UNet
from model.diffusion import GaussianDiffusion
from load_data import get_dataloader
import torchvision
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
    train_steps = 1000
    unet = UNet(T=train_steps, ch=16, ch_mult=[1, 4, 8], attn=[2], num_res_blocks=2, dropout=0.15).to(device)
    diffusion = GaussianDiffusion(
        unet,
        timesteps=train_steps,
        batch=batch_size,
        device=device,
    )

    diffusion = diffusion.to(device)
    print("参数总量:", sum(p.numel() for p in diffusion.parameters()))
    ################################################################################################
    dataloader, info = get_dataloader(batch_size=batch_size)
    optimizer = AdamW(diffusion.parameters(), lr=lr)
    batch_count = math.ceil(info['count']/ (batch_size))
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=batch_count, T_mult=2)
    ################################################################################################ train
    epoch_num = 127
    max_clip = 1.0
    scaler = amp.GradScaler()
    for epoch in range(epoch_num):
        epoch_loss = 0
        diffusion.train()
        for idx, (data, label) in enumerate(dataloader):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            with amp.autocast():
                loss = diffusion(data, label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_clip)
            # amp update
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
            if idx % 50 == 0:
                print(f"[IDX] {idx} [LOSS] {loss.item()}")
        print(f"[EPOCH] {epoch:<3d} {epoch_loss:.4f}")
        time.sleep(0.01)
        #sample
        diffusion.eval()
        images = diffusion.sample_loop(10, torch.arange(10).to(device))
        os.makedirs(f"./results/{epoch}", exist_ok=True)
        for i in range(10):
            torchvision.utils.save_image(images[i], f'./results/{epoch}/{i}.jpg')
    torch.save(diffusion.state_dict(), "./model.pth")
