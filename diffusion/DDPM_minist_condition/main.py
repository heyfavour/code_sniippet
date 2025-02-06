import sys
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


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seed(96)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 1e-4
    batch_size = 256
    train_steps = 200
    unet = UNet(T=train_steps, ch=16, ch_mult=[1, 2, 4], attn=[2], num_res_blocks=2, dropout=0.15).to(device)
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
    ################################################################################################ train
    epoch_num = 20
    max_clip = 1.0
    scaler = amp.GradScaler()
    for epoch in range(epoch_num):
        epoch_loss = 0
        for idx, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            with amp.autocast():
                loss = diffusion(data,labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_clip)
            # amp update
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            sys.exit(0)
            if idx % 50 == 0:
                print(f"[IDX] {idx} [LOSS] {loss.item()}")
        print(f"[EPOCH] {epoch} {epoch_loss:.4f}")
        time.sleep(0.01)
        # sample
        images = diffusion.sample_loop(1)
        torchvision.utils.save_image(images[0], f'./results/epoch_{epoch}.jpg')
    torch.save(diffusion.state_dict(), "./model.pth")
