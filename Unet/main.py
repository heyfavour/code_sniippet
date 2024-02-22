import datetime
import sys
import torch
import random
import numpy as np

from model import UNet
from load_data import get_dataloader
from utils import dice_loss
from torch.cuda import amp


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seed(99)
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_channels=3, n_classes=2)
    model = model.to(memory_format=torch.channels_last)  # CNN加速

    #model.load_state_dict(torch.load("./unet_carvana_scale0.5_epoch2.pth", map_location=device))
    model.to(device=device)
    # 参数
    epochs = 200
    batch_size = 32
    lr = 1e-5
    scale = 0.5
    max_clip = 1.0

    start_time = datetime.datetime.now()
    train_loader, valid_loader, info = get_dataloader(batch_size, 0.5)
    end_time = datetime.datetime.now()
    print("load_time",end_time-start_time)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    print(f"参数总量:{sum(p.numel() for p in model.parameters())}")
    scaler = amp.GradScaler()
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        start_time = datetime.datetime.now()
        for idx, (img, mask) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask = mask.to(device=device, dtype=torch.long)
            with amp.autocast():
                mask_pred = model(img)
                loss = criterion(mask_pred, mask) + dice_loss(mask_pred, mask)
            scaler.scale(loss).backward()
            # clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip)
            # amp update
            scaler.step(optimizer)
            scaler.update()
            #scheduler_lr.step()
            if idx%10==0:print(f"[TRAIN] {idx} [loss] {loss:.6f}")
        end_time = datetime.datetime.now()
        print(f"[EPOCH] {epoch} [TIME] {end_time-start_time}")
        model.eval()
        start_time = datetime.datetime.now()
        valid_loss = 0 
        with torch.no_grad():
            for idx, (img, mask) in enumerate(valid_loader):
                optimizer.zero_grad()
                img = img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                mask = mask.to(device=device, dtype=torch.long)
                mask_pred = model(img)
                loss = criterion(mask_pred, mask) + dice_loss(mask_pred, mask)
                valid_loss += loss.item()
        end_time = datetime.datetime.now()
        valid_loss = valid_loss/(idx+1)
        print(f"[VALID] {epoch} [loss] {loss:.6f} [TIME] {end_time-start_time}")
        if valid_loss < best_loss:
            print(f"BEST {valid_loss:.6f}")
            torch.save(model.state_dict(), f"best_{epoch}.pth")
            best_loss = valid_loss

