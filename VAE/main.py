import sys

import torch
import random
import datetime
import numpy as np

from load_data import get_loader
from model import VAE
from torch.cuda import amp


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD


if __name__ == '__main__':
    same_seeds(1265)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE()
    model.to(device=device)
    # 参数
    epochs = 100
    batch_size = 512
    lr = 0.001
    max_clip = 5.0

    start_time = datetime.datetime.now()
    train_loader, valid_loader, info = get_loader(batch_size)
    end_time = datetime.datetime.now()
    print("load_time", end_time - start_time)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    print(f"参数总量:{sum(p.numel() for p in model.parameters())}")
    scaler = amp.GradScaler()
    best_loss = float("inf")
    kld_weight = 0.00025

    for epoch in range(epochs):
        model.train()
        start_time = datetime.datetime.now()
        for idx, img in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device=device, dtype=torch.float32)
            with amp.autocast():
                recons, input, mu, log_var = model(img)
                loss = model.loss_function(recons, input, mu, log_var, kld_weight)
            scaler.scale(loss).backward()
            # clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip)
            # amp update
            scaler.step(optimizer)
            scaler.update()
            if idx % 10 == 0: print(f"[TRAIN] {idx} [loss] {loss:.6f}")
        end_time = datetime.datetime.now()
        print(f"[EPOCH] {epoch} [TIME] {end_time - start_time}")
        model.eval()
        start_time = datetime.datetime.now()
        valid_loss = 0
        with torch.no_grad():
            for idx, img in enumerate(valid_loader):
                optimizer.zero_grad()
                img = img.to(device=device, dtype=torch.float32)
                recons, input, mu, log_var = model(img)
                loss = model.loss_function(recons, input, mu, log_var, kld_weight)
                valid_loss += loss.item()
        end_time = datetime.datetime.now()
        valid_loss = valid_loss / (idx + 1)
        print(f"[VALID] {epoch} [loss] {loss:.6f} [TIME] {end_time - start_time}")
        if valid_loss < best_loss:
            print(f"BEST {valid_loss:.6f}")
            torch.save(model.state_dict(), f"best.pth")
            best_loss = valid_loss
