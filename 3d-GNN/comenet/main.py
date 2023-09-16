import os, sys
import datetime
import random
import numpy as np

import torch
import torch.optim as optim

from model import ComENet
from load_data import QM9_dataloader
from schedules import LinearWarmupExponentialDecay
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, train_count, valid_count = QM9_dataloader()
    model = ComENet(cutoff=5.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    scheduler_lr = LinearWarmupExponentialDecay(optimizer, warmup_steps=1000, decay_rate=0.5, decay_steps=50000)  #
    criterion = torch.nn.L1Loss()
    writer = SummaryWriter("./logs")

    model.train()
    steps = 0
    for epoch in range(1200):
        epoch_loss = 0
        start_time = datetime.datetime.now()
        for idx, data in enumerate(train_loader):
            data.to(device)
            optimizer.zero_grad()
            out = model(data)
            y = torch.reshape(data.y[:, 4], (-1, 1))
            loss = criterion(out, y)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler_lr.step()
            epoch_loss += loss.item() * data.num_graphs
            steps += 1
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], steps)
            sys.exit(0)
        print(f"[EPOCH]:{epoch} loss:{epoch_loss / train_count}] lr:{optimizer.param_groups[0]['lr']}")
        writer.add_scalar('train', epoch_loss / train_count, steps)
        end_time = datetime.datetime.now()
        use_time = end_time - start_time
        model.eval()
        with torch.no_grad():
            epoch_loss = 0
            for idx, data in enumerate(valid_loader):
                data.to(device)
                out = model(data)  # [bs,1]
                y = torch.reshape(data.y[:, 4], (-1, 1))
                loss = criterion(out, y)
                epoch_loss += loss.item() * data.num_graphs
        print(f"[VALID] loss:{epoch_loss / valid_count}]")
        writer.add_scalar('valid', epoch_loss / valid_count, steps)
