import sys
import datetime
import random
import numpy as np

import torch
import torch.optim as optim
from model import PaiNN
from load_data import get_dataloader
from schedules import LinearWarmupExponentialDecay
from torch.utils.tensorboard import SummaryWriter
import schnetpack as spk

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seed(99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qm9 = get_dataloader()
    model = PaiNN(
        n_atom_basis=128,
        n_interactions=3,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    scheduler_lr = LinearWarmupExponentialDecay(optimizer, warmup_steps=3000, decay_rate=0.01, decay_steps=20000)  #
    criterion = torch.nn.L1Loss()
    writer = SummaryWriter("./logs")

    model.train()
    steps = 0
    for epoch in range(500):
        epoch_loss = 0
        start_time = datetime.datetime.now()
        for idx, data in enumerate(qm9.train_dataloader()):
            optimizer.zero_grad()
            out = model(data)
            y = torch.reshape(data.y[:, 4], (-1, 1))
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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
            for idx, data in enumerate(qm9.val_dataloader()):
                out = model(data)  # [bs,1]
                y = torch.reshape(data.y[:, 4], (-1, 1))
                loss = criterion(out, y)
                epoch_loss += loss.item() * data.num_graphs
        print(f"[VALID] loss:{epoch_loss / valid_count}]")
        writer.add_scalar('valid', epoch_loss / valid_count, steps)
