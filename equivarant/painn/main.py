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

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(inputs,device):
    for key,value in inputs.items():
        inputs[key] = value.to(device)
    data["batch_num"] = len(inputs["_idx"])
    return inputs

if __name__ == '__main__':
    set_seed(99)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            data = to_device(data,device)
            optimizer.zero_grad()
            q,mu = model(data)
            y = torch.reshape(data["gap"], (-1, 1))
            loss = criterion(q, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler_lr.step()
            epoch_loss += loss.item() * data["batch_num"]
            steps += 1
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], steps)
            sys.exit(0)
        print(f"[EPOCH]:{epoch} loss:{epoch_loss / 110000}] lr:{optimizer.param_groups[0]['lr']}")
        writer.add_scalar('train', epoch_loss / 110000, steps)
        end_time = datetime.datetime.now()
        use_time = end_time - start_time
        model.eval()
        with torch.no_grad():
            epoch_loss = 0
            for idx, data in enumerate(qm9.val_dataloader()):
                data = to_device(data, device)
                out = model(data)  # [bs,1]
                y = torch.reshape(data.y[:, 4], (-1, 1))
                loss = criterion(out, y)
                epoch_loss += loss.item() * data["batch_num"]
        print(f"[VALID] loss:{epoch_loss / 10000}]")
        writer.add_scalar('valid', epoch_loss / 10000, steps)
