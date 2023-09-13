import datetime
import random
import torch
import torch.optim as optim
import numpy as np

from model import DimeNet, DimeNetPlusPlus
from load_data import QM9_dataloader
from schedules import LinearWarmupExponentialDecay

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seed(99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, train_count, valid_count = QM9_dataloader()
    # model = DimeNet(
    #     hidden_channels=128,
    #     out_channels=1,
    #     num_blocks=6,
    #     num_bilinear=8,
    #     num_spherical=7,
    #     num_radial=6,
    #     cutoff=5.0,
    #     envelope_exponent=5,
    #     num_before_skip=1,
    #     num_after_skip=2,
    #     num_output_layers=3,
    # ).to(device)
    model = DimeNetPlusPlus(
        hidden_channels=128,
        out_channels=1,
        num_blocks=4,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        max_num_neighbors=32,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
    ).to(device)
    learning_rate =  0.001
    eps = 1e-7
    weight_decay =  0.000002
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,eps=eps,weight_decay=weight_decay)
    warmup_steps = 4000
    decay_steps = 4500000
    decay_rate = 0.01
    staircase = False
    scheduler = LinearWarmupExponentialDecay(optimizer,  warmup_steps, decay_steps, decay_rate, staircase)
    criterion = torch.nn.L1Loss()

    model.train()
    for epoch in range(1):
        epoch_loss = 0
        start_time = datetime.datetime.now()
        for idx, data in enumerate(train_loader):
            batch_start_time = datetime.datetime.now()
            data.to(device)
            optimizer.zero_grad()
            out = model(data.z, data.pos, data.batch)  # [bs,1]
            y = torch.reshape(data.y[:, 3], (-1, 1))
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            batch_end_time = datetime.datetime.now()
            batch_use_time = batch_end_time - batch_start_time
            print(f"[BATCH]:{idx} [batch_use_time]:{batch_use_time}")
            break
        end_time = datetime.datetime.now()
        use_time = end_time - start_time
        print(f"[use_time]:{use_time}")
