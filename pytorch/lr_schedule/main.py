import torch
import random
import numpy as np
from torch import nn
from torch import optim

from torch.utils.tensorboard import SummaryWriter

from model import Model
from load_data import get_dataloader
from schedules import LinearWarmupExponentialDecay


def same_seeds(seed):
    # 每次运行网络的时候相同输入的输出是固定的
    random.seed(seed)
    np.random.seed(seed)  # Numpy module.初始化种子保持一致
    torch.manual_seed(seed)  # 初始化种子保持一致
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    same_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    train, valid = get_dataloader()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # warmup_steps, decay_steps, decay_rate
    scheduler_lr = LinearWarmupExponentialDecay(optimizer, warmup_steps=400, decay_rate=0.1,decay_steps=20000)#

    writer = SummaryWriter("./logs")
    steps = 0
    for epoch in range(10000000):
        model.train()
        train_loss = 0
        for x, y in train:
            steps += 1
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad()
            out = model(x)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler_lr.step()
            train_loss += loss.item()
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], steps)
        model.eval()
        valid_loss = 0
        for x, y in valid:
            x,y = x.to(device),y.to(device)
            with torch.no_grad():
                out = model(x)
                loss = criterion(out, y)
                valid_loss += loss
        writer.add_scalars('loss', {"train": train_loss, "valid": valid_loss}, steps)
