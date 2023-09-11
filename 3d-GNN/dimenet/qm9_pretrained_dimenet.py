import argparse
import datetime
import os.path as osp

import torch

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from model import DimeNet, DimeNetPlusPlus

# Model = DimeNetPlusPlus #DimeNet
Model = DimeNet #DimeNet
print(Model)

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
dataset = QM9(path)

# DimeNet uses the atomization energy for targets U0, U, H, and G, i.e.:
# 7 -> 12, 8 -> 13, 9 -> 14, 10 -> 15
idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
dataset.data.y = dataset.data.y[:, idx]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
target_list = [2,3]
# for target in range(12):
for target in target_list:
    # Skip target \delta\epsilon, since it can be computed via
    # \epsilon_{LUMO} - \epsilon_{HOMO}:
    if target == 4:
        continue

    model, datasets = Model.from_qm9_pretrained(path, dataset, target)
    train_dataset, val_dataset, test_dataset = datasets

    model = model.to(device)
    loader = DataLoader(val_dataset, batch_size=256)
    start_time = datetime.datetime.now()
    maes = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            # print(data.z.shape)
            # print(data.pos.shape)
            # print(data.batch.shape)
            pred = model(data.z, data.pos, data.batch)#N,1
            # print(pred.shape)
        if target == 2:homo = pred
        if target == 3:lumo = pred
        mae = (pred.view(-1) - data.y[:, target]).abs()
        maes.append(mae)
    end_time = datetime.datetime.now()

    mae = torch.cat(maes, dim=0)

    # Report meV instead of eV:
    mae = 1000 * mae if target in [2, 3, 4, 6, 7, 8, 9, 10] else mae
    use_time = end_time-start_time
    print(f'Target: {target:02d}, MAE: {mae.mean():.5f} ± {mae.std():.5f},{use_time}')

mae = ((lumo - homo).view(-1)-data.y[:,4]).abs().mean()*1000
print(f'MAE: {mae.mean():.5f}')
