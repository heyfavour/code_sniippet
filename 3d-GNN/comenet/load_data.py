from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
def QM9_dataloader():
    dataset = QM9(root='./data')
    train_dataset, valid_dataset = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False)
    return train_loader,valid_loader,len(train_dataset),len(valid_dataset)


if __name__ == '__main__':
    QM9_dataloader()