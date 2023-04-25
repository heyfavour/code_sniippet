import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


class LearnDataset(Dataset):
    def __init__(self):
        self.data = np.array([i for i in range(1000)]).reshape((-1, 2))
        # print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        idx_data = self.data[index]
        x = torch.FloatTensor(idx_data)
        y = torch.Tensor(x[1])
        return x, y  # [65] [1]


def get_dataloader():
    dataset = LearnDataset()  #
    train_num = int(0.7 * len(dataset))
    lengths = [train_num, len(dataset) - train_num]
    trainset, validset = random_split(dataset, lengths)
    train_loader = DataLoader(trainset, batch_size=8, shuffle=True, drop_last=True)
    valid_loader = DataLoader(validset, batch_size=12, shuffle=True, drop_last=True)
    return train_loader, valid_loader


if __name__ == '__main__':
    dataset = LearnDataset()
    print(dataset[0])
    train, valid = get_dataloader()
    for idx, (x, y) in enumerate(train):
        print(x, y)
