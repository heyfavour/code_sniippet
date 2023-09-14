import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


class LearnDataset(Dataset):
    def __init__(self):
        self.data = np.array([i for i in range(5000)]).reshape((-1, 2))
        # print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        idx_data = self.data[index]
        x = torch.FloatTensor(idx_data)
        y = (x[0].item()**0.5+x[1].item()**0.5+2)
        y = torch.tensor([y])
        return x, y  # [65] [1]


def get_dataloader():
    dataset = LearnDataset()  #
    train_num = int(0.6 * len(dataset))
    lengths = [train_num, len(dataset) - train_num]
    trainset, validset = random_split(dataset, lengths)
    train_loader = DataLoader(trainset, batch_size=5, shuffle=True, drop_last=True)
    valid_loader = DataLoader(validset, batch_size=5, shuffle=True, drop_last=True)
    return train_loader, valid_loader


if __name__ == '__main__':
    # dataset = LearnDataset()
    # print(dataset[0])
    train, valid = get_dataloader()
    for i in range(2):
        print("=====================================")
        for idx, (x, y) in enumerate(train):
            print(x)
            print(y)
            output = torch.LongTensor([[0,1],[0,1],[0,1],[1,0],[1,0]])
            print(torch.mean((output.argmax(1) == y).float()).item())
        for idx, (x, y) in enumerate(valid):
            print(x, y)
