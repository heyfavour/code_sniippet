import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader


class CustomTensorDataset(TensorDataset):
    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:self.tensors = tensors.permute(0, 3, 1, 2) #[64 64 3]->[3 64 64]
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float32)),
            transforms.Lambda(lambda x: 2 * x / 255 - 1),  #[0 255]-> [-1 1]
        ])

    def __getitem__(self, index):
        x = self.tensors[index]
        if self.transform: x = self.transform(x)
        return x

    def __len__(self):
        return len(self.tensors)


def get_loader(batch_size=256):
    train = np.load('./data/trainingset.npy', allow_pickle=True)
    test = np.load('./data/testingset.npy', allow_pickle=True)
    train_dataset = CustomTensorDataset(torch.from_numpy(train))
    valid_dataset = CustomTensorDataset(torch.from_numpy(test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True)
    info = {
        "train_num": len(train_dataset),
        "valid_num": len(valid_dataset),
    }
    print(info)
    return train_loader, valid_loader, info


if __name__ == '__main__':
    get_loader(256)
