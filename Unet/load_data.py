import os
import numpy as np
import torch
import pickle

from PIL import Image
from os.path import splitext
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


class CarvanaDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '_mask'):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in os.listdir(images_dir)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask = Image.open(list(self.mask_dir.glob(name + self.mask_suffix + '.*'))[0])
        img = Image.open(list(self.images_dir.glob(name + '.*'))[0])
        assert img.size == mask.size
        # 缩放
        w, h = img.size
        newW, newH = int(self.scale * w), int(self.scale * h)
        img = img.resize((newW, newH), resample=Image.BICUBIC)

        img = np.asarray(img).transpose((2, 0, 1))  # [w h c]->[c w h]
        mask = mask.resize((newW, newH), resample=Image.NEAREST)
        mask = np.asarray(mask)

        return torch.as_tensor(img.copy()).float().contiguous(), torch.as_tensor(mask.copy()).long().contiguous()


class MemoryDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '_mask'):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in os.listdir(images_dir)]
        # self.init_data()
        self.load_data()

    def init_data(self):
        self.memory = {}
        for i in tqdm(range(len(self.ids)), desc="init dataset memory"):
            name = self.ids[i]
            mask = Image.open(list(self.mask_dir.glob(name + self.mask_suffix + '.*'))[0])
            img = Image.open(list(self.images_dir.glob(name + '.*'))[0])
            assert img.size == mask.size
            # 缩放
            w, h = img.size
            newW, newH = int(self.scale * w), int(self.scale * h)
            img = img.resize((newW, newH), resample=Image.BICUBIC)

            img = np.asarray(img).transpose((2, 0, 1))  # [w h c]->[c w h]
            mask = mask.resize((newW, newH), resample=Image.NEAREST)
            mask = np.asarray(mask)
            self.memory[i] = (img, mask)
        with open('./data/processed/memory.pickle', 'wb') as f:
            # 序列化并保存到文件
            pickle.dump(self.memory, f)

    def load_data(self):
        with open('./data/processed/memory.pickle', 'rb') as f:
            self.memory = pickle.load(f)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img, mask = self.memory[idx]
        return torch.as_tensor(img.copy()).float().contiguous(), torch.as_tensor(mask.copy()).long().contiguous()


def get_dataloader(batch_size=256, img_scale=0.5):
    data_path = "./data"
    images_dir = os.path.join(data_path, "imgs")
    mask_dir = os.path.join(data_path, "masks")
    dataset = MemoryDataset(images_dir, mask_dir, img_scale)
    train_dataset, valid_dataset = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    info = {
        "train_count": len(train_dataset),
        "valid_count": len(valid_dataset),
    }
    print(info)
    return train_loader, valid_loader, info


if __name__ == '__main__':
    get_dataloader()
