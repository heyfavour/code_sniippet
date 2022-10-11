import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

objects_list = ['aeroplane', 'chair', 'person', 'bus', 'car', 'horse', 'motorbike', 'bicycle', 'boat', 'cat', 'bottle',
                'sofa', 'tvmonitor', 'pottedplant', 'cow', 'train', 'diningtable', 'bird', 'dog', 'sheep']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448)),
    transforms.ToTensor()
])


def xywhc2label(bboxs, S, B, num_classes):
    # bboxs is a xywhc list: [(x,y,w,h,c),(x,y,w,h,c),....]
    label = np.zeros((S, S, 5 * B + num_classes))  # (7,7,30)
    for x, y, w, h, c in bboxs:
        x_grid, y_grid = int(x // (1.0 / S)), int(y // (1.0 / S))  # 格子
        label[y_grid, x_grid, 0:5] = np.array([x, y, w, h, 1])  # x,y,w,h,iou
        label[y_grid, x_grid, 5:10] = np.array([x, y, w, h, 1])
        label[y_grid, x_grid, 10 + c] = 1  # class
    return label


class YOLODataset(Dataset):
    def __init__(self, S, B, num_classes, transform):
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.transform = transform
        self.filenames = os.listdir("../data/JPEGImages")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        #
        file = self.filenames[idx]
        img = np.array(Image.open(f"../data/JPEGImages/{file}"))  # [96 96 3]
        img = self.transform(img)
        xywhc = []
        with open(os.path.join("../data/Labels/", file.split('.')[0]), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                x, y, w, h, c = float(line[0]), float(line[1]), float(line[2]), float(line[3]), int(line[4])
                xywhc.append((x, y, w, h, c))
        label = torch.Tensor(xywhc2label(xywhc, 7, 2, 20))
        return img, label


def get_dataloder(batch_size=32):
    dataset = YOLODataset(S=7, B=2, num_classes=20, transform=transform)

    dataset_size = len(dataset)
    train_size, val_size = int(dataset_size * 0.8), int(dataset_size * 0.1)  # 训练集 验证集
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    dataset = YOLODataset(S=7, B=2, num_classes=20, transform=transform)
    img, label = dataset[0]
    print(img, label)
