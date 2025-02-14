from torchvision import datasets, transforms
from torch.utils.data import DataLoader

_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),#[0,1]->[-1,1]
])


def get_dataloader(batch_size=256):
    dataset = datasets.MNIST('./data', train=True, transform=_transforms, download=True)  # [1 28 28] label
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    info = {
        "count": len(dataset),
    }
    return dataloader, info


if __name__ == '__main__':
    get_dataloader()
