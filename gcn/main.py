import torch
import torch.optim as optim

from basic.model import GCN
from dataset.memory_dataset import GCNDataset
from torch_geometric.loader import DataLoader

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN().to(device)
    dataset = GCNDataset(root='./data')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(1):
        for idx, batch in enumerate(dataloader):
            print(batch)
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            print(out)
            print(out.size())
            break
    #         loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #         loss.backward()
    #         optimizer.step()
    #         print(idx,loss)