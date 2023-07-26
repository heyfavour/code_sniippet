import torch

from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.optim as optim


class GNN(nn.Module):
    def __init__(self, features, num_class):
        super().__init__()
        self.gcn_1 = GCNConv(features, 16)
        self.gcn_1 = GCNConv(16, num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn_1(x, edge_index)
        x = nn.Dropout(nn.ReLU(x), 0.5)
        x = self.gcn_2(x, edge_index)
        return x


def get_data():
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    # 节点特征
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNN().to(device)
    dataloader = get_data().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()  # 定義損失函數，這裡我們使用 binary cross entropy loss

    model.train()
    for epoch in range(100):
        for idx, (data, label) in dataloader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            accuracy = torch.mean((out.argmax(1) == label).float()).item()
