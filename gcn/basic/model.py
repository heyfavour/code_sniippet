import torch
from torch_geometric.data import Data
from torch.utils.data import random_split
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader


def mytrain():
    edge_index = torch.tensor([[1, 2, 3],
                               [0, 0, 0], ], dtype=torch.long)
    x = torch.tensor([[1], [1], [1], [2]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    model = GCN()
    out = model(data)
    print(out.size())  # [节点数]


def cora_train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    dataset = Planetoid(root='./data', name='Cora')
    print(dataset)
    print(dataset.num_node_features, dataset.num_classes)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(2):
        for idx, batch in enumerate(dataloader):
            print(idx)
            optimizer.zero_grad()
            print(batch)
            out = model(batch)
            print(out.size())
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
            print(loss)
            loss.backward()
            optimizer.step()


def graph_classification():
    #GCN GCN GAT 准确率可上70
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(dataset.num_node_features, 256)
            self.conv2 = GCNConv(256, 256)
            self.conv3 = GATConv(256, 126)
            self.layer = torch.nn.Sequential(
                torch.nn.Linear(126, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 6),
            )

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            x1 = global_max_pool(x, batch)  # [node_nums,feature]->[graph_nums,feature]
            x2 = global_mean_pool(x, batch)  # [node_nums,feature]->[graph_nums,feature]
            x = x1 + x2
            x = self.layer(x)
            return x

    dataset = TUDataset(root='./data', name='ENZYMES')  # 600
    train_dataset, valid_dataset = random_split(dataset, [540, 60])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=120, shuffle=False)
    model = GCN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(200000):
        model.train()
        train_loss = 0
        train_acc = 0
        for idx, batch in enumerate(train_loader):
            # DataBatch(edge_index=[2, 8098], x=[2062, 3], y=[64], batch=[2062], ptr=[65])
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            train_acc += out.max(dim=1)[1].eq(batch.y).sum().item()
        print(
            f"[Train][{epoch}] 损失[{train_loss / len(train_dataset):.4f}]准确率:[{train_acc / len(train_dataset):.4f}]")
        model.eval()
        valid_loss = 0
        valid_acc = 0
        for idx, batch in enumerate(valid_loader):
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            valid_loss += loss.item() * batch.num_graphs
            valid_acc += out.max(dim=1)[1].eq(batch.y).sum().item()
        print(
            f"[Valid][{epoch}] 损失[{valid_loss / len(valid_dataset):.4f}]准确率:[{valid_acc / len(valid_dataset):.4f}]")


if __name__ == '__main__':
    graph_classification()
