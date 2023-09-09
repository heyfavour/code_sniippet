import torch
import torch.optim as optim

from model import SchNet
from load_data import QM9_dataloader


if __name__ == '__main__':
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    train_loader, valid_loader, train_count,valid_count = QM9_dataloader()
    model = SchNet(cutoff=3,num_layers=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.L1Loss()

    model.train()
    for epoch in range(10000):
        epoch_loss = 0
        for idx, batch in enumerate(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            out = model(batch)  # [bs,1]
            y = torch.reshape(batch.y[:, 4], (-1, 1))
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            break
        break
        #     epoch_loss += loss.item() * batch.num_graphs
        #     if idx % 10 == 0:
        #         info = f"[epoch:{epoch:>4d}] loss:{loss.item()}]"
        #         print(info)
        # print(f"[EPOCH] loss:{epoch_loss / train_count}]")
        # model.eval()
        # with torch.no_grad():
        #     epoch_loss = 0
        #     for idx, batch in enumerate(valid_loader):
        #         batch.to(device)
        #         out = model(batch)  # [bs,1]
        #         y = torch.reshape(batch.y[:, 4], (-1, 1))
        #         loss = criterion(out, y)
        #         epoch_loss += loss.item() * batch.num_graphs
        # print(f"[VALID] loss:{epoch_loss / valid_count}]")
