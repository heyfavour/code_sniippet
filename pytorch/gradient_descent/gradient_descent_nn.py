import torch
from torch import nn
from torch.optim import Adam
import torch.nn.init as init


def init_weight(layer):
    if type(layer) == nn.Linear:
        layer.weight.data.fill_(1.00)
        layer.bias.data.fill_(1.00)
        # layer.weight.data = torch.randn(layer.weight.data.size()) * 0.01
        # layer.bias.data = torch.zeros(layer.bias.data.size())


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)
        init_weight(layer=self.layer)

    def forward(self, x):
        y = self.layer(x)
        return y


if __name__ == '__main__':
    x = torch.Tensor([i for i in range(1, 5)]).reshape(-1,1)
    y = torch.Tensor([i * 2 + 2 for i in range(1, 5)]).reshape(-1,1)
    model = LinearModel()
    criterion = nn.MSELoss()#size_average 求均值 reduce 求和降维
    optimizer = Adam(model.parameters(), lr=0.01)
    print(model.layer.weight.data)
    for epoch in range(1000):
        predict = model(x)
        loss = criterion(predict,y)
        print(f"[epoch:{epoch}] [loss:{loss}]")
        optimizer.zero_grad()
        loss.backward()
        print(model.layer.weight.data,model.layer.bias.data,model.layer.weight.grad.data)
        optimizer.step()#update
        print(model.layer.weight.data,model.layer.bias.data)



