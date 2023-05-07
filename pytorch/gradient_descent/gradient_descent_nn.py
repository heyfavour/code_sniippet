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


class Demo(nn.Module):
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
    model = Demo()
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    print(x)
    print(y)
    # print(model.parameters())
    for epoch in range(2):
        print(f"[epoch] [{epoch}]================================")
        predict = model(x)
        optimizer.zero_grad()
        loss = criterion(predict,y)
        print("loss",loss)
        print(predict)
        print("before *****************************")
        print(model.layer.weight.grad,model.layer.bias.grad)
        loss.backward()
        print("after *****************************")
        #print(model.layer.weight,model.layer.bias)
        print(model.layer.weight.grad,model.layer.bias.grad)
        optimizer.step()
        print("weight *****************************")
        print(model.layer.weight,model.layer.bias)


