import torch
from torch import nn


def gradient_descent(x, y):
    w = 1.0
    def forward(x):
        y = x * w
        return y

    def cost(xs, ys):
        # loss = 1/N * sum(y-y)**2
        cost = 0
        for x, y in zip(xs, ys):
            pred = forward(x)
            cost += (y - pred) ** 2
        return cost / len(xs)

    def gradient(xs, xy):
        # g = 1/N *SUM(2*X(x*w-y))
        grad = 0
        for x, y in zip(xs, xy):
            grad += 2 * x * (x * w - y)
        return grad / len(xs)

    for epoch in range(100):
        loss = cost(x, y)
        grad = gradient(x, y)
        w = w - 0.01 * grad
        print(f"[epoch:{epoch}] [w={w}][loss={loss}]")

def torch_descent(x,y):
    w = torch.Tensor([1.0])
    w.requires_grad = True
    def forward(x):
        return x*w

    def loss(x,y):
        y_pred = forward(x)
        return (y_pred-y)**2

    for epoch in range(20):
        for _x,_y in zip(x,y):
            l = loss(_x,_y)
            l.backward()
            print(_x,_y,w.grad.item(),l.data)
            print(w.data)
            w.data = w.data - 0.01*w.grad.data
            print(w.data)
            w.grad.data.zero_()





if __name__ == '__main__':
    x = torch.Tensor([i for i in range(1, 5)])
    y = torch.Tensor([i * 2 for i in range(1, 5)])
    #gradient_descent(x, y)
    torch_descent(x, y)
