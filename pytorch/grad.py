import torch
from torch import autograd


# 自动微分
def grad_demo_1():
    # 变量围绕tensor对象
    x = autograd.Variable(torch.Tensor([1., 2., 3]), requires_grad=True)
    # 您可以使用.data属性访问数据.
    print(x.data)

    # 你也可以用变量来做与张量相同的运算.
    y = autograd.Variable(torch.Tensor([4., 5., 6]), requires_grad=True)
    print(y.data)

    z = x + y
    print(z.data)

    # 我们来将z中所有项作和运算
    s = z.sum()

    print(s)
    print(s.grad_fn)

    # 但是z知道一些额外的东西.
    print(z.grad_fn)
    s.backward()
    print(x.grad)
    print(x.data)


def grad_demo_2():
    x = autograd.Variable(torch.Tensor([2.]), requires_grad=True)
    y = autograd.Variable(torch.Tensor([5.]), requires_grad=True)
    f = torch.log(x) + x * y - torch.sin(y)
    print(f)
    f.backward()
    print(x.grad)
    print(y.grad)


def grad_demo_3():
    x1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, requires_grad=True)
    x2 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, requires_grad=True)
    y = torch.tensor(0)

    x3 = torch.zeros((2,))
    x3[0] = torch.sum(x1[:3] * x2[:3])
    x3[1] = torch.sum(x1[3:] * x2[3:])
    x4 = torch.sum(x3)
    loss = x4-y
    loss.backward()
    print(x1.grad)
    print(x2.grad)


if __name__ == '__main__':
    grad_demo_3()
