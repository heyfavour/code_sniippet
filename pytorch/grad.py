import torch
from torch import autograd

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