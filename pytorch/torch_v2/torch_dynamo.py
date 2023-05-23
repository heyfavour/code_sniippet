import time
import torch
import torch._dynamo as dynamo
import torchvision.models as models

print(torch.__version__)
"""
dynamo 图模式 正向图
    TorchScript torch.jit.script/torch.jit.trace
    TorchFX
    Lazy Tensor
    TorchDynamo
AOTAutograd 自动微分 反向图
    torch dispatch
torchinductor
primtorch:2000个算子用250个基础算子实现 (ATen ops 750 Prim ops 250)  
"""

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


def torch_complile():
    start = time.time()
    compiled_model = torch.compile(foo)
    out = compiled_model(torch.randn(10, 10), torch.randn(10, 10))
    end = time.time()
    print(end - start)


def dynano_compile():
    start = time.time()
    compiled_model = dynamo.optimize("inductor")(foo)
    out = compiled_model(torch.randn(10, 10), torch.randn(10, 10))
    end = time.time()
    print(end - start)


def test():
    model = models.alexnet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    compiled_model = torch.compile(model)

    x = torch.rand(16, 3, 224, 224)
    optimizer.zero_grad()
    start = time.time()
    out = compiled_model(x)
    out.sum().backward()
    optimizer.step()
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    pass
