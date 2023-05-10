import torch  # C语言核心
import numpy as np
import torch.nn as nn


def leanrnging_torch():
    data = [[1, 2], [3, 4]]
    torch_data = torch.Tensor(data)
    print(torch_data)

    np_array = np.array(data)
    torch_np = torch.from_numpy(np_array)
    print(torch_np)


def cnn():
    cnn = nn.Conv2d(in_channels=5, out_channels=6, kernel_size=3, stride=1, padding=1)
    input = torch.randn(1, 5, 10, 10)  # batch_size in_channel w h
    print(input.shape)
    print(cnn.weight.shape)#[out in 3 3]
    output = cnn(input)
    print(output.shape)


if __name__ == '__main__':
    cnn()
