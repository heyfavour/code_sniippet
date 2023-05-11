import numpy
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

def rnncell():
    cell = nn.RNNCell(4,2)
    data = torch.randn(3,1,4)#seq_len,batch_size input_size
    hidden = torch.zeros(1,2)#batch_size hidden_size
    print(data)
    print(hidden)
    print("===================================")
    for i,input in enumerate(data):
        print(input)
        hidden = cell(input,hidden)
        print(hidden)

def rnn():
    rnn = nn.RNN(input_size=4,hidden_size=2,num_layers=2,batch_first=True)
    input = torch.Tensor(numpy.array([i for i in range(1,13)]).reshape(1,3,4))#batch_size seq_len feature
    print(input)#整个输入序列
    print("===================================")
    out,hidden = rnn(input,None)
    print(out)#h1->hn
    print(hidden)#hn
    print(out[:, -1, :])#ouput


if __name__ == '__main__':
    #cnn()
    #rnncell()
    rnn()
