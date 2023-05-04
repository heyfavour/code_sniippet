import torch
import math
from torch import nn
from torch.autograd import Variable


# 定义Embeddings类来实现文本嵌入层，这里s说明代表两个一模一样的嵌入层, 他们共享参数.
# 该类继承nn.Module, 这样就有标准层的一些功能, 这里我们也可以理解为一种模式, 我们自己实现的所有层都会这样去写.
class Embeddings(nn.Module):
    def __init__(self, vocab, dim):
        """类的初始化函数, 有两个参数, d_model: 指词嵌入的维度, vocab: 指词表的大小."""
        # 接着就是使用super的方式指明继承nn.Module的初始化函数, 我们自己实现的所有层都会这样去写.
        super(Embeddings, self).__init__()
        # 之后就是调用nn中的预定义层Embedding, 获得一个词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, dim)
        # 最后就是将d_model传入类中
        self.dim = dim

    def forward(self, x):
        """可以将其理解为该层的前向传播逻辑，所有层中都会有此函数
           当传给该类的实例化对象参数时, 自动调用该类函数
           参数x: 因为Embedding层是首层, 所以代表输入给模型的文本通过词汇映射后的张量"""

        # 将x传给self.lut并与根号下self.d_model相乘作为结果返回
        return self.lut(x) * math.sqrt(self.dim)


def word_embedding():
    """
    词表 10 3维
    1 ->[a b c]
    2 ->[a b c]
    3 ->[a b c]
    4 ->[a b c]
    """
    embedding = nn.Embedding(10, 3)  # 单词总数 dim
    input = torch.LongTensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
    print(input)
    print(input.shape)
    output = embedding(input)
    print(output)
    print(output.shape)


# 定义位置编码器类, 我们同样把它看做一个层, 因此会继承nn.Module
class PositionalEncoding(nn.Module):
    def __init__(self,max_len=4,dim=65):
        super().__init__()
        pe = torch.zeros((max_len, dim))
        position = torch.arange(0, max_len).unsqueeze(1)
        print(pe.shape,position.shape)
        print(position)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        print(div_term.shape)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

if __name__ == '__main__':
    #word_embedding()
    #emb = Embeddings(1000, 65)
    #x = torch.LongTensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
    #y = emb(x)
    #print("embr:", y.shape)
    postition = PositionalEncoding(4,65)
    print(postition.pe)
    #y = postition(y)
    #print(y)
