import torch
import math
import torch.nn.functional as F

from torch import optim

dist = None


class Net():
    pass


def paration_dataset():
    pass


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data / size


def run(rank, size):
    torch.manual_seed(1234)
    train_set, batch_size = paration_dataset()
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    num_batches = math.ceil(len(train_set.dataset) / float(batch_size))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, label in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, label)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()


def run(rank, size):
    # 点对点 0-1
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        req = dist.isend(tensor=tensor, dst=1)
    else:
        req = dist.irecv(tensor=tensor,src=0)
    req.wait()




if __name__ == '__main__':
    run()
