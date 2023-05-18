import torch
import math
import torch.nn.functional as F

from torch import optim


class Net():
    pass


def paration_dataset():
    pass


def average_gradients(model):
    dist = None
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


if __name__ == '__main__':
    run()
