import torch
from torch import nn

input_dim = 768
output_dim = 768

rank = 8

W = None

W_A = nn.Parameter(torch.empty(input_dim, rank))
W_B = nn.Parameter(torch.empty(rank, output_dim))


def regular_forward(x, W):
    h = x @ W
    return h


def lora_forward(x, W, W_A, W_B):
    h = x @ W
    h = h + x @ (W_A @ W_B)
    return h
