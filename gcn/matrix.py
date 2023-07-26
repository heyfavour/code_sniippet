import numpy as np
import torch
from torch_geometric.utils import degree, add_self_loops

a = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a @ b)
a = torch.Tensor(a)
b = torch.Tensor(b)
print(a @ b)
print("===========================================")
a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
c = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
print("左乘对角矩阵")
print(c @ b)
print("右乘对角矩阵")
print(a @ b @ c)
a = torch.Tensor(a)
b = torch.Tensor(b)
print(a @ b)

print("===========================================")
A = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [0, 1, 1, 0, 1], [1, 1, 1, 1, 0]])
F = np.array([[-1.1, 3.2, 4.2], [0.4, 5.1, -1.2], [1.2, 1.3, 2.1], [1.4, -1.2, 2.5], [1.4, 2.5, 4.5]])
print("A*F 考虑自连接")
A = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
print(A @ F)

print("===========================================")
A = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [0, 1, 1, 0, 1], [1, 1, 1, 1, 0]])
F = np.array([[-1.1, 3.2, 4.2], [0.4, 5.1, -1.2], [1.2, 1.3, 2.1], [1.4, -1.2, 2.5], [1.4, 2.5, 4.5]])
print("A*F 考虑自连接")
A = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
print(A @ F)
A = torch.Tensor(A)
F = torch.Tensor(F)
index_edge = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
                           [4, 3, 4, 3, 4, 1, 2, 4, 0, 1, 2, 3]], dtype=torch.long)
index_edge, _ = add_self_loops(index_edge, num_nodes=5)
f, t = index_edge
D = degree(t, 5, dtype=torch.long)
# print(torch.diag_embed(D))
print("D is")
print(torch.diag_embed(D.pow(-1.0)))
"""
D*A*F
"""
print("D^-1 * A * F")
print(torch.diag_embed(D.pow(-1.0)) @ A @ F)
print("归一化")
print((torch.diag_embed(D.pow(-1.0)) @ A @ torch.diag_embed(D.pow(-1.0))))
print((torch.diag_embed(D.pow(-1.0)) @ A @ torch.diag_embed(D.pow(-1.0))) @ F)
print("归一化修正")
print((torch.diag_embed(D.pow(-0.5)) @ A @ torch.diag_embed(D.pow(-0.5))))
print((torch.diag_embed(D.pow(-0.5)) @ A @ torch.diag_embed(D.pow(-0.5))) @ F)
