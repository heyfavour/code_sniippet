import e3nn

irreps = e3nn.o3.Irreps("3x0e+2x1o")
print(irreps)

import torch

alpha, beta, gamma = torch.randn(3)
rotate = irreps.D_from_angles(alpha, beta, gamma)  # 旋转矩阵

print(rotate)

# composition *
# addition +
# multiplication X
# a [x1 y1 z1] b [x2 y2 z2]
# a.view(3,1) * b.view(1,3)
# x1x2 x1y2 x1z2
# y1x2 y1y2 y1z2
# z1x2 z1y2 z1z2

# 可约化 用更小的向量表示
# 点积
# x1x2 + y1y2 + z1z2 不变
# 叉积
# y1z2 - z1y2
# z1x2 - x1z2
# x1y2 - y1x2
# 对称无恒矩阵
# c(x1z2 + z1x2)
# c(x1y2 + y2x2)
# 2y1y2 - x1x2 - z1z2
# c(y1z2 + z1y2)
# c(z1z2 - x1x2)

# 3*3 -> 1 + 3 + 5
# 不可约表示
# L =    0   1   2
# even
# L=0 d=1 scalar
# L=1 d=3 peudo vector
# L=2 d=5
# odd
# L=0 d=1 scalar
# L=1 d=3 vector
# L=2 d=5