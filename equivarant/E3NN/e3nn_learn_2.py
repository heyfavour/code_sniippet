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

# D 3*3 -> 1 + 3 + 5
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


# L X L = |L - L|  + ... + |L + L|
# L 2 X 1 = 1 + 2 + 3
# L 2 X 2 = 0 + 1 + 2 + 3 + 4

tp = e3nn.o3.FullTensorProduct("1o","1o")
print(tp)
a = torch.tensor([1.0,2.0,3.0])
b = torch.tensor([4.0,5.0,6.0])
print(tp(a,b))
# tensor([18.4752, -2.1213,  4.2426, -2.1213, 12.7279,  9.1924, -0.8165, 19.0919, 9.8995])
# (x1x2 + y1y2 + z1z2)/sqrt(3) = 18.4752
# 点积
# x1x2 + y1y2 + z1z2  = 32/sqrt(3) = 18.4752
# 叉积 N = sqrt((2l+1)/4pi)
# y1z2 - z1y2 = 2⋅6−3⋅5 = -3/sqrt(2)
# z1x2 - x1z2 = 3⋅4−1⋅6 = 6/sqrt(2)
# x1y2 - y1x2 = 1⋅5−2⋅4 = -3/sqrt(2)
# 对称无恒矩阵
# c(x1z2 + z1x2) = 1*6+3*4=18/sqrt(2)
# c(x1y2 + y1x2) = 1*5+2*4=13/sqrt(2)
# 2y1y2 - x1x2 - z1z2 =20-4-18=-2/sqrt(6)
# c(y1z2 + z1y2) = 2*6+3*5=27/sqrt(2)
# c(z1z2 - x1x2) = 3*6-1*4=14/sqrt(2)
#
tp = e3nn.o3.FullTensorProduct("1o","1o",["0e"])
print(tp)
a = torch.tensor([1.0,2.0,3.0])
b = torch.tensor([4.0,5.0,6.0])
print(tp(a,b))
tp = e3nn.o3.FullTensorProduct("1o","1o",["2e"])
print(tp)
a = torch.tensor([1.0,2.0,3.0])
b = torch.tensor([4.0,5.0,6.0])
print(tp(a,b))
######################################################
# 如果向量相等,则为球谐函数
# 2zx
# 2xy
# 2y2-x2-y2
# 2yz
# z2-x2

def sph_2(x):
    return e3nn.o3.FullTensorProduct("1o","1o",["2e"])(x,x)

def sph_3(x):
    y = sph_2(x)
    #print(y)
    #print(e3nn.o3.FullTensorProduct("2e","1o")(y,x))
    return e3nn.o3.FullTensorProduct("2e","1o",["3o"])(y,x)

    # print(e3nn.o3.FullTensorProduct("2e","1o"))
    #  1x1o+1x2o+1x3o
print(sph_3(a))
#tensor([ 13.0000,  14.6969,   2.3238, -13.9140,   6.9714,  19.5959,   9.0000])
angels = e3nn.o3.rand_angles()
print(angels)

irreps_in = e3nn.o3.Irreps("1o")
irreps_out = e3nn.o3.Irreps("3o")#7*7
print(irreps_out.D_from_angles(*angels))

x = irreps_in.randn(100,-1)
print(x.shape)
y = sph_3(x)
print(y.shape)

out1 = sph_3(torch.einsum("ij,zj->zi",irreps_in.D_from_angles(*angels),x))
out2 = torch.einsum("ij,zj->zi",irreps_out.D_from_angles(*angels),sph_3(x))

print((out1 - out2).max())

tp = e3nn.o3.FullTensorProduct("3x1o","3x1o",)
print(tp)
tp = e3nn.o3.FullTensorProduct("1o","1o",["1e"])
print(tp)
tp = e3nn.o3.FullyConnectedTensorProduct("1o","1o","1e")# with weight
print(tp)
tp = e3nn.o3.FullyConnectedTensorProduct("1o+1o","1o+1o","1e")# with weight
print(tp)
print(tp.visualize()[0].show())
tp = e3nn.o3.FullyConnectedTensorProduct("2x1o","2x1o","1e")# with weight 更高效
print(tp)
print(tp.visualize()[0].show())
print(tp.weight)


########reduce tensor product
print(e3nn.o3.ReducedTensorProducts("ij=ji",i="1o"))
print(e3nn.o3.ReducedTensorProducts("ij=ji",i="1o").irreps_out)
print(e3nn.o3.ReducedTensorProducts("ij=ji",i="1o")(a,b))
print(e3nn.o3.ReducedTensorProducts("ijkl=jikl=klij",i="1o"))


















