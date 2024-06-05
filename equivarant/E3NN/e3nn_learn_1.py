import e3nn

Rs_s_orbital = e3nn.o3.Irrep("0e")
Rs_p_orbital = e3nn.o3.Irrep("1o")
Rs_d_orbital = e3nn.o3.Irrep("2e")
Rs_f_orbital = e3nn.o3.Irrep("3o")
# 球谐函数不是不可约表示 just transform same way 不可约表示包含球谐


Rs_vector = e3nn.o3.Irrep("1o")  # 矢量 奇
Rs_pseudovector = e3nn.o3.Irrep("1e")  # 伪矢量 角动量
Rs_doubleray = e3nn.o3.Irrep("2e")  # 双射  偶
Rs_spiral = e3nn.o3.Irrep("2o")  # 螺旋 奇

# 完整路径
tp = e3nn.o3.FullTensorProduct("1x1o", "1x1o")
# FullTensorProduct(1x1o x 1x1o -> 1x0e+1x1e+1x2e | 3 paths | 0 weights)
print(tp)

tp = e3nn.o3.FullyConnectedTensorProduct("1x1o", "1x1o", "1x0e+1x2e")
# FullyConnectedTensorProduct(1x1o x 1x1o -> 1x0e+1x2e | 2 paths | 2 weights)
print(tp)
# 1x0e+1x1e+1x2e ->1x0e+1x2e  选择对应路径 给与参数
