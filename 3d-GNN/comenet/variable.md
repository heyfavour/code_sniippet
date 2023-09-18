edge_index 													# 以半径为cutoff重新切出的COO矩阵
j,i = edge_index
ves = pos[j] - pos[i] 										# [两体] i->j的向量
dist = vecs.norm(dim=-1) 									# [两体] ij的距离
argmin0 = scatter_min(dist, i, dim_size=num_nodes)  		# [单体] i第一近邻的边索引
n0 = j[argmin0] 											# [单体] i第一近邻的原子索引
argmin1 = scatter_min(dist1, i, dim_size=num_nodes)  		# [单体] i第二近邻的边索引
n1 = j[argmin1] 											# [单体] i第二近邻的原子索引

argmin0_j = scatter_min(dist, j, dim_size=num_nodes)		# [单体] j第一近邻的边索引
n0_j = i[argmin0_j]											# [单体] j第一近邻的原子索引
argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)		# [单体] j第二近邻的边索引
n1_j = i[argmin1_j] 										# [单体] j第二近邻的原子索引


n0 = n0[i]													# [两体] 每条边ji,i的第一近邻原子索引
n1 = n1[i]													# [两体] 每条边ji,i的第二近邻原子索引
	
n0_j = n0_j[j]												# [两体] 每条边ji,j的第一近邻原子索引
n1_j = n1_j[j]												# [两体] 每条边ji,j的第一近邻原子索引


iref = torch.clone(n0)										# [两体] 每条边ji,i的第一近邻原子索引 做参考系 第一近邻-i-第一近邻 替换成	第一近邻-i-第二近邻
idx_iref = argmin0[i]										# [两体] 每条边ji,i第一近邻的边索引   做参考系 第一近邻-i-第一近邻 替换成	第一近邻-i-第二近邻

jref = torch.clone(n0_j)									# [两体] 每条边ji,j的第一近邻原子索引 做参考系 第一近邻-i-第一近邻 替换成	第一近邻-i-第二近邻
idx_jref = argmin0_j[j]										# [两体] 每条边ji,j第一近邻的边索引   做参考系 第一近邻-i-第一近邻 替换成	第一近邻-i-第二近邻


pos_ji     = vecs											# [两体] i->j的向量
pos_in0    = vecs[argmin0][i]								# [两体] 每条边ji,i->第一近邻的向量 为了方便计算
pos_in1    = vecs[argmin1][i]								# [两体] 每条边ji,i->第二近邻的向量 为了方便计算
pos_iref   = vecs[idx_iref] 								# [两体] 每条边ji,i->第一近邻的向量 [第一近邻-i-第一近邻 替换成	第一近邻-i-第二近邻]
pos_jref_j = vecs[idx_jref]									# [两体] 每条边ji,j->第一近邻的向量 [第一近邻-j-第一近邻 替换成	第一近邻-j-第二近邻]

a = ((-pos_ji) * pos_in0).sum(dim=-1)						# v1 点乘 v2 = |v1| |v2| cosθ
b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)				# v1 叉乘 v2 = ∣A×B∣=∣A∣×∣B∣ sin(θ)
theta = torch.atan2(b, a)									# 计算ji和i-n0得弧度