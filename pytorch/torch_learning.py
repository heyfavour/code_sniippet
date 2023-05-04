import torch  # C语言核心
import numpy as np

data = [[1, 2], [3, 4]]
torch_data = torch.Tensor(data)
print(torch_data)

np_array = np.array(data)
torch_np = torch.from_numpy(np_array)
print(torch_np)
