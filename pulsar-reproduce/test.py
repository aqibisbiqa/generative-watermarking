import torch

seed = 0

g_0 = torch.manual_seed(seed)
arr = torch.rand(size=(10, ), generator=g_0)
print(arr)

g_0 = torch.manual_seed(seed)
arr = torch.rand(size=(10, ), generator=g_0)
print(arr)