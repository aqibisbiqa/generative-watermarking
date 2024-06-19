print("Hello world!")
print([i for i in range(10)])

# help('modules')
import torch
import numpy as np

print(torch.cuda.is_available())
torch.zeros(1).cuda()