
import torch
import numpy as np

from utils import mix_samples_using_payload

payload = []
rate = None
shape = ()
samp_0 = torch.Tensor.new_zeros(shape)
samp_1 = torch.Tensor.new_ones(shape)

mix_samples_using_payload(payload, rate, samp_0, samp_1, "cpu", verbose=False)