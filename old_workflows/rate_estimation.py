import torch
import numpy as np
from ecc import get_code

# @title Rate Estimation
def estimate_rate(pulsar, samp):
    if True: return 0
    # l, u are chosen based on Jois et al.
    l, u = 1, 100
    errs = calc_errors(pulsar, samp, l)     # `errs` is a list of pairs (pt, err_value)
    regions = bucketize(errs, u)            # each `region` has form {avg_err: __, pts: <set>}
    rate = []
    for region in regions:
        regionErr = region["avg_err"]
        params = get_code(regionErr)
        rate.append()
    return 1

@torch.no_grad()
def calc_errors(pulsar, samp, l):
    g_k_s, g_k_0, g_k_1 = tuple([torch.manual_seed(k) for k in pulsar.keys])
    model = pulsar.pipe.unet
    scheduler = pulsar.pipe.scheduler

    sz = model.config.sample_size
    errs = []

    for _ in range(l):
        m = torch.rand(sz**2, generator=g_k_s)

        t = 1  # penultimate timestep
        residual = get_residual(model, samp, t)
        samp_0 = scheduler.step(residual, t, samp, g_k_0).prev_sample
        samp_1 = scheduler.step(residual, t, samp, g_k_1).prev_sample
        for i in range(sz):
            for j in range(sz):
                pos = sz * i + j
                match m[pos]:
                    case 0:
                        samp[:, :, i, j] = samp_0[:, :, i, j]
                    case 1:
                        samp[:, :, i, j] = samp_1[:, :, i, j]

        t = 0  # last timestep
        residual = get_residual(model, samp, t)
        residual_0 = get_residual(model, samp_0, t)
        residual_1 = get_residual(model, samp_1, t)
        img = scheduler.step(residual, t, samp).prev_sample
        img_0 = scheduler.step(residual_0, t, samp_0).prev_sample
        img_1 = scheduler.step(residual_1, t, samp_1).prev_sample

        m_dec = [0]*(sz**2)
        for i in range(sz):
            for j in range(sz):
                pos = sz * i + j
                n_0 = torch.norm(img[:, :, i, j] - img_0[:, :, i, j])
                n_1 = torch.norm(img[:, :, i, j] - img_1[:, :, i, j])
                if n_0 > n_1:
                    m_dec[pos] = 1
        errs.append(torch.abs(m - m_dec))
    errs = torch.cat(errs, dim=0)
    return torch.mean(errs, dim=0)

def bucketize(errs, u):
    pass