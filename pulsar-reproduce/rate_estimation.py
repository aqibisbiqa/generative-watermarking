import torch
import numpy as np

# @title Rate Estimation
def estimate_rate(s, k):
    # l and u are chosen based on Jois et al.
    # l = 1
    # u = 100
    # err = calc_errors(s, l, k)
    # regions = bucketize(err, u)

    # TODO
    return 1

def calc_errors(samp, l, k):
    k_s, k_0, k_1 = k
    g_k_s, g_k_0, g_k_1 = torch.Generator(), torch.Generator(), torch.Generator()
    g_k_s.manual_seed(k_s)
    g_k_0.manual_seed(k_0)
    g_k_1.manual_seed(k_1)
    sz = model.config.sample_size

    errs = []
    for g in range(0, l):
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

def bucketize(err, u):
    pass