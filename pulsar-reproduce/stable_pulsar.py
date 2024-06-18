import torch
import numpy as np
import random
import copy
import functools
import tqdm

# own files
from ecc import *
from rate_estimation import *
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"


print("### importing warnings ###")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print("### defining methods ###")

# @title Pulsar Encode

def encode(m: str, k, verbose=False):

    ######################
    # Offline phase      #
    ######################
    eta = 1
    k_s, k_0, k_1 = k
    torch.manual_seed(k_s)
    
    g_k_s, g_k_0, g_k_1 = torch.Generator(), torch.Generator(), torch.Generator()
    g_k_s.manual_seed(k_s)
    g_k_0.manual_seed(k_0)
    g_k_1.manual_seed(k_1)

    samp = torch.randn(
        1, model.config.in_channels, model.config.sample_size, model.config.sample_size
    )

    for i, t in enumerate(tqdm.tqdm(scheduler.timesteps[:-2])):
        residual = get_residual(model, samp, t)
        samp = scheduler.step(residual, t, samp, generator=g_k_s, eta=eta).prev_sample
        if verbose and ((timesteps-3-i) % 5 == 0):
            display_sample(samp, i + 1)
    # print("OFFLINE SAMPLE:", samp[:, :, :3, :3], sep="\n")

    # rate = estimate_rate(samp, k)
    rate = 0

    ######################
    # Online phase       #
    ######################
    sz = model.config.sample_size
    m_ecc = ecc_encode(m, rate)
    m_ecc = np.reshape(m_ecc, (sz, sz))
    if verbose: print("Message BEFORE Transmission:", m_ecc, sep="\n")

    t = scheduler.timesteps[-2]  # penultimate timestep

    prev_timestep = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    variance = scheduler._get_variance(t, prev_timestep)
    # print(f"t: {t}\nPREVIOUS TIMESTEP: {prev_timestep}\nVARIANCE: {variance}")

    residual = get_residual(model, samp, t)
    torch.manual_seed(k_0)
    samp_0 = scheduler.step(residual, t, samp, generator=g_k_0, eta=eta).prev_sample
    # print("\n\n SAMPLE 0:", samp_0[:, :, :3, :3], sep="\n")

    torch.manual_seed(k_1)
    samp_1 = scheduler.step(residual, t, samp, generator=g_k_1, eta=eta).prev_sample
    # print("\n\n SAMPLE 1:", samp_1[:, :, :3, :3], sep="\n")

    for i in range(sz):
        for j in range(sz):
            match m_ecc[i][j]:
                case 0:
                    samp[:, :, i, j] = samp_0[:, :, i, j]
                case 1:
                    samp[:, :, i, j] = samp_1[:, :, i, j]
    # print("\n\n PEN SAMPLE:", samp[:, :, :3, :3], sep="\n")

    t = scheduler.timesteps[-1]  # last timestep
    residual = get_residual(model, samp, t)
    img = scheduler.step(residual, t, samp).prev_sample
    # print("\n\n FINAL IMAGE:", img[:, :, :3, :3], sep="\n")
    return img

# @title Pulsar Decode
def decode(img, k, verbose=False):

    ######################
    # Offline phase      #
    ######################
    eta = 1
    k_s, k_0, k_1 = k
    torch.manual_seed(k_s)

    g_k_s, g_k_0, g_k_1 = torch.Generator(), torch.Generator(), torch.Generator()
    g_k_s.manual_seed(k_s)
    g_k_0.manual_seed(k_0)
    g_k_1.manual_seed(k_1)

    samp = torch.randn(
        1, model.config.in_channels, model.config.sample_size, model.config.sample_size
    )

    for i, t in enumerate(tqdm.tqdm(scheduler.timesteps[:-2])):
        residual = get_residual(model, samp, t)
        samp = scheduler.step(residual, t, samp, generator=g_k_s, eta=eta).prev_sample
        if verbose and ((i + 1) % 5 == 0):
            display_sample(samp, i + 1)

    # rate = estimate_rate(samp, k)
    rate = 0

    t = scheduler.timesteps[-2]   # penultimate step
    residual = get_residual(model, samp, t)
    samp_0 = scheduler.step(residual, t, samp, generator=g_k_0, eta=eta).prev_sample
    samp_1 = scheduler.step(residual, t, samp, generator=g_k_1, eta=eta).prev_sample

    t = scheduler.timesteps[-1]   # last step
    residual_0 = get_residual(model, samp_0, t)
    residual_1 = get_residual(model, samp_1, t)
    img_0 = scheduler.step(residual_0, t, samp_0, eta=eta).prev_sample
    img_1 = scheduler.step(residual_1, t, samp_1, eta=eta).prev_sample

    ######################
    # Online phase       #
    ######################
    sz = model.config.sample_size
    m_dec = np.zeros((sz, sz), dtype=int)
    for i in range(sz):
        for j in range(sz):
            # pos = sz * i + j
            n_0 = torch.norm(img[:, :, i, j] - img_0[:, :, i, j])
            n_1 = torch.norm(img[:, :, i, j] - img_1[:, :, i, j])
            if n_0 > n_1:
                m_dec[i][j] = 1
    if verbose: print("Message AFTER Transmission:", m_dec, sep="\n")
    m_dec = m_dec.flatten()
    m = ecc_recover(m_dec, rate)
    return m

def run_experiment(iters=1):
    accs = []
    for i in range(iters):
        print("#"*75)
        # m_sz = (model.config.sample_size, model.config.sample_size)
        m_sz = 25600
        m = np.random.randint(2, size=m_sz)
        k = tuple(int(r) for r in np.random.randint(1000, size=(3,)))
        print(f"Iteration {i+1} using keys {k}")
        print("ENCODING")
        img = encode(m, k)
        print("DECODING")
        out = decode(img, k)
        acc = calc_acc(m, out)
        accs.append(acc)
        print(f"Run accuracy {acc}")
    print("#"*75)
    print(f"Final Average Accuracy {np.mean(accs)}")

print("### importing model+scheduler ###")

from diffusers import StableDiffusionImg2ImgPipeline

model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

# timesteps = 3
timesteps = 50


print("### running experiments ###")

run_experiment(10)