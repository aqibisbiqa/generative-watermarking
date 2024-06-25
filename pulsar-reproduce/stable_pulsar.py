import torch
import numpy as np
import random
import copy
import functools
import tqdm

# own files
import utils
from pulsar_methods import Pulsar

print("### importing warnings ###")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print("### defining methods ###")

def run_experiment(iters=1):
    verbose = False
    accs = []
    i = 0
    while i < iters:
        print("#"*75)
        # m_sz = (model.config.sample_size, model.config.sample_size)
        img_sz = pipe.unet.config.sample_size
        bytes_in = True
        if bytes_in:
            m_sz = (img_sz**2 // 512) * 25
            m = np.random.randint(256, size=m_sz, dtype=np.uint8)
        else:
            m_sz = (img_sz**2 // 512) * 200
            m = np.random.randint(2, size=m_sz)
        k = tuple(int(r) for r in np.random.randint(1000, size=(3,)))
        # k = (10, 11, 12)
        print(f"Iteration {i+1} using keys {k}")
        prompt = "Portrait photo of a man with mustache."
        p = Pulsar(pipe, k, timesteps, prompt=prompt)
        print("ENCODING")
        img = p.encode(m, verbose=verbose)
        print("DECODING")
        out = p.decode(img, verbose=verbose)
        print(f"length of m is {len(m)} bytes")
        print(f"length of out is {len(out)} bytes")
        acc = utils.calc_acc(m, out)
        accs.append(acc)
        print(f"Run accuracy {acc}")
        i += 1
    print("#"*75)
    print(f"Final Average Accuracy {np.mean(accs)}")


print("### running experiments ###")

device = "cuda" if torch.cuda.is_available() else "cpu"
use_stable = False

if use_stable:
    from diffusers import StableDiffusionImg2ImgPipeline
    from diffusers import StableDiffusionPipeline
    repos = [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1-base",
        "friedrichor/stable-diffusion-2-1-realistic",
    ]
    model_id_or_path = repos[1]
    pipe = StableDiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)
else:
    from diffusers import DDIMPipeline

    repos = [
        "google/ddpm-church-256",
        "google/ddpm-bedroom-256",
        "google/ddpm-cat-256",
        "google/ddpm-celebahq-256",
        "dboshardy/ddim-butterflies-128",
        "YanivWeiss123/sd-class-poke-64-new",
        "lukasHoel/ddim-model-128-lego-diffuse-1000",
        "krasnova/ddim_afhq_64",
    ]
    model_id_or_path = repos[0]
    pipe = DDIMPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)

# timesteps = 3
timesteps = 50

run_experiment(1)