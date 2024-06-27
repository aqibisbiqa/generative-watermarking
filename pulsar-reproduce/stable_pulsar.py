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
    if device != "cuda":
        raise Exception("use gpu sir")
    verbose = False
    accs = []
    i = 0
    while i < iters:
        try:
            print("#"*75)
            img_sz = pipe.unet.config.sample_size
            # m_sz = (img_sz**2 // 512) * 25
            m_sz = 1000
            m = np.random.randint(256, size=m_sz, dtype=np.uint8)
            k = tuple(int(r) for r in np.random.randint(1000, size=(3,)))
            # k = (10, 11, 12)
            print(f"Iteration {i+1} using keys {k}")
            prompt = "Portrait photo of a man with mustache."
            p = Pulsar(pipe, k, timesteps, prompt=prompt)
            print("ENCODING")
            img = p.encode(m, verbose=verbose)
            print("DECODING")
            out = p.decode(img, verbose=verbose)
        except ValueError:
            print("stupid broadcast error, retrying")
        except ZeroDivisionError:
            print("stupid galois field error, retrying")
        else:
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
use_stable = True

if use_stable:
    from diffusers import StableDiffusionImg2ImgPipeline
    from diffusers import StableDiffusionPipeline
    repos = [
        (StableDiffusionPipeline, "runwayml/stable-diffusion-v1-5"),
        (StableDiffusionPipeline, "stabilityai/stable-diffusion-2-1-base"),
        (StableDiffusionPipeline, "friedrichor/stable-diffusion-2-1-realistic"),
    ]
    pipeline_cls, model_id_or_path = repos[2]
    pipe = pipeline_cls.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)
else:
    from diffusers import DDPMPipeline
    from diffusers import DDIMPipeline
    from diffusers import PNDMPipeline

    repos = [
        (DDIMPipeline, "google/ddpm-church-256"),
        (DDIMPipeline, "google/ddpm-bedroom-256"),
        (DDIMPipeline, "google/ddpm-cat-256"),
        (DDIMPipeline, "google/ddpm-celebahq-256"),

        (DDIMPipeline, "dboshardy/ddim-butterflies-128"),
        (DDIMPipeline, "lukasHoel/ddim-model-128-lego-diffuse-1000"),
    ]
    pipeline_cls, model_id_or_path = repos[5]
    # pipe = pipeline_cls.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = pipeline_cls.from_pretrained(model_id_or_path)
    pipe = pipe.to(device)

# timesteps = 3
timesteps = 50

iters = 10

run_experiment(iters)