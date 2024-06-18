import torch
import numpy as np
import random
import copy
import functools
import PIL.Image

def draw_rand(model):
    return torch.randn(
        1, model.config.in_channels, model.config.sample_size, model.config.sample_size
    )


def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Image at step {i}")
    display(image_pil)

def get_residual(model, sample, t):
    with torch.no_grad():
        residual = model(sample, t).sample
    return residual

def dbg_print(name, arr):
    debug = False
    if debug:
        print(f"{name:13s}: {str(len(arr)):5s} {arr}")

def calc_acc(m, out):
    return len(np.where(m==out)[0]) / m.size
