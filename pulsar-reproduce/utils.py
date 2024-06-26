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

# def get_residual(model, sample, t):
#     with torch.no_grad():
#         residual = model(sample, t).sample
#     return residual

def dbg_print(name, arr):
    debug = True
    if debug:
        print(f"{name:13s}: {str(len(arr)):5s} {arr}")

def calc_acc(m, out):
    min_len = min(len(m), len(out))
    m = m[:min_len]
    out = out[:min_len]
    
    m = np.unpackbits(m)
    out = np.unpackbits(out)

    print(m[:10])
    print(out[:10])
    return len(np.where(m==out)[0]) / m.size

def apply_op_to_chunks(arr: np.ndarray, chunk_size, op):
    l = len(arr)
    arr = np.resize(arr, chunk_size * ((l - 1) // chunk_size + 1))
    out = []
    for i in range(len(arr) // chunk_size):
        start = i * chunk_size
        chunk = arr[start:start+chunk_size].tolist()
        out.extend(op(chunk))
    return np.array(out, dtype=np.uint8)

def bitarray_to_int(bitlist):
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out