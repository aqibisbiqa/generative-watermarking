import torch
import numpy as np
import random
import copy
import functools
from PIL import Image, ImageOps

import ecc

def process_pixel(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    dims = (0, 2, 3, 1) if images.ndim==4 else (1, 2, 0)
    images = images.cpu().permute(*dims).numpy()
    images = numpy_to_pil(images)
    return images

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

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

# Function to rescale image and add padding if necessary
def prepare_image(image_path, target_height=576, target_width=1024):
    image = Image.open(image_path)

    # Calculate aspect ratio
    aspect_ratio = image.width / image.height
    target_aspect_ratio = target_width / target_height

    # Rescale the image to fit the target width or height while maintaining aspect ratio
    if aspect_ratio > target_aspect_ratio:
        # Image is wider than target aspect ratio
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Image is taller than target aspect ratio
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Add padding to the image to match target dimensions
    padding_color = (0, 0, 0)  # Black padding
    image = ImageOps.pad(image, (target_width, target_height), color=padding_color)

    return image

empirical_success_rates = {
    # "longvideo": 0.85,
    "video": 0.90,
    "latent": 0.90,
    "pixel": 0.70,
}

def mix_samples_using_payload(payload, samp_0, samp_1, err_rate, verbose=False):
    if samp_0.dim() == 5:
        # video -- permute to [batch, channels, frames, h, w]
        samp_0 = samp_0.permute((0, 2, 1, 3, 4))
        samp_1 = samp_1.permute((0, 2, 1, 3, 4))
    m_ecc = ecc.ecc_encode(payload, err_rate)
    m_ecc.resize(samp_0[0, 0].shape, refcheck=False)
    m_ecc = torch.from_numpy(m_ecc).to(samp_0.device)
    res = torch.where(m_ecc == 0, samp_0[:, :], samp_1[:, :])
    if samp_0.dim() == 5:
        # video -- permute back to [batch, frames, channels, h, w]
        samp_0 = samp_0.permute((0, 2, 1, 3, 4))
        samp_1 = samp_1.permute((0, 2, 1, 3, 4))
    return res

def decode_message_from_image_diffs(samp, samp_0, samp_1, verbose=False, debug=True):
    diffs_0 = torch.norm(samp - samp_0, dim=(0, 1))
    diffs_1 = torch.norm(samp - samp_1, dim=(0, 1))

    if debug:
        show = 5
        print(diffs_0[:show, :show])
        print(diffs_1[:show, :show])

    m_dec = torch.where(diffs_0 < diffs_1, 0, 1).cpu().detach().numpy().astype(int)
    if verbose: print("Message AFTER Transmission:", m_dec, sep="\n")
    m_dec = m_dec.flatten()
    return ecc.ecc_recover(m_dec)

def calc_acc(m: np.ndarray, out: np.ndarray, bitwise=True):
    min_len = min(len(m), len(out))
    m = m[:min_len]
    out = out[:min_len]
    
    if bitwise:
        m = np.unpackbits(m)
        out = np.unpackbits(out)
    
    show = 37
    print(m[:show])
    print(out[:show])
    print(sum(out))
    return len(np.where(m==out)[0]) / m.size

def calc_channel_capacity(bytes_encoded, accuracy):
    pass