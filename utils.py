import torch
import numpy as np
from PIL import Image, ImageOps
from einops import rearrange

import ecc

#####################
### Coding Related ##
#####################

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

########################
### Image Processing ###
########################

def process_pixel(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    dims = (0, 2, 3, 1) if images.ndim==4 else (1, 2, 0)
    images = images.cpu().permute(*dims).numpy()
    images = numpy_to_pil(images)
    return images

    # Adapted from function in diffusers library
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

    # Prep image for SVD input (rescale image and add padding if necessary)
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

#################################
### PSyDUCK Mixing + Unmixing ###
#################################

def mix_samples_using_payload(payload, samp_0, samp_1, model_type):
    if samp_0.dim() == 5:
        # video
        samp_0 = rearrange(samp_0, 'b f c h w -> b c f h w').contiguous()
        samp_1 = rearrange(samp_1, 'b f c h w -> b c f h w').contiguous()
    
    m_ecc = ecc.ecc_encode(payload, model_type)
    m_ecc.resize(samp_0[0, 0].shape, refcheck=False)
    m_ecc = torch.from_numpy(m_ecc).to(samp_0.device)
    res = torch.where(m_ecc == 0, samp_0[:, :], samp_1[:, :])
    if samp_0.dim() == 5:
        # video
        res = rearrange(res, 'b c f h w -> b f c h w').contiguous()
    return res

def decode_message_from_image_diffs(samp, samp_0, samp_1, model_type):
    diffs_0 = torch.norm(samp - samp_0, dim=(0, 1))
    diffs_1 = torch.norm(samp - samp_1, dim=(0, 1))

    m_dec = torch.where(diffs_0 < diffs_1, 0, 1).cpu().detach().numpy().astype(int)
    m_dec = m_dec.flatten()
    
    # debugging
    if model_type in ["latent"]:
        show = 5
        print(diffs_0[:show, :show])
        print(diffs_1[:show, :show])
        print("Message AFTER Transmission:", m_dec, sep="\n")

    return ecc.ecc_decode(m_dec, model_type)

########################
### Experiment Utils ###
########################

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