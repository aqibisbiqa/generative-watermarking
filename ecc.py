import torch
import numpy as np
import utils
import functools
import math
from coding.grs import GRSCode
from coding.hamming import HammingCode
from coding.brm import BRMCode

# @title ECC

def get_code(model_type):
    # HammingCode(field_size=2, msg_len=12, pay_len=8),
    # GRSCode(field_size=256, msg_len=512, pay_len=200),
    if model_type == "pixel":
        print("using pixel")
        outer, inner = (
            None,
            None,
        )
    elif model_type == "latent":
        print("using latent")
        outer, inner = (
            BRMCode(r=1, m=5),
            None,
        )
    elif model_type == "video":
        print("using video")
        outer, inner = (
            BRMCode(r=1, m=7),
            None,
        )
        # outer, inner = (
        #     GRSCode(field_size=256, msg_len=512, pay_len=200),
        #     BRMCode(r=1, m=5),
        # )
    elif model_type == "longvideo":
        print("using longvideo")
        outer, inner = (
            None,
            None,
        )
    else:
        raise ValueError(f"model_type cannot be {model_type}")
    return outer, inner

# Encodes BYTEarray --> BITarray
def ecc_encode(pay_bytes, model_type):
    
    # get code
    outer, inner = get_code(model_type)
    
    # (bytes -> bits)
    enc = np.unpackbits(pay_bytes)

    # (bits -> bits) OUTER code ENCODES payload
    if outer is not None:
        enc = outer.encode(enc)

    # (bits -> bits) INNER code ENCODES payload
    if inner is not None:
        enc = inner.encode(enc)
    
    return enc

# Decodes BITarray --> BYTEarray
def ecc_decode(msg_bits, model_type):
    
    # get code
    outer, inner = get_code(model_type)
    
    # (bits -> bits)
    dec = msg_bits

    # (bits -> bits) INNER code DECODES message
    if inner is not None:
        dec = inner.decode(dec)

    # (bits -> bits) OUTER code DECODES message
    if outer is not None:
        dec = outer.decode(dec)
    
    # (bits -> bytes) ensure output is array of bytes
    dec = np.packbits(dec)

    return dec