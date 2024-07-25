import torch
import numpy as np
import utils
import functools
import math
from coding.grs import GRSCode
from coding.hamming import HammingCode
from coding.brm import BRMCode

# @title ECC

def get_code(err_rate):
    if err_rate < 0.05:
        print("using nothing")
        outer, inner = (
            None,
            None,
        )
    elif err_rate < 0.10:
        print("using hamming")
        outer, inner = (
            # HammingCode(field_size=2, msg_len=12, pay_len=8),
            BRMCode(r=1, m=5),
            None,
        )
    elif err_rate < 0.15:
        print("using BRM")
        outer, inner = (
            BRMCode(r=1, m=7),
            None,
        )
    else:
        print("using grs + brm")
        outer, inner = (
            GRSCode(field_size=256, msg_len=512, pay_len=200),
            BRMCode(r=1, m=5),
        )
    return outer, inner

# Encodes BYTEarray --> BITarray
def ecc_encode(pay_bytes, err_rate=0):
    
    # get code
    outer, inner = get_code(err_rate)
    
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
def ecc_recover(msg_bits, err_rate=0):
    
    # get code
    outer, inner = get_code(err_rate)
    
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