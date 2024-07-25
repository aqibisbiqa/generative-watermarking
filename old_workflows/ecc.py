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
    err_rate = 0.12
    # err_rate = 100
    if err_rate < 0.05:
        # outer = GeneralizedReedSolomonCode(GF(256)[:255], 200)
        # inner = HammingCode(GF(2), 3)
        outer, inner = (
            GRSCode(field_size=256, msg_len=512, pay_len=200),
            HammingCode(field_size=2, msg_len=12, pay_len=8)
        )
    elif err_rate < 0.10:
        # outer = GeneralizedReedSolomonCode(GF(256)[:255], 100)
        # inner = HammingCode(GF(2), 3)
        outer, inner = (
            GRSCode(field_size=256, msg_len=512, pay_len=200),
            HammingCode(field_size=2, msg_len=12, pay_len=8)
        )
    elif err_rate < 0.15:
        # outer = GeneralizedReedSolomonCode(GF(256)[:255], 220)
        # inner = BinaryReedMullerCode(1, 5)
        outer, inner = (
            GRSCode(field_size=256, msg_len=512, pay_len=200),
            BRMCode(r=1, m=5)
        )
    elif err_rate < 0.30:
        # outer = GeneralizedReedSolomonCode(GF(256)[:255], 200)
        # inner = BinaryReedMullerCode(1, 7)
        outer, inner = (
            GRSCode(field_size=256, msg_len=512, pay_len=200),
            BRMCode(r=1, m=7)
        )
    else:
        # outer = GeneralizedReedSolomonCode(GF(256)[:255], 100)
        # inner = BinaryReedMullerCode(1, 7)
        outer, inner = (
            GRSCode(field_size=256, msg_len=512, pay_len=200),
            BRMCode(r=1, m=7)
        )
    # res = outer, inner
    # res = None, inner,
    # res = outer, None,
    res = None, None
    # return outer, inner
    return res

# Encodes BYTEarray --> BITarray
def ecc_encode(pay_bytes, rate=None):
    
    # get code
    outer, inner = get_code(rate)
    
    # (bytes -> bits)
    enc = np.unpackbits(pay_bytes)

    # (bits -> bits) OUTER code ENCODES payload
    if outer is not None:
        enc = outer.encode(enc)

    # (bits -> bits) INNER code ENCODES payload
    if inner is not None:
        enc = inner.encode(enc)
    
    if True: return enc
    
    enc = pay_bytes
    outer, inner = get_code(rate)

    # OUTER code ENCODES payload (bytes -> bytes)
    if outer is not None:
        enc = utils.apply_op_to_chunks(
            arr=enc,
            chunk_size=outer.pay_len, 
            op=outer.encode
        )

    # INNER code ENCODES payload (bytes -> bytes)
    if inner is not None:
        enc = np.concatenate([inner.encode(byte) for byte in enc])
        enc = np.packbits(enc) # allows for cleaner code

    # convert to bitarray (bytes -> bits)
    enc = np.unpackbits(enc)

    return enc

# Decodes BITarray --> BYTEarray
def ecc_recover(msg_bits, rate=None):
    
    # get code
    outer, inner = get_code(rate)
    
    # (bits -> bits)
    dec = msg_bits

    # (bits -> bits) INNER code DECODES message
    if inner is not None:
        dec = inner.decode(dec)

    # (bits -> bits) OUTER code DECODES message
    if outer is not None:
        dec = outer.decode(dec)
    
    # ensure output is array of bytes (bits -> bytes)
    dec = np.packbits(dec)

    if True: return dec

    dec = msg_bits
    outer, inner = get_code(rate)

    # INNER code DECODES message (bits -> bits)
    if inner is not None:
        dec = utils.apply_op_to_chunks(
            arr=dec, 
            chunk_size=inner.msg_len, 
            op=inner.decode
        )

    # OUTER code DECODES message (bits -> bits)
    if outer is not None:
        dec = np.packbits(dec)
        dec = utils.apply_op_to_chunks(
            arr=dec, 
            chunk_size=outer.msg_len, 
            op=outer.decode
        )
        dec = np.unpackbits(dec) # allows for cleaner code
    
    # ensure output is array of bytes (bits -> bytes)
    dec = np.packbits(dec)

    return dec