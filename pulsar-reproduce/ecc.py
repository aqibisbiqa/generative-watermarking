import torch
import numpy as np
import utils
import functools
import math
from coding.grs import GRSCode
from coding.hamming import HammingCode

# @title ECC

def get_code(err_rate):
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
            HammingCode(field_size=2, msg_len=12, pay_len=8)
        )
    elif err_rate < 0.30:
        # outer = GeneralizedReedSolomonCode(GF(256)[:255], 200)
        # inner = BinaryReedMullerCode(1, 7)
        outer, inner = (
            GRSCode(field_size=256, msg_len=512, pay_len=200),
            HammingCode(field_size=2, msg_len=12, pay_len=8)
        )
    else:
        # outer = GeneralizedReedSolomonCode(GF(256)[:255], 100)
        # inner = BinaryReedMullerCode(1, 7)
        outer, inner = (
            GRSCode(field_size=256, msg_len=512, pay_len=200),
            HammingCode(field_size=2, msg_len=12, pay_len=8)
        )
    return outer, inner

# Encodes BYTEarray --> BITarray
def ecc_encode(pay_bytes, rate=None):
    enc = pay_bytes
    outer, inner = get_code(rate)

    # OUTER code ENCODES payload (bytes -> bytes)
    if outer is not None:
        enc = utils.apply_op_to_chunks(
            arr=enc,
            chunk_size=outer.pay_len, 
            op=outer.encode
        )

    # (TODO) INNER code ENCODES payload (bytes -> bytes)
    if inner is not None:
        enc = np.concatenate([inner.encode(byte) for byte in enc])
        enc = np.packbits(enc) # allows for cleaner code

    # convert to bitarray (bytes -> bits)
    enc = np.unpackbits(enc)

    return enc

# Decodes BITarray --> BYTEarray
def ecc_recover(msg_bits, rate=None):
    dec = msg_bits
    outer, inner = get_code(rate)

    # (TODO) INNER code DECODES message (bits -> bits)
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