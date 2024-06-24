import torch
import numpy as np
import utils
from coding.grs import GRSCode
# from coding.hamming import *

# @title ECC

def get_code(err_rate):
    field_size, msg_len, pay_len = 256, 512, 200
    outer, inner = GRSCode(field_size, msg_len, pay_len, 1, 3), None
    # outer, inner = None, None
    """                     # comment this line to uncomment code
    if err_rate < 0.05:
        outer = GeneralizedReedSolomonCode(GF(256)[:255], 200)
        inner = HammingCode(GF(2), 3)
    elif err_rate < 0.10:
        outer = GeneralizedReedSolomonCode(GF(256)[:255], 100)
        inner = HammingCode(GF(2), 3)
    elif err_rate < 0.15:
        outer = GeneralizedReedSolomonCode(GF(256)[:255], 220)
        inner = BinaryReedMullerCode(1, 5)
    elif err_rate < 0.30:
        outer = GeneralizedReedSolomonCode(GF(256)[:255], 200)
        inner = BinaryReedMullerCode(1, 7)
    else:
        outer = GeneralizedReedSolomonCode(GF(256)[:255], 100)
        inner = BinaryReedMullerCode(1, 7)
    #"""
    return outer, inner

# Turns bitstring into encoded bytestring
def ecc_encode(pay_bytes, rate=None):
    outer, inner = get_code(rate)

    # (TODO) encode payload with INNER code
    

    # encode payload with OUTER code
    msg_bytes = utils.apply_op_to_chunks(pay_bytes, outer.payload_length, outer.encode)
    
    # convert bytes -> bits
    msg_bits = np.unpackbits(np.array(msg_bytes, dtype=np.uint8))

    return msg_bits

def ecc_recover(msg_bits, rate=None):
    outer, inner = get_code(rate)

    # convert bits -> bytes
    msg_bytes = np.packbits(msg_bits)

    # decode message with OUTER code
    pay_bytes = utils.apply_op_to_chunks(msg_bytes, outer.message_length, outer.decode)
    
    # (TODO) decode message with INNER code


    # ensure output is array of bytes
    pay_bytes = np.array(pay_bytes, dtype=np.uint8)

    return pay_bytes