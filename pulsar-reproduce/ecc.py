import torch
import numpy as np
from utils import *
from coding.grs import *
from coding.hamming import *

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
def ecc_encode(payload_bits, rate=None):
    dbg_print("payload_bits", payload_bits)
    # convert bits -> bytes
    num_bits = len(payload_bits)
    full_payload = []
    for i in range(num_bits)[::8]:
        missing = -min(i+8, num_bits) % 8
        num = 0
        for b in payload_bits[i:i+8-missing]:
            num <<= 1
            num |= b
        num <<= missing
        full_payload.append(num)
    dbg_print("full_payload", full_payload)

    # encode
    outer, inner = get_code(rate)
    message_bytes = []
        # first inner (TODO)
        # then outer
    pay_len = outer.payload_length
    msg_len = outer.message_length
    for i in range(len(full_payload))[::pay_len]:
        missing = -min(i+pay_len, len(full_payload)) % pay_len
        curr_pay_ext = full_payload[i:i+pay_len-missing] + [0]*missing
        dbg_print("curr_pay_ext", curr_pay_ext)
        msg_bytes = outer.encode(curr_pay_ext)
        message_bytes.extend(msg_bytes + [0]*(msg_len-len(msg_bytes)))
    dbg_print("message_bytes", message_bytes)
    
    # convert bytes -> bits
    message = []
    for byte in message_bytes:
        message.extend(
            [(byte >> i) % 2 for i in range(7,-1,-1)]
        )
    dbg_print("message", message)
    return message

def ecc_recover(message_bits, rate=None):
    dbg_print("message_bits", message_bits)
    # convert bits -> bytes
    num_bits = len(message_bits)
    message_bytes = []
    for i in range(num_bits)[::8]:
        missing = -min(i+8, num_bits) % 8
        num = 0
        for b in message_bits[i:i+8-missing]:
            num <<= 1
            num |= b
        num <<= missing
        message_bytes.append(num)
    dbg_print("message_bytes", message_bytes)

    # encode
    outer, inner = get_code(rate)
    payload_bytes = []    
        # first outer
    msg_len = outer.message_length
    for i in range(len(message_bytes))[::msg_len]:
        missing = -min(i+msg_len, len(message_bytes)) % msg_len
        msg_bytes = message_bytes[i:i+msg_len-missing] + [0]*missing
        pay_bytes = outer.decode(msg_bytes)
        dbg_print("pay_bytes", pay_bytes)
        payload_bytes.extend(pay_bytes)
    dbg_print("payload_bytes", payload_bytes)
        # then inner (TODO)
    
    # convert bytes -> bits
    payload = []
    for byte in payload_bytes:
        payload_bits = [(byte >> i) % 2 for i in range(7,-1,-1)]
        payload.extend(payload_bits)
    dbg_print("payload", payload)
    return payload