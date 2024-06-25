import hamming_codec
import numpy as np
import utils

class HammingCode():
    def __init__(self, field_size, msg_len, pay_len, order=3):
        self.field_size = field_size
        self.msg_len = msg_len
        self.pay_len = pay_len
        self.order = order

    def encode(self, payload):
        if type(payload) in [np.ndarray, list]:
            assert len(payload) == self.pay_len
            payload = utils.bitarray_to_int(payload)
        if type(payload) is str:
            assert len(payload) == self.pay_len
            payload = int(payload,2)
        enc = hamming_codec.encode(payload, self.pay_len)
        return np.array(list(enc), dtype=np.uint8)

    def decode(self, message):
        if type(message) in [np.ndarray, list]:
            assert len(message) == self.msg_len
            message = utils.bitarray_to_int(message)
        if type(message) is str:
            assert len(message) == self.msg_len
            message = int(message,2)
        dec = hamming_codec.decode(message, self.msg_len)
        return np.array(list(dec), dtype=np.uint8)
    