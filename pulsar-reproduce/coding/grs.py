from . import generalizedReedSolomon
import numpy as np
from typing import Union

class GRSCode():
    def __init__(self, field_size, msg_len, pay_len,
                 symbol_size=1, p_factor=3, debug=False):
        self.field_size = field_size
        self.msg_len = msg_len
        self.pay_len = pay_len
        self.symbol_size = symbol_size
        self.p_factor = p_factor
        self.debug = debug
        self.grs = generalizedReedSolomon.Generalized_Reed_Solomon(
            field_size=field_size,
            message_length=msg_len,
            payload_length=pay_len,
            symbol_size=symbol_size,
            p_factor=p_factor,
            debug=debug
        )

    def encode(self, payload):
        assert len(payload) == self.pay_len
        enc = self.grs.encode(payload)
        return np.array(enc, dtype=np.uint8)

    def decode(self, message):
        assert len(message) == self.msg_len
        dec = self.grs.decode(message)
        return np.array(dec, dtype=np.uint8)