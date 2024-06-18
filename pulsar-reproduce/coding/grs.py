# import GRS code
"""
!git clone https://github.com/raeudigerRaeffi/generalizedReedSolomon.git
!python3 -m pip install galois
!pip install pathos
"""
from . import generalizedReedSolomon

class GRSCode():
    def __init__(self, field_size, message_length, payload_length,
                 symbol_size, p_factor, debug=False):
        self.field_size = field_size
        self.message_length = message_length
        self.payload_length = payload_length
        self.symbol_size = symbol_size
        self.p_factor = p_factor
        self.debug = debug
        self.grs = generalizedReedSolomon.Generalized_Reed_Solomon(
            field_size=field_size,
            message_length=message_length,
            payload_length=payload_length,
            symbol_size=symbol_size,
            p_factor=p_factor,
            debug=debug
        )

    def encode(self, payload):
        assert len(payload) == self.payload_length
        return self.grs.encode(payload)

    def decode(self, message):
        assert len(message) == self.message_length
        return self.grs.decode(message)