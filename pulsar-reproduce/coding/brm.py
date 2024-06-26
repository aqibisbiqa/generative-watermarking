from reedmuller.reedmuller import ReedMuller, _dot_product, _vector_reduce, _vector_add
import numpy as np
from typing import Union

class BRMCode():
    def __init__(self, r, m):
        self.brm = ReedMuller(r, m)
        self.pay_len = self.brm.message_length()
        self.msg_len = self.brm.block_length()

    def encode(self, payload: Union[list, np.ndarray, np.uint8]):
        if type(payload) is np.uint8:
            payload = np.unpackbits([payload])
        if type(payload) is list:
            payload = np.array(payload)
        if len(payload) < self.pay_len:
            payload = np.pad(payload, (0, self.pay_len - len(payload)), "constant", constant_values=(0))
        enc = self.brm.encode(payload)
        return np.array(enc, dtype=np.uint8)

    def decode(self, message: Union[list, np.ndarray]):
        if type(message) is list:
            message = np.array(message)
        if len(message) < self.msg_len:
            message = np.pad(message, (0, self.msg_len - len(message)), "constant", constant_values=(0))
        dec = self.brm_decode(message)
        return np.array(dec, dtype=np.uint8)
    
    def brm_decode(self, eword):
        """Decode a length-n vector back to its original length-k vector using majority logic."""
        # We want to iterate over each row r of the matrix and determine if a 0 or 1 appears in
        # position r of the original word w using majority logic.

        brm = self.brm

        row = brm.k - 1
        word = [-1] * brm.k


        for degree in range(brm.r, -1, -1):
            # We calculate the entries for the degree. We need the range of rows of the code matrix
            # corresponding to degree r.
            upper_r = brm.row_indices_by_degree[degree]
            lower_r = 0 if degree == 0 else brm.row_indices_by_degree[degree - 1] + 1

            # Now iterate over these rows to determine the value of word for positions lower_r
            # through upper_r inclusive.
            for pos in range(lower_r, upper_r + 1):
                # We vote for the value of this position based on the vectors in voting_rows.
                votes = [_dot_product(eword, vrow) % 2 for vrow in brm.voting_rows[pos]]

                # If there is a tie, there is nothing we can do.
                if votes.count(0) == votes.count(1):
                    word[pos] = 0
                # Otherwise, we set the position to the winner.
                else:
                    word[pos] = 0 if votes.count(0) > votes.count(1) else 1

            # Now we need to modify the word. We want to calculate the product of what we just
            # voted on with the rows of the matrix.
            # QUESTION: do we JUST do this with what we've calculated (word[lower_r] to word[upper_r]),
            #           or do we do it with word[lower_r] to word[k-1]?
            s = [_dot_product(word[lower_r:upper_r + 1], column[lower_r:upper_r + 1]) % 2 for column in brm.M]
            eword = _vector_reduce(_vector_add(eword, s), 2)

        # We have now decoded.
        return word