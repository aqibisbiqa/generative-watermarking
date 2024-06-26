from reedmuller.reedmuller import ReedMuller, _dot_product, _vector_reduce, _vector_add

def decode(self, eword):
    """Decode a length-n vector back to its original length-k vector using majority logic."""
    # We want to iterate over each row r of the matrix and determine if a 0 or 1 appears in
    # position r of the original word w using majority logic.

    row = self.k - 1
    word = [-1] * self.k


    for degree in range(self.r, -1, -1):
        # We calculate the entries for the degree. We need the range of rows of the code matrix
        # corresponding to degree r.
        upper_r = self.row_indices_by_degree[degree]
        lower_r = 0 if degree == 0 else self.row_indices_by_degree[degree - 1] + 1

        # Now iterate over these rows to determine the value of word for positions lower_r
        # through upper_r inclusive.
        for pos in range(lower_r, upper_r + 1):
            # We vote for the value of this position based on the vectors in voting_rows.
            votes = [_dot_product(eword, vrow) % 2 for vrow in self.voting_rows[pos]]

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
        s = [_dot_product(word[lower_r:upper_r + 1], column[lower_r:upper_r + 1]) % 2 for column in self.M]
        eword = _vector_reduce(_vector_add(eword, s), 2)

    # We have now decoded.
    return word



inner = ReedMuller(1, 7)

msg = [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0
, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1
, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1
, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0]

print(decode(inner, msg))

msg = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0
, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1
, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1
, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0]

print(decode(inner, msg))

# print(len(msg))
# print(inner.decode(msg))