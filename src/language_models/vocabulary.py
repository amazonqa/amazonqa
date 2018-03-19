class Vocabulary(object):
    """
    A Vocabulary stores a set of words in the corpus mapped to unique integer IDs.

    In addition to the words in the actual language, a Vocabulary includes three
    reserved tokens (and IDs) for the start-of-sequence and end-of-sequence
    markers, and for a special 'UNK' marker used to handle rare/unknown words.

    The Vocabulary is sorted in descending order based on frequency. If the
    number of words seen is greater than the maximum size of the Vocabulary,
    the remaining least-frequent words are ignored.

    Args: size(int): maximum number of words allowed in this vocabulary
    """
    def __init__(self, max_vocab_size):
        self.PAD_token_name = "<PAD>"
        self.UNK_token_name = "<UNK>"
        self.SOS_token_name = "<SOS>"
        self.EOS_token_name = "<EOS>"
        self.PAD_token_id = 0
        self.UNK_token_id = 1
        self.SOS_token_id = 2
        self.EOS_token_id = 3

        self._reserved = set([self.PAD_token_name, self.UNK_token_name, \
                self.SOS_token_name, self.EOS_token_name])
        self._reserved_token_id = [
                (self.PAD_token_name, self.PAD_token_id),
                (self.UNK_token_name, self.UNK_token_id),
                (self.SOS_token_name, self.SOS_token_id),
                (self.EOS_token_name, self.EOS_token_id)
        ]

        self._token2index = dict([(tok, idx) for tok, idx in self._reserved_token_id])
        self._index2token = dict([(idx, tok) for tok, idx in self._reserved_token_id])

        self._token2count = {}

        self._num_tokens = 0
        self._num_reserved = 4
        self.sorted = False
        self.size = max_vocab_size

    def trim(self):
        """
        Sorts the vocabulary in descending order based on frequency
        """
        sorted_vocab_count = sorted(self._token2count.items(), \
                key=lambda x: x[1], reverse=True)[:self.size]
        self._token2index = dict( [ (w, self._num_reserved + idx) \
                for idx, (w, _) in enumerate(sorted_vocab_count) ] )
        self._index2token = dict( [ (idx, w) \
                for w, idx in self._token2index.items() ])

        for tok, idx in self._reserved_token_id:
            self._token2index[tok] = idx
            self._index2token[idx] = tok

        if self._num_tokens > self.size:
            self._num_tokens = self.size

        self.sorted = True

    def sort_vocabulary(self):
        """
        Sorts the vocabulary (if it is not already sorted).
        """
        if not self.sorted:
            self.trim()


    def get_index(self, token):
        """
        Returns: int: ID of the given token.
        """
        self.sort_vocabulary()
        return self._token2index[token]


    def get_token(self, index):
        """
        Returns: str: token with ID equal to the given index.
        """
        self.sort_vocabulary()
        return self._index2token[index]


    def get_vocab_size(self):
        """
        Returns: int: maximum number of words in the vocabulary.
        """
        self.sort_vocabulary()
        return self._num_tokens + self._num_reserved


    def add_token(self, token):
        """
        Adds an occurrence of a token to the vocabulary,
        incrementing its observed frequency if the word already exists.
        Args: token (int): word to add
        """
        if token in self._reserved:
            return
        if token not in self._token2count:
            self._token2count[token] = 1
            self._num_tokens += 1
        else:
            self._token2count[token] += 1
        self.sorted = False

    def add_sequence(self, sequence):
        """
        Adds a sequence of words to the vocabulary.
        Args: sequence(list(str)): list of words, e.g. representing a sentence.
        """
        for tok in sequence:
            self.add_token(tok)

    def indices_from_sequence(self, sequence):
        """
        Maps a list of words to their token IDs, or else <UNK>
        if the word is rare/unknown.
        Args: sequence (list(str)): list of words to map
        Returns: list(int): list of mapped IDs
        """
        self.sort_vocabulary()
        return [self._token2index[tok]
                if tok in self._token2index
                else self.UNK_token_id
                for tok in sequence]

    def sequence_from_indices(self, indices):
        """
        Recover a sentence from a list of token IDs.
        Args: indices (list(int)): list of token IDs.
        Returns: list(str): recovered sentence, represented as a list of words
        """
        seq = [self._index2token[idx] for idx in indices]
        return seq


    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        self.sort_vocabulary()
        other.sort_vocabulary()

        if self._token2count == other._token2count \
                and self._token2index == other._token2index \
                and self._index2token == other._index2token:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._token2index)
