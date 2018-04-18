import constants as C

class Vocabulary():
    def __init__(self, max_vocab_size):
        self.reserved = set([C.PAD_TOKEN, C.EOS_TOKEN, C.SOS_TOKEN,\
                              C.UNK_TOKEN])
        self.res_token2index = {
                C.PAD_TOKEN : C.PAD_INDEX,
                C.UNK_TOKEN : C.UNK_INDEX,
                C.SOS_TOKEN : C.SOS_INDEX,
                C.EOS_TOKEN : C.EOS_INDEX
        }

        self.token2index = self.res_token2index
        self.index2token = dict([(self.res_token2index[tok], tok) for tok in self.res_token2index])
        self.token2freq = {}  # would never have reserved tokens, would have all the tokens
        self.token2count = {}

        self.num_tokens = 0  # non reserved tokens
        self.num_reserved = C.RESERVED_IDS

        self.is_sorted = False

        if max_vocab_size != -1:
            self.size = max_vocab_size
        else:
            self.size = 10**9

    def sort_vocabulary(self):
        """
        Sorts the vocabulary in descending order based on frequency
        token2index and index2token has only top self.size words
        """
        if self.is_sorted:
            return

        trimmed_token2freq = sorted(self.token2freq.items(), \
                key=lambda kv:kv[1] , reverse=True)[0:self.size]

        for index, (token, freq) in enumerate(trimmed_token2freq):
            self.token2index[token] = index + self.num_reserved
            self.index2token[index + self.num_reserved] = token

        if self.num_tokens > self.size:
            self.num_tokens = self.size

        self.is_sorted = True

    def get_index(self, token):
        self.sort_vocabulary()
        return self.token2index[token]

    def get_token(self, index):
        self.sort_vocabulary()
        return self.index2token[index]

    def get_vocab_size(self):
        self.sort_vocabulary()
        return self.num_tokens + self.num_reserved

    def add_token(self, token):
        if token not in self.reserved:
            if token not in self.token2freq:
                self.token2freq[token] = 1
                self.num_tokens += 1
            else:
                self.token2freq[token] += 1
        self.is_sorted = False

    def add_sequence(self, sequence):
        for tok in sequence:
            self.add_token(tok)

    def indices_from_token_list(self, tok_list):
        """
        Maps a list of words to token IDs
        if the word is rare/unknown, map to <UNK>.
        """
        self.sort_vocabulary()
        indices = [C.SOS_INDEX]
        for tok in tok_list:
            if tok in self.token2index:
                indices.append(self.token2index[tok])
            else:
                indices.append(C.UNK_INDEX)
        indices.append(C.EOS_INDEX)
        return indices

    def token_list_from_indices(self, indices):
        """
        Get a list of tokens from a list of token IDs.
        """
        self.sort_vocabulary()
        seq = [self.index2token[idx] for idx in indices]
        return seq
