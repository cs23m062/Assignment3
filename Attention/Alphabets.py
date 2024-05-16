
SOS_char = "<SOS>"
EOS_char = "<EOS>"
PAD_char = "$"

class AlphabetCreation:
    def __init__(self, name):
        self.name = name
        self.char2index = {SOS_char: 0, EOS_char: 1, PAD_char: 2}
        self.char2count = {}
        self.index2char = {0: SOS_char, 1: EOS_char, 2: PAD_char}
        self.n_chars = 3  # Count SOS, EOS, PAD 

    def addWordtoDict(self, word):
        for char in word:
            if char not in self.char2index:
                self.char2index[char] = self.n_chars
                self.char2count[char] = 1
                self.index2char[self.n_chars] = char
                self.n_chars += 1
            else:
                self.char2count[char] += 1