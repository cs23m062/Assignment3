SOS_char = "<SOS>"
EOS_char = "<EOS>"
PAD_char = "$"

class AlphabetCreation:
    def __init__(self, name):
        self.name = name
        self.char2index = {SOS_char: 0, EOS_char: 1, PAD_char: 2}  # Initialize character to index mapping with special tokens
        self.char2count = {}  # Initialize character count dictionary
        self.index2char = {0: SOS_char, 1: EOS_char, 2: PAD_char}  # Initialize index to character mapping with special tokens
        self.n_chars = 3  # Start count at 3 to account for SOS, EOS, and PAD tokens

    def addWordtoDict(self, word):
        # Add each character in the word to the dictionaries
        for char in word:
            if char not in self.char2index:
                # If character is not in the dictionary, add it
                self.char2index[char] = self.n_chars
                self.char2count[char] = 1
                self.index2char[self.n_chars] = char
                self.n_chars += 1
            else:
                # If character is already in the dictionary, increment its count
                self.char2count[char] += 1
