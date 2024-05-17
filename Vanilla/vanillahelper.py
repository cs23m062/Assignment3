from torch import optim
from torch.nn.utils.rnn import pad_sequence
import torch

# Define special characters for Start of Sequence, End of Sequence, and Padding
SOS_char = "<SOS>"
EOS_char = "<EOS>"
PAD_char = ""

# Class to create and manage an alphabet for a language
class AlphabetCreation:
    def __init__(self, name):
        self.name = name
        self.char2index = {SOS_char: 0, EOS_char: 1, PAD_char: 2}
        self.char2count = {}
        self.index2char = {0: SOS_char, 1: EOS_char, 2: PAD_char}
        self.n_chars = 3  # Initial count including SOS, EOS, PAD

    def addWordtoDict(self, word):
        for char in word:
            if char not in self.char2index:
                self.char2index[char] = self.n_chars
                self.char2count[char] = 1
                self.index2char[self.n_chars] = char
                self.n_chars += 1
            else:
                self.char2count[char] += 1

# Helper class with static methods for various tasks
class Helper:
    @staticmethod
    def LanguageVocabulary(data, input_lang, output_lang):
        # Create vocabularies for input and output languages
        input_vocab = AlphabetCreation(input_lang)
        output_vocab = AlphabetCreation(output_lang)
        
        for pair in data:
            input_vocab.addWordtoDict(pair[0])
            output_vocab.addWordtoDict(pair[1])
        
        return input_vocab, output_vocab
    
    @staticmethod
    def WordtoTensor(word, vocab, sos=False, eos=False):
        char_list = []
        if sos:
            char_list.append(vocab.char2index[SOS_char])
        for char in word:
            char_list.append(vocab.char2index[char])
        if eos:
            char_list.append(vocab.char2index[EOS_char])
        char_tensor = torch.tensor(char_list, dtype=torch.long)
        return char_tensor

    @staticmethod
    def DataProcessing(data, vocab, sos=False, eos=False):
        tensor_list = []
        for word in data:
            word_tensor = Helper.WordtoTensor(word, vocab, sos, eos)
            tensor_list.append(word_tensor)
        word_tensor_pad = pad_sequence(tensor_list, padding_value=2, batch_first=True)
        return word_tensor_pad
    
    @staticmethod
    def Optimizer(model, opt, learning_rate):
        if opt == 'Adam':
            return optim.Adam(model.parameters(), lr=learning_rate)
        elif opt == 'Nadam':
            return optim.NAdam(model.parameters(), lr=learning_rate)
        else:
            return optim.SGD(model.parameters(), lr=learning_rate)

