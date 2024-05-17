from Alphabets import AlphabetCreation
import torch
from torch import optim
from torch.nn.utils.rnn import pad_sequence


SOS_char = "<SOS>"
EOS_char = "<EOS>"
PAD_char = "$"

class Helper:
    @staticmethod
    def LanguageVocabulary(data, input_lang, output_lang):
        # Create vocabularies for input and output languages
        input_vocab = AlphabetCreation(input_lang)
        output_vocab = AlphabetCreation(output_lang)
        
        # Add words to the vocabularies
        for pair in data:
            input_vocab.addWordtoDict(pair[0])
            output_vocab.addWordtoDict(pair[1])
        
        return input_vocab, output_vocab

    @staticmethod
    def WordtoTensor(word, vocab, sent=(False, False)):
        char_list = []
        # Add Start of Sentence (SOS) token if specified
        if sent[0]:
            char_list.append(vocab.char2index[SOS_char])
        # Convert each character in the word to its index in the vocabulary
        for char in word:
            char_list.append(vocab.char2index[char])
        # Add End of Sentence (EOS) token if specified
        if sent[1]:
            char_list.append(vocab.char2index[EOS_char])
        # Convert list of indices to a tensor
        char_tensor = torch.tensor(char_list, dtype=torch.long)
        return char_tensor

    @staticmethod
    def DataProcessing(data, vocab, sent=(False, False)):
        tensor_list = []
        # Process each word in the dataset
        for word in data:
            word_tensor = Helper.WordtoTensor(word, vocab, sent)
            tensor_list.append(word_tensor)
        # Pad the sequences to ensure they have the same length
        word_tensor_pad = pad_sequence(tensor_list, padding_value=2, batch_first=True)
        return word_tensor_pad

    ''' 
        Returns the optimizer based on users choice
        Input : opt -> users optimizer = string, learning_rate
        returns optimizer by initializing it to learining rate
    '''
    @staticmethod
    def Optimizer(model, opt, learning_rate):
        # Select and return the optimizer based on the specified type
        if opt == 'Adam':
            return optim.Adam(model.parameters(), lr=learning_rate)
        elif opt == 'Nadam':
            return optim.NAdam(model.parameters(), lr=learning_rate)
        else:
            return optim.SGD(model.parameters(), lr=learning_rate)
