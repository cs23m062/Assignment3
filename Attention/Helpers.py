from Alphabets import AlphabetCreation
import torch
from torch import optim
from torch.nn.utils.rnn import pad_sequence


SOS_char = "<SOS>"
EOS_char = "<EOS>"
PAD_char = "$"

class Helper :
    @staticmethod
    def LanguageVocabulary(data, input_lang, output_lang):
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
    def Optimizer(model,opt,learning_rate):
        if(opt=='Adam'):
            return optim.Adam(model.parameters(),lr=learning_rate)
        elif(opt=='Nadam'):
            return optim.NAdam(model.parameters(),lr=learning_rate)
        else : 
            return optim.SGD(model.parameters(), lr=learning_rate)