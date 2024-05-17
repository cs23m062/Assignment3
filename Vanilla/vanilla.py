# import cell
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from VanillaSeq2Seq import Encoder
from VanillaSeq2Seq import Decoder
from VanillaSeq2Seq import LangToLang

# Paste your own key here
import wandb
#wandb.login()
# functions with comments explained in script file
# no redundant comments here
# only new functions are explained here in comments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SOS_char = "<SOS>"
EOS_char = "<EOS>"
PAD_char = ""

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

class Validator :
    @staticmethod
    def evaluateModel(model, dataloader, criterion, batch_size):
        model.eval()
        
        total = len(dataloader) * batch_size
        loss_epoch = 0
        correct = 0
        
        with torch.no_grad():
            for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                input_seq = torch.transpose(input_seq,0,1).to(device)
                target_seq = torch.transpose(target_seq,0,1).to(device)
                output = model(input_seq, target_seq, teacher_force_ratio=0.0)
                
                pred_seq = output.argmax(dim=2)
                # Create a boolean mask where either preds equals target or target equals padding character
                mask = torch.logical_or(pred_seq == target_seq, target_seq == 2)
                # Check along dimension 0 (columns) if all elements are True, sum the True values, and convert to item
                correct += mask.all(dim=0).sum().item()
                
                output = output[1:].reshape(-1, output.shape[2])
                target = target_seq[1:].reshape(-1)
                
                loss = criterion(output, target)
                loss_epoch += loss.item()
            
            accuracy = correct / total
            accuracy = accuracy * 100.0
            loss_epoch /= len(dataloader)
            return loss_epoch, accuracy

import sys

def trainer(model,train_dataloader, valid_dataloader, num_epochs,opt_str,batch_size, learning_rate):
    criterion = nn.CrossEntropyLoss()

    optimizer = Helper.Optimizer(model,opt_str,learning_rate)
    for epoch in range(num_epochs):
        print('====================================')
        print(f"[Epoch {epoch+1} / {num_epochs}]")
        
        model.train()

        for batch_idx, (input_seq, target_seq) in enumerate(train_dataloader):
            
            input_seq = torch.transpose(input_seq,0,1).to(device)
            target_seq = torch.transpose(target_seq,0,1).to(device)

            output = model(input_seq, target_seq)

            output = output[1:].reshape(-1, output.shape[2])
            target = target_seq[1:].reshape(-1)
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        #-----------------------------------------------
        # Train loss and accuracy
        train_loss, train_acc = Validator.evaluateModel(model, train_dataloader, criterion, batch_size)
        print(f"Training Loss: {train_loss:.2f}")
        print(f"Training Accuracy: {train_acc:.2f}")

        #-----------------------------------------------
        # Valid loss and accuracy
        val_loss, val_acc = Validator.evaluateModel(model, valid_dataloader, criterion, batch_size)
        print(f"Validation Loss: {val_loss:.2f}")
        print(f"Validation Accuracy: {val_acc:.2f}")

#        wandb.log({'tr_loss' : tr_loss, 'tr_acc' : tr_acc, 'val_loss' : val_loss, 'val_acc' : val_acc})

config = {
    'cell_type' : 'LSTM',
    'embedding_size': 64,
    'hidden_size': 256,
    'enc_num_layers': 2,
    'dec_num_layers': 3,
    'dropout': 0.3,
    'bidirectional': True,
}

inp_lang = 'eng'
target_lang  = 'hin'
PATH_TO_DATA = 'D:\\Deep Learning\\Assignment 4 RNN LSTM GRU\\aksharantar_sampled\\aksharantar_sampled\hin' + target_lang

eng_alphabets = 'abcdefghijklmnopqrstuvwxyz'
tar_alphabets = ''
for alpha in range(2304, 2432):
    tar_alphabets += chr(alpha)

# load dataset
#PATH_TO_DATA + '\\' + target_lang + '_train.csv'
TrainDataFrame = pd.read_csv('/kaggle/input/aksharantar/aksharantar_sampled/hin/hin_train.csv',header=None)
train_data = TrainDataFrame.values
#PATH_TO_DATA + '\\' + target_lang + '_valid.csv',
VadiationDataFrame = pd.read_csv('/kaggle/input/aksharantar/aksharantar_sampled/hin/hin_valid.csv',header=None)
valid_data = VadiationDataFrame.values
TestDataFrame = pd.read_csv('/kaggle/input/aksharantar/aksharantar_sampled/hin/hin_test.csv',header=None)
test_data = TestDataFrame.values

# build vocabulary
english_vocab, target_vocab = Helper.LanguageVocabulary([[eng_alphabets,tar_alphabets]],inp_lang,target_lang)

print(english_vocab.n_chars)
print(target_vocab.n_chars)

english_train = Helper.DataProcessing(train_data[:,0], english_vocab, eos=True).to(device=device)
english_valid = Helper.DataProcessing(valid_data[:,0], english_vocab, eos=True).to(device=device)
english_test = Helper.DataProcessing(test_data[:,0], english_vocab, eos=True).to(device=device)

target_train = Helper.DataProcessing(train_data[:,1], target_vocab, sos=True, eos=True).to(device=device)
target_valid = Helper.DataProcessing(valid_data[:,1], target_vocab, sos=True, eos=True).to(device=device)
target_test = Helper.DataProcessing(test_data[:,1], target_vocab, sos=True, eos=True).to(device=device)

n_train = english_train.size(0)
n_valid = english_valid.size(0)

print(n_train, n_valid)
batch_size = 32

train_dataset = TensorDataset(english_train,target_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = TensorDataset(english_valid, target_valid)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(english_test, target_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


def main():
    num_epochs = 10
    learning_rate = 0.001

    # Fixed parameters for encoder and decoder
    input_size_encoder = english_vocab.n_chars
    input_size_decoder = target_vocab.n_chars
    output_size = input_size_decoder
    
    config['input_size'] = input_size_encoder
    config['output_size'] = output_size
    config['bidirectional'] = True
    encoder = Encoder(config).to(device)
    decoder = Decoder(config).to(device)

    model = LangToLang(encoder, decoder).to(device)
        
    trainer(model,train_dataloader, valid_dataloader, num_epochs,'Adam',batch_size, learning_rate)
    
    loss,acc = Validator.evaluateModel(model, test_dataloader,nn.CrossEntropyLoss(), batch_size) 
    print('Test Loss',loss)
    print('Test Accuracy',acc)
    
main()
