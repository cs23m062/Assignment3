PATH_TO_DATA = '/kaggle/input/aksharantar-sampled/aksharantar_sampled/ben'
out_lang = 'ben'

# import required libraries
import numpy as np
import pandas as pd
import random


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SOS_token = "@"
EOS_token = "#"
PAD_token = "^"
UNK_token = "$"

SOS_idx = 0
EOS_idx = 1
PAD_idx = 2
UNK_idx = 3

batch_size = 32

def timeInMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    s = format(s, ".0f")
    return str(m) + "m " + str(s) + "s"

class Alphabets:
    def __init__(self, name):
        self.name = name
        self.char2index = {SOS_token: SOS_idx, EOS_token: EOS_idx, PAD_token: PAD_idx, UNK_token: UNK_idx}
        self.char2count = {}
        self.index2char = {SOS_idx: SOS_token, EOS_idx: EOS_token, PAD_idx: PAD_token, UNK_idx: UNK_token}
        self.num_chars = 4  # Count SOS, EOS, PAD and UNK

    def addWord(self, word):
        for char in word:
            self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.num_chars
            self.char2count[char] = 1
            self.index2char[self.num_chars] = char
            self.num_chars = self.num_chars + 1
        else:
            self.char2count[char] += 1


class Processing :
    
    @staticmethod
    def CreateVocabulary(data, in_scr, out_scr):
        output_lang = Alphabets(out_scr)
        input_lang = Alphabets(in_scr)

        for lang_pairs in data:
            input_lang.addWord(lang_pairs[0])
            output_lang.addWord(lang_pairs[1])

        return input_lang, output_lang

    @staticmethod
    def WordToTensor(word, vocab, sos=False, eos=False):
        #tensorFromWord
        char_list = []
        if sos:
            char_list.append(vocab.char2index[SOS_token])
        for char in word:
            if char in vocab.char2index:
                char_list.append(vocab.char2index[char])
            else:
                char_list.append(vocab.char2index[UNK_token])
        if eos:
            char_list.append(vocab.char2index[EOS_token])
        char_tensor = torch.tensor(char_list, dtype=torch.long)
        return char_tensor
    
    @staticmethod
    def processData(data, vocab, sos=False, eos=False):
        tensor_list = []
        for word in data:
            word_tensor = Processing.WordToTensor(word, vocab, sos, eos)
            tensor_list.append(word_tensor)
        word_tensor_pad = pad_sequence(tensor_list, padding_value=PAD_idx, batch_first=True)
        return word_tensor_pad
    
    @staticmethod
    def wordFromTensor(word_tensor, vocab):
        word = ""
        for idx in word_tensor:
            if idx == EOS_idx:
                break
            if idx >= UNK_idx:
                word += vocab.index2char[idx.item()]
        return word
    

class Encoder(nn.Module):
    def __init__(self, config): 
        #cell_type, input_size, embedding_size, hidden_size, num_layers, dp, bidir=False):
        super(Encoder, self).__init__()
        self.cell_type = config['cell_type']
        self.input_size = config['input_size']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['enc_num_layers']
        self.bidir = config['bidirectional']
        
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        if self.cell_type == "RNN":
            self.cell = nn.RNN(self.embedding_size, self.hidden_size, self.num_layers, dropout=config['dropout'], bidirectional=self.bidir)
        elif self.cell_type == "GRU":
            self.cell = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers, dropout=config['dropout'], bidirectional=self.bidir)
        elif self.cell_type == "LSTM":
            self.cell = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout=config['dropout'], bidirectional=self.bidir)
        
        self.dropout = nn.Dropout(config['dropout'])
        self.config = config
        
    def forward(self, inp):
        embedding = self.dropout(self.embedding(inp))
        cell = None
        if self.cell_type == "LSTM":
            outputs, (hidden, cell) = self.cell(embedding)
            if self.bidir:
                b_sz = cell.size(1)
                cell = cell.view(self.num_layers, 2, b_sz, -1)
                cell = cell[-1]
                cell = cell.mean(axis=0)
            else:
                cell = cell[-1,:,:]
            cell = cell.unsqueeze(0)
        else:
            outputs, hidden = self.cell(embedding)
        
        if self.bidir:
            b_sz = hidden.size(1)
            hidden = hidden.view(self.num_layers, 2, b_sz, -1)
            hidden = hidden[-1]
            hidden = hidden.mean(axis=0)
        else:
            hidden = hidden[-1,:,:]
        hidden = hidden.unsqueeze(0)

        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, config):
        #cell_type, input_size, embedding_size, hidden_size, output_size, num_layers, dp
        super(Decoder, self).__init__()
        self.cell_type = config['cell_type']
        self.input_size = config['input_size']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.num_layers = config['dec_num_layers']
        
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        
        if self.cell_type == "RNN":
            self.cell = nn.RNN(self.embedding_size, self.hidden_size, self.num_layers, dropout=config['dropout'])
        elif self.cell_type == "GRU":
            self.cell = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers, dropout=config['dropout'])
        elif self.cell_type == "LSTM":
            self.cell = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout=config['dropout'])
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        
        self.dropout = nn.Dropout(config['dropout'])
        self.config = config

    def forward(self, inp, hidden, cell):
        inp = inp.unsqueeze(0)
        embedding = self.dropout(self.embedding(inp))

        if self.cell_type == "LSTM":
            outputs, (hidden, cell) = self.cell(embedding, (hidden, cell))
        else:
            outputs, hidden = self.cell(embedding, hidden)
        
        # Reshape outputs to (sequence_length, batch_size, hidden_size)
        #outputs = outputs.permute(1, 0, 2)
        
        print('In decoder: ',outputs.shape)
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        
        # Reshape predictions to (sequence_length, batch_size, output_size)
        #predictions = predictions.permute(1, 0, 2)
        
        return predictions, hidden, cell
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_sz = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.output_size

        outputs = torch.zeros(target_len, batch_sz, target_vocab_size).to(device)
        
        print(outputs.shape)
        hidden, cell = self.encoder(source)
        print(hidden.shape)
        print(cell.shape)
        hidden = hidden.repeat(self.decoder.num_layers,1,1)
        if self.decoder.cell_type == "LSTM":
            cell = cell.repeat(self.decoder.num_layers,1,1)

        
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(dim=1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
    

def sum_accuracy(preds, target):
    num_equal_columns = torch.logical_or(preds == target, target == PAD_idx).all(dim=0).sum().item()
    return num_equal_columns

def evaluateModel(model, dataloader, criterion, b_sz=32):
    model.eval()
    n_data = len(dataloader) * b_sz
    loss_epoch = 0
    n_correct = 0
    
    with torch.no_grad():
        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            input_seq = input_seq.T.to(device)
            target_seq = target_seq.T.to(device)
            output = model(input_seq, target_seq, teacher_force_ratio=0.0)
            pred_seq = output.argmax(dim=2)
            n_correct += sum_accuracy(pred_seq, target_seq)
            output = output[1:].reshape(-1, output.shape[2])
            target = target_seq[1:].reshape(-1)
            loss = criterion(output, target)
            loss_epoch += loss.item()
        
        acc = n_correct / n_data
        acc = acc * 100.0
        loss_epoch /= len(dataloader)
        return loss_epoch, acc
    
def trainFunction(model,optimizer,train_dataloader,valid_dataloader,epochs):
#     start = time.time()
    criterion = nn.CrossEntropyLoss()
    batch_size=32
    
    max_val_acc = -1.0
    max_val_epoch = 0
    trigger = 0
    
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    
    for epoch in range(epochs):
#         print(f"[Epoch {epoch+1} / {num_epochs}]")
        
        model.train()
        for batch_idx, (input_seq, target_seq) in enumerate(train_dataloader):
            input_seq = input_seq.T.to(device)
            target_seq = target_seq.T.to(device)
            
            print(input_seq.shape)
            print(target_seq.shape)
            #input_seq = torch.transpose(input_seq, 0, 1).to(device)
            #target_seq = torch.transpose(target_seq, 0, 1).to(device)

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
        tr_loss, tr_acc = evaluateModel(model, train_dataloader, criterion, batch_size)
        training_loss.append(tr_loss)
        training_accuracy.append(tr_acc)
        print(f"Training Loss: {tr_loss:.2f}")
        print(f"Training Accuracy: {tr_acc:.2f}")
        

        #-----------------------------------------------
        # Valid loss and accuracy
        val_loss, val_acc = evaluateModel(model, valid_dataloader, criterion, batch_size)
        validation_loss.append(val_loss)
        validation_accuracy.append(val_acc)
        print(f"Validation Loss: {val_loss:.2f}")
        print(f"Validation Accuracy: {val_acc:.2f}")

#         wandb.log({'tr_loss' : tr_loss, 'tr_acc' : tr_acc, 'val_loss' : val_loss, 'val_acc' : val_acc})

        if val_acc >= max_val_acc:
            patience = 0
            max_val_acc = val_acc
            max_val_epoch = epoch
        else:
            patience += 1
        
        if patience == 5:
            print('Early stopping!')
            break

#         end = time.time()
#         print("Time: ", timeInMinutes(end-start))
#         print("----------------------------------")
#    for i in range(max_val_epoch+1):
#        wandb.log({'tr_loss' : tr_loss_list[i], 'tr_acc' : tr_acc_list[i], 'val_loss' : val_loss_list[i], 'val_acc' : val_acc_list[i]})

def optimizer_func(model,learning_rate,opt):
    if(opt=='Adam'):
        return optim.Adam(model.parameters(), lr=learning_rate)
    else :
        return optim.SGD(model.parameters(), lr=learning_rate)
    
# load dataset
train_data = pd.read_csv(PATH_TO_DATA+'/'+out_lang+'_train.csv',header=None).values
valid_data = pd.read_csv(PATH_TO_DATA+'/'+out_lang+'_valid.csv',header=None).values

# build vocabulary
in_lang = 'eng'
eng_vocab, target_vocab = Processing.CreateVocabulary(train_data,in_lang,out_lang)

eng_train = Processing.processData(train_data[:,0], eng_vocab, eos=True).to(device=device)
eng_valid = Processing.processData(valid_data[:,0], eng_vocab, eos=True).to(device=device)

ben_train = Processing.processData(train_data[:,1], target_vocab, sos=True, eos=True).to(device=device)
ben_valid = Processing.processData(valid_data[:,1], target_vocab, sos=True, eos=True).to(device=device)

train_dataset = TensorDataset(eng_train, ben_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = TensorDataset(eng_valid, ben_valid)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

config = {
    'cell_type' : 'LSTM',
    'embedding_size': 64,
    'hidden_size': 64,
    'enc_num_layers': 2,
    'dec_num_layers': 2,
    'dropout': 0.2,
    'bidirectional': True,
    'epochs' : 3,
    'learning_rate' : 0.001
}

def main():
    
    input_size_encoder = eng_vocab.num_chars
    input_size_decoder = target_vocab.num_chars
    output_size = input_size_decoder
    
    config['input_size'] = input_size_encoder
    config['output_size'] = output_size
    
    #encoder_net = Encoder(cell_type, input_size_encoder, embedding_size, hidden_size, enc_num_layers, dropout, bidirectional).to(device)
    encoder = Encoder(config).to(device)
    #decoder_net = Decoder(cell_type,input_size_decoder,embedding_size,hidden_size,output_size,dec_num_layers,dropout).to(device)
    decoder = Decoder(config).to(device)
        
    model = Seq2Seq(encoder,decoder).to(device)
    optimizer = optimizer_func(model,config['learning_rate'],'Adam')
    
    trainFunction(model,optimizer, train_dataloader, valid_dataloader, config['epochs'])
    
main()