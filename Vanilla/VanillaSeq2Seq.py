
import torch.nn as nn
import torch
import random 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def create_cell(self,dropout):
        if self.cell_type == "RNN":
            return nn.RNN(self.embedding_size, self.hidden_size, self.num_layers, dropout=dropout, bidirectional=self.bidir)
        elif self.cell_type == "GRU":
            return nn.GRU(self.embedding_size, self.hidden_size, self.num_layers, dropout=dropout, bidirectional=self.bidir)
        elif self.cell_type == "LSTM":
            return nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout=dropout, bidirectional=self.bidir)

    def __init__(self,config):
        super(Encoder, self).__init__()
        self.cell_type = config['cell_type']
        self.input_size = config['input_size']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['enc_num_layers']
        self.bidir = config['bidirectional']
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.cell = self.create_cell(config['dropout'])
        self.dropout = nn.Dropout(config['dropout'])
        self.config = config

    def forward(self, inp):
        embedding = self.dropout(self.embedding(inp))
        cell = None
        hidden = None
        outputs,cell_data = self.cell(embedding)
        hidden = cell_data[0]

        if(len(cell_data) == 2):
            cell = cell_data[1]

        if self.bidir:
            hidden = hidden.view(self.num_layers, 2, hidden.size(1), -1)[-1].mean(axis=0)
            if(cell!=None):
                cell = cell.view(self.num_layers, 2, cell.size(1), -1)[-1].mean(axis=0)
        else:
            hidden = hidden[-1,:,:]
            if(cell!=None):
                cell = cell[-1,:,:]

        hidden = hidden.unsqueeze(0)
        if(cell!=None):
            cell = cell.unsqueeze(0)

        return hidden, cell
    

class Decoder(nn.Module):
    def __init__(self,config):
        super(Decoder, self).__init__()
        self.cell_type = config['cell_type']
        self.input_size = config['output_size']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.num_layers = config['dec_num_layers']
        dp = config['dropout']

        self.dropout = nn.Dropout(dp)

        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        
        if self.cell_type == "RNN":
            self.cell = nn.RNN(self.embedding_size, self.hidden_size, self.num_layers, dropout=dp)
        elif self.cell_type == "GRU":
            self.cell = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers, dropout=dp)
        elif self.cell_type == "LSTM":
            self.cell = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout=dp)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden, cell):    
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))    
        outputs,hidden = self.cell(embedding, hidden)   

        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        if(len(hidden)>1):
            cell = hidden[1]
            return predictions, hidden[0], cell
        return predictions, hidden, cell
    

class LangToLang(nn.Module):
    def __init__(self, encoder, decoder):
        super(LangToLang, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_length = target.shape[0]
        target_vocab_size = self.decoder.output_size

        outputs = torch.zeros(target_length, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)
        hidden = hidden.repeat(self.decoder.num_layers,1,1)
        if self.decoder.cell_type == "LSTM":
            cell = cell.repeat(self.decoder.num_layers,1,1)

        x = target[0]
        for i in range(1, target_length):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[i] = output
            best_guess = output.argmax(dim=1)
            x = target[i] if random.random() < teacher_force_ratio else best_guess

        return outputs