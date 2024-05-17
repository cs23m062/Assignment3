import torch.nn as nn
import torch
import random 

# Determine if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Encoder class
class Encoder(nn.Module):
    # Function to create the appropriate RNN cell based on the configuration
    def create_cell(self, dropout):
        if self.cell_type == "RNN":
            return nn.RNN(self.embedding_size, self.hidden_size, self.num_layers, dropout=dropout, bidirectional=self.bidir)
        elif self.cell_type == "GRU":
            return nn.GRU(self.embedding_size, self.hidden_size, self.num_layers, dropout=dropout, bidirectional=self.bidir)
        elif self.cell_type == "LSTM":
            return nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout=dropout, bidirectional=self.bidir)

    # Initialize the Encoder with the given configuration
    def __init__(self, config):
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

    # Define the forward pass of the Encoder
    def forward(self, inp):
        embedding = self.dropout(self.embedding(inp))
        outputs, cell_data = self.cell(embedding)
        hidden = cell_data[0]

        # Check if the RNN cell is an LSTM which returns (hidden, cell)
        if len(cell_data) == 2:
            cell = cell_data[1]
        else:
            cell = None

        # Handle bidirectional RNN case
        if self.bidir:
            hidden = hidden.view(self.num_layers, 2, hidden.size(1), -1)[-1].mean(axis=0)
            if cell is not None:
                cell = cell.view(self.num_layers, 2, cell.size(1), -1)[-1].mean(axis=0)
        else:
            hidden = hidden[-1, :, :]
            if cell is not None:
                cell = cell[-1, :, :]

        hidden = hidden.unsqueeze(0)
        if cell is not None:
            cell = cell.unsqueeze(0)

        return hidden, cell

# Define the Decoder class
class Decoder(nn.Module):
    # Function to create the appropriate RNN cell based on the configuration
    def create_cell(self, dropout):
        if self.cell_type == "RNN":
            return nn.RNN(self.embedding_size, self.hidden_size, self.num_layers, dropout=dropout)
        elif self.cell_type == "GRU":
            return nn.GRU(self.embedding_size, self.hidden_size, self.num_layers, dropout=dropout)
        elif self.cell_type == "LSTM":
            return nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout=dropout)
            
    # Initialize the Decoder with the given configuration
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.cell_type = config['cell_type']
        self.input_size = config['output_size']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.num_layers = config['dec_num_layers']
        self.dropout = nn.Dropout(config['dropout'])
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.cell = self.create_cell(config['dropout'])
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    # Define the forward pass of the Decoder
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))

        if self.cell_type == "LSTM":
            outputs, (hidden, cell) = self.cell(embedding, (hidden, cell))
        else:
            outputs, hidden = self.cell(embedding, hidden)

        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell

# Define the Seq2Seq model which combines the Encoder and Decoder
class LangToLang(nn.Module):
    def __init__(self, encoder, decoder):
        super(LangToLang, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # Define the forward pass of the Seq2Seq model
    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_length = target.shape[0]
        target_vocab_size = self.decoder.output_size

        outputs = torch.zeros(target_length, batch_size, target_vocab_size).to(device)

        # Get the initial hidden and cell states from the Encoder
        hidden, cell = self.encoder(source)
        hidden = hidden.repeat(self.decoder.num_layers, 1, 1)
        if self.decoder.cell_type == "LSTM":
            cell = cell.repeat(self.decoder.num_layers, 1, 1)

        # The first input to the Decoder is the <sos> token
        x = target[0]
        for i in range(1, target_length):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[i] = output
            best_guess = output.argmax(dim=1)

            # Use teacher forcing
            x = target[i] if random.random() < teacher_force_ratio else best_guess

        return outputs
