import torch
import torch.nn as nn
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def create_cell(self, dropout):
        # Create the recurrent cell based on the specified cell type
        if self.cell_type == "RNN":
            return nn.RNN(self.embedding_size, self.hidden_size, self.num_layers, dropout=dropout, bidirectional=self.bidir)
        elif self.cell_type == "GRU":
            return nn.GRU(self.embedding_size, self.hidden_size, self.num_layers, dropout=dropout, bidirectional=self.bidir)
        elif self.cell_type == "LSTM":
            return nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout=dropout, bidirectional=self.bidir)

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.cell_type = config['cell_type']
        self.input_size = config['input_size']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['enc_num_layers']
        self.bidir = config['bidirectional']
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)  # Embedding layer
        self.cell = self.create_cell(config['dropout'])  # Recurrent cell
        self.dropout = nn.Dropout(config['dropout'])  # Dropout layer
        self.config = config

    def forward(self, inp):
        embedding = self.dropout(self.embedding(inp))  # Apply embedding and dropout
        hidden = None
        cell = None
        outputs, cell_data = self.cell(embedding)  # Forward pass through the recurrent cell
        hidden = cell_data[0]

        if len(cell_data) == 2:
            cell = cell_data[1]

        if self.bidir:
            # For bidirectional cells, combine the forward and backward hidden states
            hidden = hidden.view(self.num_layers, 2, hidden.size(1), -1)[-1].mean(axis=0)
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
            if cell is not None:
                cell = cell.view(self.num_layers, 2, cell.size(1), -1)[-1].mean(axis=0)
        else:
            hidden = hidden[-1, :, :]
            if cell is not None:
                cell = cell[-1, :, :]

        hidden = hidden.unsqueeze(0)
        if cell is not None:
            cell = cell.unsqueeze(0)

        return outputs, hidden, cell  # Return outputs, hidden state, and cell state (if applicable)

class Decoder(nn.Module):
    def create_cell(self, dropout):
        # Create the recurrent cell based on the specified cell type
        if self.cell_type == "RNN":
            return nn.RNN(self.dec_cell_input, self.hidden_size, self.num_layers, dropout=dropout)
        elif self.cell_type == "GRU":
            return nn.GRU(self.dec_cell_input, self.hidden_size, self.num_layers, dropout=dropout)
        else:
            return nn.LSTM(self.dec_cell_input, self.hidden_size, self.num_layers, dropout=dropout)
            
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.cell_type = config['cell_type']
        self.input_size = config['output_size']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.num_layers = config['dec_num_layers']
        self.config = config
        self.dropout = nn.Dropout(config['dropout'])  # Dropout layer
        self.dec_cell_input = self.embedding_size + self.hidden_size  # Input size for the decoder cell
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)  # Embedding layer
        self.cell = self.create_cell(config['dropout'])  # Recurrent cell
        self.fc1 = nn.Linear(self.hidden_size * 2, self.output_size)  # Fully connected layer for output
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  # Fully connected layer for attention mechanism
    
    def change_mat(self, mat, dim1, dim2, dim3):
        # Helper function to permute the dimensions of a tensor
        return mat.permute(dim1, dim2, dim3)
    
    def forward(self, target_alphabet, encoder_outputs, hidden, cell):
        target_alphabet = target_alphabet.unsqueeze(0)  # Add batch dimension
        embedding = self.dropout(self.embedding(target_alphabet))  # Apply embedding and dropout

        encoder_outputs_fc = self.fc2(encoder_outputs)  # Linear transformation for attention
        last_hidden = hidden[-1:]
        temp_enc = self.change_mat(encoder_outputs_fc, 1, 0, 2)
        temp_lh = self.change_mat(last_hidden, 1, 2, 0)
        
        score_tensor = torch.matmul(temp_enc, temp_lh)  # Calculate attention scores
        score_tensor = self.change_mat(score_tensor, 2, 0, 1)
        
        attention_weights = F.softmax(score_tensor, dim=2)  # Apply softmax to get attention weights
        temp_attn = self.change_mat(attention_weights, 1, 0, 2)
        
        context_tensor = torch.matmul(temp_attn, temp_enc)  # Calculate context vector
        context_tensor = self.change_mat(context_tensor, 1, 0, 2)
        
        new_embedding = torch.cat([embedding, context_tensor], dim=2)  # Concatenate embedding and context
        
        if self.cell_type == "LSTM":
            outputs, (hidden, cell) = self.cell(new_embedding, (hidden, cell))  # LSTM cell forward pass
        else:
            outputs, hidden = self.cell(new_embedding, hidden)  # RNN/GRU cell forward pass

        concat_outputs = torch.cat([outputs, context_tensor], dim=2)  # Concatenate outputs and context
        predictions = self.fc1(concat_outputs)  # Linear transformation for output prediction
        predictions = predictions.squeeze(0)
        
        return predictions, hidden, cell, attention_weights  # Return predictions, hidden state, cell state, and attention weights

class LangToLang(nn.Module):
    def __init__(self, encoder, decoder):
        super(LangToLang, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        outputs = torch.zeros(target.shape[0], source.shape[1], self.decoder.output_size).to(device)  # Initialize output tensor
        enc_out, hidden, cell = self.encoder(source)  # Encode the source sequence
        hidden = hidden.repeat(self.decoder.num_layers, 1, 1)  # Repeat hidden state for each decoder layer
        if self.decoder.cell_type == "LSTM":
            cell = cell.repeat(self.decoder.num_layers, 1, 1)

        attn_matrix = torch.zeros(target.shape[0], source.shape[1], source.shape[0]).to(device)  # Initialize attention matrix

        x = target[0]  # Start with the first target token
        for i in range(1, target.shape[0]):
            output, hidden, cell, attn_w = self.decoder(x, enc_out, hidden, cell)  # Decode the next token
            outputs[i] = output  # Store the output
            attn_matrix[i] = attn_w  # Store the attention weights
            best_guess = output.argmax(dim=1)  # Get the best guess for the next token
            x = target[i] if random.random() < teacher_force_ratio else best_guess  # Use teacher forcing or predicted token

        return outputs, attn_matrix  # Return the output predictions and attention matrix
