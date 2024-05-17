# Import necessary libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from VanillaSeq2Seq import Encoder
from VanillaSeq2Seq import Decoder
from VanillaSeq2Seq import LangToLang
from vanillahelper import Helper
from vanilladataset import datasetcreator

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Validator class for evaluating the model
class Validator:
    @staticmethod
    def evaluateModel(model, dataloader, criterion, batch_size):
        model.eval()  # Set the model to evaluation mode
        
        total = len(dataloader) * batch_size
        loss_epoch = 0
        correct = 0
        
        with torch.no_grad():  # Disable gradient calculation
            for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                input_seq = torch.transpose(input_seq, 0, 1).to(device)
                target_seq = torch.transpose(target_seq, 0, 1).to(device)
                
                # Forward pass through the model without teacher forcing
                output = model(input_seq, target_seq, teacher_force_ratio=0.0)
                
                # Get the predictions and create a mask to compare with target sequence
                pred_seq = output.argmax(dim=2)
                mask = torch.logical_or(pred_seq == target_seq, target_seq == 2)
                correct += mask.all(dim=0).sum().item()
                
                output = output[1:].reshape(-1, output.shape[2])
                target = target_seq[1:].reshape(-1)
                
                # Calculate loss
                loss = criterion(output, target)
                loss_epoch += loss.item()
            
            # Calculate accuracy
            accuracy = correct / total * 100.0
            loss_epoch /= len(dataloader)
            return loss_epoch, accuracy

# Trainer function for training the model
def trainer(model, train_dataloader, valid_dataloader, num_epochs, opt_str, batch_size, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = Helper.Optimizer(model, opt_str, learning_rate)
    
    for epoch in range(num_epochs):
        print('====================================')
        print(f"[Epoch {epoch+1} / {num_epochs}]")
        
        model.train()  # Set the model to training mode

        for batch_idx, (input_seq, target_seq) in enumerate(train_dataloader):
            input_seq = torch.transpose(input_seq, 0, 1).to(device)
            target_seq = torch.transpose(target_seq, 0, 1).to(device)
            
            # Forward pass through the model
            output = model(input_seq, target_seq)
            output = output[1:].reshape(-1, output.shape[2])
            target = target_seq[1:].reshape(-1)
            
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        # Evaluate the model on training and validation datasets
        train_loss, train_acc = Validator.evaluateModel(model, train_dataloader, criterion, batch_size)
        print(f"Training Loss: {train_loss:.2f}")
        print(f"Training Accuracy: {train_acc:.2f}")

        val_loss, val_acc = Validator.evaluateModel(model, valid_dataloader, criterion, batch_size)
        print(f"Validation Loss: {val_loss:.2f}")
        print(f"Validation Accuracy: {val_acc:.2f}")

# Configuration dictionary for the model parameters
config = {
    'cell_type': 'LSTM',
    'embedding_size': 64,
    'hidden_size': 256,
    'enc_num_layers': 2,
    'dec_num_layers': 3,
    'dropout': 0.3,
    'bidirectional': True,
}

# Create dataset and dataloaders
dataset = datasetcreator()
batch_size = 32
train_dataloader, valid_dataloader, test_dataloader = dataset.datasetcreation()

# Main function to initialize and train the model
def main():
    num_epochs = 10
    learning_rate = 0.001

    input_size_encoder = 29
    input_size_decoder = 131
    output_size = input_size_decoder
    
    # Update config dictionary with input and output sizes
    config['input_size'] = input_size_encoder
    config['output_size'] = output_size
    config['bidirectional'] = True
    
    # Initialize encoder and decoder with the configuration
    encoder = Encoder(config).to(device)
    decoder = Decoder(config).to(device)

    # Initialize the Seq2Seq model
    model = LangToLang(encoder, decoder).to(device)
        
    # Train the model
    trainer(model, train_dataloader, valid_dataloader, num_epochs, 'Adam', batch_size, learning_rate)
    
    # Evaluate the model on the test dataset
    loss, acc = Validator.evaluateModel(model, test_dataloader, nn.CrossEntropyLoss(), batch_size) 
    print('Test Loss:', loss)
    print('Test Accuracy:', acc)
    
main()
