import torch
import torch.nn as nn
from Helpers import Helper
from Seq2Seq import Encoder
from Seq2Seq import Decoder
from Seq2Seq import LangToLang
from CreateDataset import DataPreparation
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_saving_path = ''

class TrainingAndValidation :
    
    '''
        Evaluator function takes 3 arguments:
        Input : 
            Trained Model
            Dataloader - > Train,Test,Validation
            batch_size
        Returns :
            Loss and Accuracy of the model on the dataloader
    '''
    
    @staticmethod
    def evaluateModel(model, dataloader, batch_size):
        criterion = nn.CrossEntropyLoss()  # Define the loss function
        model.eval()  # Put the model in evaluation mode
        loss_epoch = 0  # Initialize the epoch loss to zero
        correct = 0  # Initialize the count of correct predictions to zero
        
        with torch.no_grad():  # No need to calculate gradients during evaluation
            for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                '''
                We need to match the dimensions of the input to
                multiply the dimensions; that's why we are doing the transpose here.

                Initial dimension = [batchsize, max_seq_len]
                Transposed dimension = [max_seq_len, batchsize]
                '''
                input_seq = torch.transpose(input_seq, 0, 1).to(device)  # Transpose and move input to the device
                target_seq = torch.transpose(target_seq, 0, 1).to(device)  # Transpose and move target to the device

                # As we are in evaluation mode, we are keeping the teacher force ratio to be 0.0
                output, _ = model(input_seq, target_seq, teacher_force_ratio=0.0)  # Get model output

                pred_seq = output.argmax(dim=2)  # Get the predictions by selecting the index with the highest score
                
                # Create a boolean mask where either preds equal target or target equals padding character (assumed padding character index is 2)
                mask = torch.logical_or(pred_seq == target_seq, target_seq == 2)
                # Check along dimension 0 (columns) if all elements are True, sum the True values, and convert to item
                correct += mask.all(dim=0).sum().item()
                
                # Reshape the output and target for loss calculation
                output = output[1:].reshape(-1, output.shape[2])  # Exclude the first token and flatten
                target = target_seq[1:].reshape(-1)  # Exclude the first token and flatten
                
                loss = criterion(output, target)  # Calculate the loss
                loss_epoch += loss.item()  # Accumulate the loss for the epoch

            accuracy = correct / (len(dataloader) * batch_size)  # Calculate accuracy
            accuracy = accuracy * 100.0  # Convert to percentage
            loss_epoch /= len(dataloader)  # Average the loss over all batches
            return loss_epoch, accuracy  # Return the epoch loss and accuracy


    '''
        Trainer function takes 5 arguments:
        Input : 
            1. Model to be trained
            2. Dataloader - > Training data + validation data
            3. Number of epochs to be run
            4. batch_size
            5. Learning_rate
        Returns :
            nothing
        Saves the model at the end of training so that it can be used later on
    '''

    @staticmethod    
    def trainer(model, dataloader, epochs, opt_str, batch_size, learning_rate):
        criterion = nn.CrossEntropyLoss()  # Define the loss function
        train_dataloader = dataloader[0]  # Training dataloader
        valid_dataloader = dataloader[1]  # Validation dataloader

        optimizer = Helper.Optimizer(model, opt_str, learning_rate)  # Initialize the optimizer
        for epoch in range(epochs):
            print('====================================')
            print("Epoch:", epoch + 1)
            
            model.train()  # Put the model in training mode

            for batch_idx, (input_seq, target_seq) in enumerate(train_dataloader):
                '''
                We need to match the dimensions of the input to 
                multiply the dimensions; that's why we are doing the transpose here.

                Initial dimension = [batchsize, max_seq_len]
                Transposed dimension = [max_seq_len, batchsize]
                '''
                input_seq = torch.transpose(input_seq, 0, 1).to(device)  # Transpose and move input to the device
                target_seq = torch.transpose(target_seq, 0, 1).to(device)  # Transpose and move target to the device

                # Forward pass through the model
                output, attn = model(input_seq, target_seq)
                output = output[1:].reshape(-1, output.shape[2])  # Exclude the first token and flatten
                target = target_seq[1:].reshape(-1)  # Exclude the first token and flatten
                
                optimizer.zero_grad()  # Zero the gradients
                loss = criterion(output, target)  # Calculate the loss
                loss.backward()  # Backward pass to calculate gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Clip gradients to prevent exploding gradients
                optimizer.step()  # Update model parameters

            # Evaluate model on the training data
            train_loss, train_acc = TrainingAndValidation.evaluateModel(model, train_dataloader, batch_size)
            print("Training Loss:", train_loss)
            print("Training Accuracy:", train_acc)

            # Evaluate model on the validation data
            val_loss, val_acc = TrainingAndValidation.evaluateModel(model, valid_dataloader, batch_size)
            print(f"Validation Loss: {val_loss:.2f}")
            print(f"Validation Accuracy: {val_acc:.2f}")

            # Log training and validation metrics to Weights & Biases (commented out)
            # wandb.log({'tr_loss' : train_loss, 'tr_acc' : train_acc, 'val_loss' : val_loss, 'val_acc' : val_acc, 'epoch' : epoch+1})

        # Save the trained model parameters
        torch.save(model.state_dict(), model_saving_path)



config = {
        'cell_type' : "LSTM",
        'embedding_size': 64,
        'hidden_size': 256,
        'enc_num_layers': 2,
        'dec_num_layers': 3,
        'dropout': 0.3,
        'bidirectional': True,
}


def main(args):
    inp_lang = 'eng'
    target_lang  = args.target_lang
    PATH_TO_DATA = '/content/drive/MyDrive/aksharantar_sampled/' + target_lang
    model_saving_path = '/best_model_attention.pth'
    test_pred_path = '/predictions_attention.csv'


    config['cell_type'] = args.cell_type
    config['embedding_size'] = args.embedding_size
    config['hidden_size'] = args.hidden_size
    config['enc_num_layers'] = args.encoder_layers
    config['dec_num_layers'] = args.decoder_layers
    config['dropout'] = args.dropout
    config['bidirectional'] =  args.bidirectional
    config['epochs'] = args.epochs

    epochs = args.epochs
    learning_rate = args.learning_rate
    

    dataset = DataPreparation(PATH_TO_DATA,inp_lang,target_lang)
    batch_size = args.batch_size
    train_dataloader,valid_dataloader,test_dataloader = dataset.DataSetLoader(batch_size);
    
    # Fixed parameters for encoder and decoder
    input_size_encoder = dataset.english_vocab.n_chars
    input_size_decoder = dataset.target_vocab.n_chars
    output_size = input_size_decoder
    
    config['input_size'] = input_size_encoder
    config['output_size'] = output_size
    config['bidirectional'] = True
    encoder = Encoder(config).to(device)
    decoder = Decoder(config).to(device)
    
    model = LangToLang(encoder, decoder).to(device)
    opt_str = args.optimizer
    TrainingAndValidation.trainer(model,(train_dataloader,valid_dataloader),epochs,opt_str,batch_size,learning_rate)

    model.load_state_dict(torch.load(model_saving_path))
    test_loss,test_accuracy = TrainingAndValidation.evaluateModel(model,test_dataloader,nn.CrossEntropyLoss(),batch_size,(dataset.TestDataFrame,True))
    print(f"Test Loss: {test_loss:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep_LearingAssignment1_CS23M062 -command line arguments")
    parser.add_argument("-wp","--wandb_project", type=str, default ='Shubhodeep_CS6190_DeepLearing_Assignment3', help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we","--wandb_entity", type=str, default ='shubhodeepiitm062',help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-e","--epochs",type=int,default = 10,help ='Number of epochs to train neural network.')
    parser.add_argument("-b","--batch_size",type=int,default = 32,help='Batch size used to train neural network.')  
    parser.add_argument('-lr','--learning_rate',type=float,default=0.001,help='Learning rate used to optimize model parameters')
    parser.add_argument('-t','--target_lang',type=str,default='hin',help='Target Language in which transliteration system works')
    parser.add_argument('-ct',"--cell_type",type=str,default="LSTM",help='Type of cell to be used in architecture Choose b/w [LSTM,RNN,GRU]')
    parser.add_argument('-em','--embedding_size',type=int,default=128,help='size of embedding to be used in encoder decoder')
    parser.add_argument('-hi','--hidden_size',type=int,default=512,help='Hidden layer size of encoder and decoder')
    parser.add_argument('-el',"--encoder_layers",type=int,default=4,help='Number of hidden layers in encoder')
    parser.add_argument('-dl',"--decoder_layers",type=int,default=4,help='Number of hidden layers in decoder')
    parser.add_argument('-dr','--dropout',type=float,default=0.2,help='dropout probability')
    parser.add_argument('-bi',"--bidirectional",type=bool,default=True,help='Whether you want the data to be read from both directions')
    parser.add_argument('-op','--optimizer',type=str,default='Adam',help='choices: ["Sgd","Adam", "Nadam"]')  
    args = parser.parse_args()
    main(args)
