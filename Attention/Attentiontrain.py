import torch
import torch.nn as nn
from Helpers import Helper
from Seq2Seq import Encoder
from Seq2Seq import Decoder
from Seq2Seq import LangToLang
from CreateDataset import DataPreparation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_saving_path = ''

class TrainingAndValidation :
    @staticmethod
    def evaluateModel(model, dataloader, criterion, batch_size):
        model.eval()
        loss_epoch = 0
        correct = 0
        
        with torch.no_grad():
            for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                input_seq = torch.transpose(input_seq,0,1).to(device)
                target_seq = torch.transpose(target_seq,0,1).to(device)
                output,_ = model(input_seq, target_seq, teacher_force_ratio=0.0)
                
                pred_seq = output.argmax(dim=2)
                
                # Create a boolean mask where either preds equals target or target equals padding character
                mask = torch.logical_or(pred_seq == target_seq, target_seq == 2)
                # Check along dimension 0 (columns) if all elements are True, sum the True values, and convert to item
                correct += mask.all(dim=0).sum().item()
                
                output = output[1:].reshape(-1, output.shape[2])
                target = target_seq[1:].reshape(-1)
                
                loss = criterion(output, target)
                loss_epoch += loss.item()


            accuracy = correct / (len(dataloader) * batch_size)
            accuracy = accuracy * 100.0
            loss_epoch /= len(dataloader)
            return loss_epoch, accuracy

    @staticmethod    
    def trainer(model,train_dataloader, valid_dataloader, num_epochs,opt_str,batch_size, learning_rate):
        criterion = nn.CrossEntropyLoss()

        optimizer = Helper.Optimizer(model,opt_str,learning_rate)
        for epoch in range(num_epochs):
            print('====================================')
            print(f"Epoch: {epoch+1}")
            
            model.train()

            for batch_idx, (input_seq, target_seq) in enumerate(train_dataloader):
                
                input_seq = torch.transpose(input_seq,0,1).to(device)
                target_seq = torch.transpose(target_seq,0,1).to(device)

                output,attn = model(input_seq, target_seq)            
                output = output[1:].reshape(-1, output.shape[2])
                target = target_seq[1:].reshape(-1)
                optimizer.zero_grad()
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

            #-----------------------------------------------
            # Train loss and accuracy
            train_loss, train_acc = TrainingAndValidation.evaluateModel(model, train_dataloader, criterion, batch_size)
            print(f"Training Loss: {train_loss:.2f}")
            print(f"Training Accuracy: {train_acc:.2f}")

            #-----------------------------------------------
            # Valid loss and accuracy
            val_loss, val_acc = TrainingAndValidation.evaluateModel(model, valid_dataloader, criterion, batch_size)
            print(f"Validation Loss: {val_loss:.2f}")
            print(f"Validation Accuracy: {val_acc:.2f}")

            #wandb.log({'tr_loss' : train_loss, 'tr_acc' : train_acc, 'val_loss' : val_loss, 'val_acc' : val_acc, 'epoch' : epoch+1})

        torch.save(model.state_dict(), model_saving_path)




def main():
    inp_lang = 'eng'
    target_lang  = 'hin'
    PATH_TO_DATA = 'aksharantar_sampled/aksharantar_sampled/' + target_lang
    model_saving_path = '/best_model_attn.pth'
    test_pred_path = '/pred_attn.csv'


    dataset = DataPreparation(PATH_TO_DATA)
    batch_size = 32
    train_dataloader,valid_dataloader,test_dataloader = dataset.DataSetLoader(batch_size);

    config = {
        'cell_type' : 'LSTM',
        'embedding_size': 64,
        'hidden_size': 256,
        'enc_num_layers': 2,
        'dec_num_layers': 3,
        'dropout': 0.3,
        'bidirectional': True,
    }

    num_epochs = 3
    learning_rate = 0.001

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
    TrainingAndValidation.trainer(model,train_dataloader, valid_dataloader, num_epochs,'Adam',batch_size, learning_rate)

    model.load_state_dict(torch.load(model_saving_path))
    test_loss,test_accuracy = TrainingAndValidation.evaluateModel(model,test_dataloader,nn.CrossEntropyLoss(),batch_size,(dataset.TestDataFrame,True))
    print(f"Test Loss: {test_loss:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")


if __name__ == "__main__":
    main()