import pandas as pd
from Helpers import Helper
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataPreparation:
    '''
        Function which finds out all the character in the target language
        Inputs :    Target languagage
        Returns : string containing all characters in target language

        Language supported = Hindi Bengali and Telugu    
    '''
    def target_language_alphabets(self, target_lang):
        target = ''
        # Determine the Unicode range for the target language's alphabet
        if target_lang == 'hin':  # Hindi
            for alpha in range(2304, 2432):
                target += chr(alpha)
        elif target_lang == 'ben':  # Bengali
            for alpha in range(2432, 2560):
                target += chr(alpha)
        else:  # Default to Tamil (or other similar languages)
            for alpha in range(3072, 3199):
                target += chr(alpha)
        return target

    '''
        Constructor initializes all the variables   
    '''
    def __init__(self, path, inp_lang='eng', target_lang='hin'):
        eng_alphabets = 'abcdefghijklmnopqrstuvwxyz'  # English alphabets
        tar_alphabets = self.target_language_alphabets(target_lang)  # Target language alphabets

        # Load datasets from CSV files
        self.TrainDataFrame = pd.read_csv(path + '/' + target_lang + '_train.csv', header=None)
        self.train_data = self.TrainDataFrame.values
        self.VadiationDataFrame = pd.read_csv(path + '/' + target_lang + '_valid.csv', header=None)
        self.valid_data = self.VadiationDataFrame.values
        self.TestDataFrame = pd.read_csv(path + '/' + target_lang + '_test.csv', header=None)
        self.test_data = self.TestDataFrame.values

        # Build vocabulary for the input and target languages
        new_data = [[eng_alphabets, tar_alphabets]]
        self.english_vocab, self.target_vocab = Helper.LanguageVocabulary(new_data, inp_lang, target_lang)

    '''
        Function which converts all the training,validation and test data into a tensor dataset and then into an dataloader 
        Inputs :  Batch size
        Returns : train,valid and test dataloader
    '''
    def DataSetLoader(self, batch_size):
        # Process the input sequences for the training, validation, and test datasets
        english_train = Helper.DataProcessing(self.train_data[:,0], self.english_vocab, sent=(False, True)).to(device=device)
        english_valid = Helper.DataProcessing(self.valid_data[:,0], self.english_vocab, sent=(False, True)).to(device=device)
        english_test = Helper.DataProcessing(self.test_data[:,0], self.english_vocab, sent=(False, True)).to(device=device)

        # Process the target sequences for the training, validation, and test datasets
        target_train = Helper.DataProcessing(self.train_data[:,1], self.target_vocab, sent=(True, True)).to(device=device)
        target_valid = Helper.DataProcessing(self.valid_data[:,1], self.target_vocab, sent=(True, True)).to(device=device)
        target_test = Helper.DataProcessing(self.test_data[:,1], self.target_vocab, sent=(True, True)).to(device=device)

        # Create TensorDataset and DataLoader for training data
        train_dataset = TensorDataset(english_train, target_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Create TensorDataset and DataLoader for validation data
        valid_dataset = TensorDataset(english_valid, target_valid)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        # Create TensorDataset and DataLoader for test data
        test_dataset = TensorDataset(english_test, target_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, valid_dataloader, test_dataloader
