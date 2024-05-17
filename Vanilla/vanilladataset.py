# Import necessary libraries
from vanillahelper import Helper
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the datasetcreator class
class datasetcreator:
    def datasetcreation(self):
        # Define the source and target languages
        inp_lang = 'eng'
        target_lang = 'hin'
        PATH_TO_DATA = 'D:\\Deep Learning\\Assignment 4 RNN LSTM GRU\\aksharantar_sampled\\aksharantar_sampled\\hin' + target_lang

        # Define the alphabets for English and target language (Hindi)
        eng_alphabets = 'abcdefghijklmnopqrstuvwxyz'
        tar_alphabets = ''
        for alpha in range(2304, 2432):  # Unicode range for Devanagari script
            tar_alphabets += chr(alpha)

        # Load datasets from CSV files
        TrainDataFrame = pd.read_csv('/content/drive/MyDrive/aksharantar_sampled/hin/hin_train.csv', header=None)
        train_data = TrainDataFrame.values

        VadiationDataFrame = pd.read_csv('/content/drive/MyDrive/aksharantar_sampled/hin/hin_valid.csv', header=None)
        valid_data = VadiationDataFrame.values

        TestDataFrame = pd.read_csv('/content/drive/MyDrive/aksharantar_sampled/hin/hin_test.csv', header=None)
        test_data = TestDataFrame.values

        # Build vocabulary for English and target language
        english_vocab, target_vocab = Helper.LanguageVocabulary([[eng_alphabets, tar_alphabets]], inp_lang, target_lang)

        print(english_vocab.n_chars)
        print(target_vocab.n_chars)

        # Process the data for the model
        english_train = Helper.DataProcessing(train_data[:, 0], english_vocab, eos=True).to(device=device)
        english_valid = Helper.DataProcessing(valid_data[:, 0], english_vocab, eos=True).to(device=device)
        english_test = Helper.DataProcessing(test_data[:, 0], english_vocab, eos=True).to(device=device)

        target_train = Helper.DataProcessing(train_data[:, 1], target_vocab, sos=True, eos=True).to(device=device)
        target_valid = Helper.DataProcessing(valid_data[:, 1], target_vocab, sos=True, eos=True).to(device=device)
        target_test = Helper.DataProcessing(test_data[:, 1], target_vocab, sos=True, eos=True).to(device=device)

        # Get the number of training and validation samples
        n_train = english_train.size(0)
        n_valid = english_valid.size(0)

        print(n_train, n_valid)
        batch_size = 32

        # Create TensorDatasets and DataLoaders for training, validation, and test sets
        train_dataset = TensorDataset(english_train, target_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        valid_dataset = TensorDataset(english_valid, target_valid)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = TensorDataset(english_test, target_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, valid_dataloader, test_dataloader
