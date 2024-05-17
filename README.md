## General Instructions :
1. If running on a local host: Ensure Python is present in your system and also see if these libraries are present in your system
   - pytorch lightning [(lightning)](https://lightning.ai/docs/pytorch/stable/)
   - weights and biases [(wandb)](https://docs.wandb.ai/?_gl=1*1lup0xs*_ga*NzgyNDk5ODQuMTcwNTU4MzMwNw..*_ga_JH1SJHJQXJ*MTcxMDY3NjQ2MS43Ny4xLjE3MTA2NzY0NjQuNTcuMC4w)
   - scikit-learn [(sklearn)](https://scikit-learn.org/stable/)
   - [matplotlib](https://matplotlib.org/)
3. If running on colab/kaggle ignore point 1.
4. If running on local host ensure CUDA is present in system else install anaconda, it provides a virtual environment for your codes to run, for fast execution time use either NVIDIA GPU's or use Kaggle.
5. Ensure you have pasted the paths to the inaturalist dataset in the Dataloader code
6. There is only 1file this time so no worries.

follow this guide to install Python in your system:
1. Windows: https://kinsta.com/knowledgebase/install-python/#windows
2. Linux: https://kinsta.com/knowledgebase/install-python/#linux
3. MacOS: https://kinsta.com/knowledgebase/install-python/#mac

### ENSURE PYTORCH LIGHTNING IS PRESENT IN YOUR SYSTEM
if the libraries are not present just run the command:


``` python
pip install lightning
```


``` python
pip install pytorch
```


```python
pip install wandb
```


Also ensure anaconda is present, in your system, if not present Download Anaconda [(here)](https://www.anaconda.com/download)

## Running the program:
Run the command(Runs in default settings mentioned in table below): 
``` python
python train_partA.py
```

How to pass arguments:
``` python
python Attentiontrain.py -e 10 -b 32 -lr 0.001 -t hin -ct LSTM -em 128 -hi 512 -el 4 -dl 4 -dr 0.2 -bi True -op Adam
```

#### Available commands
| Name        | Default Value   | Description  |
| --------------------- |-------------| -----|
| -wp --wandb_project | myprojectname	| Project name used to track experiments in Weights & Biases dashboard |
| -we	--wandb_entity| myname | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|-e, --epochs|5|Number of epochs to train neural network.|
|-b, --batch_size|16|Batch size used to train neural network.|
|-op, --optimizer	|Adam|choices: ["Sgd","Adam","Nadam"]|
|-lr, --learning_rate|0.001|Learning rate used to optimize model parameters|
|-t,--target_lang|hin|	Target Language in which transliteration system works, choices: ["hin", "ben", "telugu"]|
|-ct,--cell_type|LSTM|Type of cell to be used in architecture Choose b/w [LSTM,RNN,GRU]|
|-em,--embedding_size|128|size of embedding to be used in encoder decoder|
|-hi,--hidden_size|512|Hidden layer size of encoder and decoder|
|-el,--encoder_layers|4|Number of hidden layers in encoder|
|-dl,--decoder_layers|4|Number of hidden layers in decoder|
|-dr,--dropout|0.2|dropout probability|
|-bi,--bidirectional|True|Whether you want the data to be read from both directions|
