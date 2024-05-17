# Building a Transliteration system from scratch using Pytorch
In this assignment, you will experiment with a sample of the Aksharantar dataset released by AI4Bharat. This dataset contains pairs of the following form: 
x,y
ajanabee,अजनबी 
i.e., a word in the native script and its corresponding transliteration in the Latin script (how we type while chatting with our friends on WhatsApp etc). Given many such $(x_i, y_i)_{i=1}^n$ pairs your goal is to train a model $y=\hat{f}(x)$ which takes as input a romanized string (ghar) and produces the corresponding word in Devanagari (घर). 
As you would realize, this is the problem of mapping a sequence of characters in one language to a sequence of characters in another. Notice that this is a scaled-down version of the problem of translation where the goal is to translate a sequence of words in one language to a sequence of words in another language (as opposed to a sequence of characters here).


## General Instructions :
1. If running on a local host: Ensure Python is present in your system and also see if these libraries are present in your system
   - Python torch ([pytorch](https://pytorch.org/docs/stable/index.html))
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

### ENSURE PYTORCH IS PRESENT IN YOUR SYSTEM
if the libraries are not present just run the command:


``` python
pip install pytorch
```


```python
pip install wandb
```


Also ensure anaconda is present, in your system, if not present Download Anaconda [(here)](https://www.anaconda.com/download)

## Running the program:
=> Attentiontrain.py is present inside the Attention folder(It is insisted to run this model as it is better optimised according to specifications)
Download it
and update the path variable with the relative path of your dataset

Imp : 
##### Do not give relative path like this :
/content/drive/MyDrive/aksharantar_sampled/hin

##### Example of how to give the correct path is like this:
```python
/content/drive/MyDrive/aksharantar_sampled/
```

Run the command(Runs in default settings mentioned in table below): 
``` python
python Attentiontrain.py
```

``` python
python vanilla.py
```

How to pass arguments:
``` python
python Attentiontrain.py -e 10 -b 32 -lr 0.001 -t hin -ct LSTM -em 128 -hi 512 -el 4 -dl 4 -dr 0.2 -bi True -op Adam
```

``` python
python vanilla.py -e 10 -b 32 -lr 0.001 -t hin -ct LSTM -em 128 -hi 512 -el 4 -dl 4 -dr 0.2 -bi True -op Adam
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
