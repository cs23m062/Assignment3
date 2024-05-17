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


``` pip install pytorch ```


``` pip install wandb ```


Also ensure anaconda is present, in your system, if not present Download Anaconda [(here)](https://www.anaconda.com/download)

## Running the program:
### FOR PART A
Run the command(Runs in default settings mentioned in table below): 
``` python train_partA.py ```

How to pass arguments:
``` python train_partA.py -e 10 -lr 0.001 -b 32```

#### Available commands
| Name        | Default Value   | Description  |
| --------------------- |-------------| -----|
| -wp --wandb_project | myprojectname	| Project name used to track experiments in Weights & Biases dashboard |
| -we	--wandb_entity| myname | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|-e, --epochs|5|Number of epochs to train neural network.|
|-b, --batch_size|16|Batch size used to train neural network.|
|-o, --optimizer	|Mish|choices: ["Mish", "ReLU", "GELU", "CELU","SiLU","Tanh"]|
|-lr, --learning_rate|0.01|Learning rate used to optimize model parameters|
|-a, --activation|tanh|	choices: ["identity", "sigmoid", "tanh", "ReLU"]|
|-ds,--dense_size|1024|Number of hidden neurons in a fully connected layer|
|-fpl,--filter_per_layers|64|Number of filters to be used per convolution layer|
|-d,--dropout|0|Dropout probability in dense layer|
|-s,--stride|1|length of stride in maxpooling layer|
|-bn,--batchnorm|yes|yes if want to use batch normalization else no|
|-fl,--filter_length|3|length of the filter|
|-fo,--filterorg|same|same will keep same number of filters every layer, double doubles and half halves every layer(maxpool + conv)|
