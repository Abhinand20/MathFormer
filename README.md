# MathFormer - Solve math equations using NLP and transformers!

## Challenge
Implement a deep neural network model that learns to expand single variable polynomials. Model input is factorized sequence and output is predicted expanded sequence. 

* `(7-3*z)*(-5*z-9)=15*z**2-8*z-63`
* `(7-3*z)*(-5*z-9)` is the factorized input
* `15*z**2-8*z-63`  is the expanded target

For the expanded form, only the form provided is considered as correct.

## Solution
* The directory `./data` contains `train.txt`, `validation.txt` and `test.txt`
* The source and target sequence vocabulary is stored in the directory `./vocab` 
* The trained model (`best_model.pt`) is present in the directory `./model`
* All predictions made by the model on the test is stored in the file `./output/predictions.txt`
* Summary for the model and it's trainable parameters is stored in `network.txt`
* The classes for the transformer model are in - `backbone.py` and `transformer.py`
* `data.py` splits the dataset into train,val and train datasets randomly based on the input split ratio (already split dataset is provided in the repo)
* `train.py` trains the model using the defined configurations
* `test.py` runs the trained model on the test data to generate predictions and calculates the accuracy
* `text_EDA.ipynb` contains the preliminary exploratory data analysis of the dataset
* `requirements.txt` contains dependencies


## Suggested setup for running the code -
Model was trained on a single NVIDIA RTX 3090 GPU with CUDA 10.2 and torch == 1.11.0 (You might have to change the torch version depending upon the GPU and CUDA version of your machine)

- Set up a new conda virtual environment 
```shell
conda create --name <env_name> python=3.9.2
```

- Activate the environment
```shell
conda activate <env_name>
```

- Install the dependencies (Run this command in the `/Attention` directory)
```shell
pip install -r requirements.txt
```
## Commands to train the model (To only evaluate the model on test.txt, these steps can be skipped)
This solution uses the [sacred](https://sacred.readthedocs.io/en/stable/quickstart.html#) library for logging, running, configuring and organizing the code.

All the commands should be run only from the parent directory (i.e. `/Attention`)

1) Split data into train, val and test set:
```shell
python data.py with 'split_ratio=0.8'
```

2) Train the model (All configurations can be observed in `train.py` and they can also be passed from command line as shown):
```shell
python train.py with 'hyperparameters.n_iters=20'
```

## Commands to test the model
Evaluate model on test set (This will utilize GPU if available):
```shell
python test.py
```
Evaluate model on test set (Using CPU only):
```shell
python test.py with 'device="cpu"'
```

## Model Accuracy
The model is evaluated against a **strict equality** between the predicted target sequence and the groud truth target sequence of the test dataset. The model achieved an accuracy of `98.63%` (trained for 20 epochs for 45 minutes on a single GPU).

For a more comprehensive description of the solution, parameter choices and loss plots, please refer to the file `Solution-Report.docx`
