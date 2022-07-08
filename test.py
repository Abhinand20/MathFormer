#!/usr/bin/env python
# coding=utf-8

"""
Evaluating the model on test.txt
1) Load the trained model from ./model
2) Calculate and report the loss on test data
3) Generate batch predictions for all test data (Much faster than predicting one by one)
4) Store the predictions in an output file for documentation
5) Do a strict string equality match between predictions and actual sequences
6) Report the accuracy using the formula: accuracy = (correct predictions / total test samples)*100
"""

# Imports
import os
import pickle
from sacred import Experiment
import tqdm as tqdm
import logging
import utils
from transformer import Seq2Seq
from train import get_model
from data import load_pairs
import torch

### To run the script from commandline
ex = Experiment("Test Model")

logger = logging.getLogger("TestLogger")
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname).1s] %(name)s >> "%(message)s"')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")

# attach it to the experiment
ex.logger = logger

def test_pred_accuracy(model,test_data):
    """
    Generates predictions, compares with the ground truth values and reports the accuracy
    """
    prediction_tensors = []
    labels = []
    correct_count = 0
    total_count = len(test_data)

    for pair in test_data:
        prediction_tensors.append(utils.sentence_to_tensor(pair[0],model.src_lang))
        labels.append("".join(pair[1]))
    
    preds,_,_ = model.predict(prediction_tensors)
    for pred,actual in zip(preds,labels):
        if pred == actual:
            correct_count += 1
    
    accuracy = correct_count / total_count
    return accuracy,preds

@ex.config
def test_config():
    device = utils.get_device()
    model_path = os.path.join(os.getcwd(),'model','best_model.pt')
    data_path = os.getcwd()
    dirpath = os.getcwd()
    model_params = {
        'max_len':32,
        'hid_dim':256,
        'enc_layers':3,
        'dec_layers':3,
        'enc_heads':8,
        'dec_heads':8,
        'enc_pf_dim':512,
        'dec_pf_dim':512,
        'enc_dropout':0.1,
        'dec_dropout':0.1,
        }
    hyperparameters = {
        'batch_size': 512,
        'n_iters' : 20,
        'gradient_clip' : 1,
        'learning_rate' : 0.0005
    }

@ex.capture
def load_model(dirpath,model_params,hyperparameters,device,model_path,_log):
    _log.info("Loading source vocabulary")
    with open(os.path.join(dirpath,'vocab',"src_lang.pickle"), "rb") as fi:
        src_lang = pickle.load(fi)
    _log.info("Finished Loading source vocabulary!")
    _log.info("Loading Target vocabulary")
    with open(os.path.join(dirpath,'vocab', "trg_lang.pickle"), "rb") as fi:
        trg_lang = pickle.load(fi)
    _log.info("Finished Loading Target vocabulary!")
    model = get_model(src_lang,trg_lang,model_params,device,hyperparameters) 
    model.load_state_dict(torch.load(model_path,map_location=device))
    return model

@ex.automain
def run_test(data_path,_log):

    _log.info("Loading the trained model!")
    model = load_model()
    _log.info("Finished Loading the trained model!")
    
    _log.info("Loading the test data!")
    test_pairs = load_pairs(data_path,test_flag=True)
    _log.info("Finished loading the test data!")
    
    _log.info("Started batch predictions on test data!")
    accuracy,predictions = test_pred_accuracy(model,test_pairs)
    _log.info("Finished batch predictions on test data!")
    
    print("The model accuracy on test data = {}%".format(accuracy*100))
    if not os.path.exists(os.path.join(os.getcwd(),'output')):
        os.mkdir(os.path.join(data_path,'output'))
    
    with open(os.path.join(data_path,'output','predictions.txt'),'w') as f:
        for pred in predictions:
            f.write(pred + '\n')
    
    _log.info("Finished saving predictions!")