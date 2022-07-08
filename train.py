#!/usr/bin/env python
# coding=utf-8
"""
Train the model on train.txt and evaluate on val.txt
1) Read the train data and validation data
2) Set-up the model parameters and hyperparameters
3) Call training module and save the best model for inference
"""

from sacred import Experiment
import logging
import os
from collections import Counter
import re
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl
from backbone import Encoder,Decoder
import numpy as np
import random
import time
import os
import utils
from data import Lang,GenerateDataset,load_pairs
from transformer import Seq2Seq
import math
import pickle

## To run the script from commandline
ex = Experiment("Train Model")

logger = logging.getLogger("TrainLogger")
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname).1s] %(name)s >> "%(message)s"')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")

# attach it to the experiment
ex.logger = logger


def trainer(model, iterator, optimizer, criterion, clip, device):
    """
    Contains all the training steps to be performed on a batch of training data
    """
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch[0].to(device)
        trg = batch[1].to(device)
        
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion,device):
    """
    Contains all the steps to be performed to evaluate the model on validation data
    """
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch[0].to(device)
            trg = batch[1].to(device)

            output, _ = model(src, trg[:,:-1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

@ex.config
def train_config():
    """
    Configs for splitting the dataset
    """
    device = utils.get_device()
    seed = 1234
    data_path = os.getcwd()
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
    batch_size = hyperparameters['batch_size']

@ex.capture
def get_model(src_lang,trg_lang,model_params,device,hyperparameters):
    return Seq2Seq(
        src_lang=src_lang,
        trg_lang=trg_lang,
        device=device,
        batch_size=hyperparameters['batch_size'],
        max_len=model_params['max_len'],
        hid_dim=model_params['hid_dim'],
        enc_layers=model_params['enc_layers'],
        dec_layers=model_params['dec_layers'],
        enc_heads=model_params['enc_heads'],
        dec_heads=model_params['dec_heads'],
        enc_pf_dim=model_params['enc_pf_dim'],
        dec_pf_dim=model_params['dec_pf_dim'],
        enc_dropout=model_params['enc_dropout'],
        dec_dropout=model_params['dec_dropout'],
        lr=hyperparameters['learning_rate'])

@ex.capture
def train_model(model,train_dataloader,val_dataloader,hyperparameters,device,_log,data_path):
    """
    Generates batches and runs the train function for each batch
    """
    all_train_losses = []
    all_val_losses = []

    N_EPOCHS = hyperparameters['n_iters']
    CLIP = hyperparameters['gradient_clip']

    best_valid_loss = float('inf')

    for epoch in tqdm(range(N_EPOCHS)):
        
        start_time = time.time()
        
        train_loss = trainer(model, train_dataloader, model.optimizer, model.criterion, CLIP,device)
        valid_loss = evaluate(model, val_dataloader, model.criterion,device)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if not os.path.exists(os.path.join(data_path,'model')):
                os.mkdir(os.path.join(data_path,'model'))
            torch.save(model.state_dict(), os.path.join(data_path,'model','best_model.pt'))
        
        _log.info("Epoch: {} | Time: {}m {}s".format(str(epoch+1),epoch_mins,epoch_secs))
        _log.info('\tTrain Loss: {}'.format(round(train_loss,3)))
        _log.info('\Validation Loss: {}'.format(round(valid_loss,3)))

        all_train_losses.append(train_loss)
        all_val_losses.append(valid_loss)
    
    return all_train_losses,all_val_losses

@ex.automain
def run_train(seed,data_path,batch_size,_log):
    utils.set_seed(seed)

    _log.info("Started loading data")
    train_pairs,val_pairs = load_pairs(data_path,test_flag=False)
    _log.info("Completed loading data")
    language = Lang()
    # Generating vocabulary
    src_lang, trg_lang = language.build_vocab_from_pairs(train_pairs)
    _log.info("Completed building vocabulary")

    _log.info("Storing the vocabulary for later use")
    if not os.path.exists(os.path.join(os.getcwd(),'vocab')):
        os.mkdir(os.path.join(data_path,'vocab'))

    save_to_pickle = {
        "src_lang.pickle": src_lang,
        "trg_lang.pickle": trg_lang,
    }
    for k, v in save_to_pickle.items():
        with open(os.path.join(os.getcwd(),'vocab', k), "wb") as fo:
            pickle.dump(v, fo)
    _log.info("Vocabulary saved!")
    
    # Generating tensors for training
    _log.info("Processing train data")
    train_tensors = utils.pairs_to_tensors(train_pairs,src_lang,trg_lang) # (train_data.shape[0],sequence vector)
    _log.info("Processing validation data")
    val_tensors = utils.pairs_to_tensors(val_pairs,src_lang,trg_lang)

    _log.info("Setting up batches for training")
    collate_func = utils.Collate_Pad(src_lang,trg_lang,predict=False)
    train_dataloader = DataLoader(
            GenerateDataset(train_tensors),
            batch_size=batch_size,
            collate_fn=collate_func
        )

    val_dataloader = DataLoader(
            GenerateDataset(val_tensors),
            batch_size=batch_size,
            collate_fn=collate_func
        )
    _log.info("Completed setting up batches for training")
    
    _log.info("Setting up the model for training")
    model = get_model(src_lang=src_lang,trg_lang=trg_lang)
    num_model_params = utils.count_parameters(model)
    _log.info("Model with {} trainable parameters initialized!".format(num_model_params))

    _log.info("Started model training...")
    train_losses, val_losses = train_model(model,train_dataloader,val_dataloader)
    _log.info("Model training completed, model saved")

    utils.plot_loss(train_losses,val_losses)