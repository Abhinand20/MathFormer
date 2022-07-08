#!/usr/bin/env python
# coding=utf-8
"""
This script contains several classes to pre-process and manage the data for training
Running this script will split the dataset into train,val and test files
"""

import os
import numpy as np
from torch.utils.data import Dataset
import re 
from sacred import Experiment
import logging

## To run the script from commandline
ex = Experiment("Split dataset")

logger = logging.getLogger("DatasetLogger")
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname).1s] %(name)s >> "%(message)s"')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")

# attach it to the experiment
ex.logger = logger


## Functions to load pairs into memory
def gen_pair(items):
    """
    Tokenizes each sequence and returns it
    """
    v_ptr = r"sin|cos|tan|\d|\w|\(|\)|\+|-|\*+"
    pair = []
    for curr in items:
        factored,expanded = curr.split('=')
        pair.append([re.findall(v_ptr,factored),re.findall(v_ptr,expanded)])
                
    return pair

def load_pairs(data_path,test_flag=False):
    """
    Loads source and target sequence pairs into memory
    """
    train_path = os.path.join(data_path,'data','train.txt')
    val_path = os.path.join(data_path,'data','validation.txt')
    test_path = os.path.join(data_path,'data','test.txt')

    if test_flag:
        with open(test_path,'r') as f:
            raw_test = f.read().splitlines()
        
        test_pairs = gen_pair(raw_test)

        return test_pairs

    with open(train_path,'r') as f:
        raw_train = f.read().splitlines()
    
    train_pairs = gen_pair(raw_train)

    with open(val_path,'r') as f:
        raw_val = f.read().splitlines()
    
    val_pairs = gen_pair(raw_val)

    return train_pairs,val_pairs
    

class Lang:
    """
    Class to pre-process the text, convert to tensors and store the mappings
    """
    def __init__(self):
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.word2count = {}
        self.word2index = {"<pad>": self.PAD_token, "<sos>": self.SOS_token, "<eos>": self.EOS_token}
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.n_words = 3

    def addSentence(self, sequence):
        """
        Generate mappings for each sentence
        """
        for word in sequence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    
    @classmethod
    def build_vocab_from_pairs(cls, pairs, src_kwargs={}, trg_kwargs={}):
        """The build_vocab_from_pairs() is a factory method - it returns a new instances of the Lang class.
        """
        src_lang = cls(**src_kwargs)
        trg_lang = cls(**trg_kwargs)

        # Read all the data and generate vocabularies
        # Return the class objects for src and target

        # for src, trg in tqdm(pairs, desc="creating vocabs"):
        for src, trg in pairs:
            src_lang.addSentence(src)
            trg_lang.addSentence(trg)

        return src_lang, trg_lang

class GenerateDataset(Dataset):
    """
    Simple dataset class for torch dataloader
    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



@ex.config
def my_config():
    """
    Configs for splitting the dataset
    """
    data_path = os.getcwd()
    split_ratio = 0.80

@ex.capture
def data_split(data_path,split_ratio,_log):
    train_ratio = split_ratio
    test_ratio = val_ratio = (1 - train_ratio)/2
    
    with open(os.path.join(data_path,'dataset.txt'),'r') as f:
        data = f.read().splitlines()

    _log.info("Completed reading the dataset")

    n = len(data)
    np.random.shuffle(data)

    if not os.path.exists(os.path.join(data_path,'data')):
        os.mkdir(os.path.join(data_path,'data'))

    with open(os.path.join(data_path,'data','train.txt'),'w') as f:
        for item in data[:int(n*train_ratio)]:
            f.write('{}\n'.format(item))

    _log.info("Train dataset generated!")

    with open(os.path.join(data_path,'data','validation.txt'),'w') as f:
        for item in data[int(n*train_ratio):int(n*train_ratio + n*val_ratio)]:
            f.write('{}\n'.format(item))

    _log.info("Validation dataset generated!") 

    with open(os.path.join(data_path,'data','test.txt'),'w') as f:
        for item in data[int(n*train_ratio + n*val_ratio):]:
            f.write('{}\n'.format(item))

    _log.info("Test dataset generated!") 
    

@ex.automain
def run_data(_log):
    _log.info("Started splitting the dataset")
    data_split()
    _log.info("Completed splitting the dataset")