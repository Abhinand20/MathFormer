"""
Auxilliary helper functions
"""

from tqdm import tqdm
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def sentence_to_tensor(sequence, language):
    indexes = [language.word2index[w] for w in sequence]
    indexes = [language.SOS_token] + indexes + [language.EOS_token]
    return torch.LongTensor(indexes)


def pairs_to_tensors(pairs, src_lang, trg_lang):
    tensors = [
        (sentence_to_tensor(src, src_lang), sentence_to_tensor(trg, trg_lang))
        for src, trg in tqdm(pairs, desc="creating tensors")
    ]
    return tensors

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def plot_loss(train_losses,val_losses):
    x_ax = [i for i in range(1,len(train_losses) + 1)]
    plt.plot(x_ax,train_losses,'-o')
    plt.plot(x_ax,val_losses,'-o')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Loss')
    plt.savefig('loss.png',dpi=300,bbox_inches='tight')



class Collate_Pad:
    """
    Custom collater to handle unequal length tensors in the torch Data Loader
    """
    def __init__(self, src_lang, trg_lang=None, predict=False):
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.predict = predict

    def __call__(self, batch):
        # TODO: try pack_padded_sequence for faster processing
        
        if self.predict:
            # batch = src_tensors in predict mode
            return nn.utils.rnn.pad_sequence(
                batch, batch_first=True, padding_value=self.src_lang.PAD_token
            )

        src_tensors, trg_tensors = zip(*batch)
        # print("Before pad -",trg_tensors)
        src_tensors = nn.utils.rnn.pad_sequence(
            src_tensors, batch_first=True, padding_value=self.src_lang.PAD_token
        )
        
        trg_tensors = nn.utils.rnn.pad_sequence(
            trg_tensors, batch_first=True, padding_value=self.trg_lang.PAD_token
        )
        # print("After pad -",trg_tensors)
        return src_tensors, trg_tensors