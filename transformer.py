"""
Contains a class that combines encoder and decoder to implement the transformer architecture
Seq2Seq class also contains methods for generating predictions
"""

from tqdm import tqdm
from backbone import Encoder,Decoder
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader
from data import GenerateDataset
import torch
import utils
import numpy as np

class Seq2Seq(pl.LightningModule):
    def __init__(
        self,
        src_lang,
        trg_lang,
        device,
        batch_size,
        max_len=32,
        hid_dim=256,
        enc_layers=3,
        dec_layers=3,
        enc_heads=8,
        dec_heads=8,
        enc_pf_dim=512,
        dec_pf_dim=512,
        enc_dropout=0.1,
        dec_dropout=0.1,
        lr=0.0005,
        **kwargs,  # throwaway
    ):
        super().__init__()

        self.save_hyperparameters()
        del self.hparams["src_lang"]
        del self.hparams["trg_lang"]

        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.batch_size = batch_size

        self.encoder = Encoder(
            src_lang.n_words,
            hid_dim,
            enc_layers,
            enc_heads,
            enc_pf_dim,
            enc_dropout,
            device,
        )

        self.decoder = Decoder(
            trg_lang.n_words,
            hid_dim,
            dec_layers,
            dec_heads,
            dec_pf_dim,
            dec_dropout,
            device,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_lang.PAD_token)
        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        self.initialize_weights()
        self.to(device)

    def initialize_weights(self):
        def _initialize_weights(m):
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        self.encoder.apply(_initialize_weights)
        self.decoder.apply(_initialize_weights)

    def make_src_mask(self, src):

        # src = [batch size, src len]

        src_mask = (src != self.src_lang.PAD_token).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_lang.PAD_token).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len)).type_as(trg)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):

        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention

    def predict(self,pred_tensors):
        collate_func = utils.Collate_Pad(self.src_lang,self.trg_lang,True)
        pred_dataloader = DataLoader(
            GenerateDataset(pred_tensors),
            batch_size=self.batch_size,
            collate_fn=collate_func
        )
        sentences = []
        words = []
        attention = []
        for batch in tqdm(pred_dataloader, desc="predict batch num"):
            preds = self.expand_batch(batch.to(self.device))
            pred_sentences, pred_words, pred_attention = preds
            sentences.extend(pred_sentences)
            words.extend(pred_words)
            attention.extend(pred_attention)

        # sentences = [num pred sentences]
        # words = [num pred sentences, trg len]
        # attention = [num pred sentences, n heads, trg len, src len]

        return sentences, words, attention

    def expand_batch(self,batch,max_len = 32):
        
        self.eval()
            
        src_tensor = batch # Batch
        src_mask = self.make_src_mask(src_tensor)
        
        with torch.no_grad():
            enc_src = self.encoder(src_tensor, src_mask)

        trg_indexes = [[self.trg_lang.SOS_token] for _ in range(len(batch))] # Batch
        
        trg_tensor = torch.LongTensor(trg_indexes).to(self.device)

        for i in range(max_len):

            trg_mask = self.make_trg_mask(trg_tensor)
            
            with torch.no_grad():
                output, attention = self.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            
            # output = [batch size, cur trg len, output dim]

            pred_token = output.argmax(2)[:,-1].reshape(-1, 1) # Batch
            # pred_token = [batch_size, 1]

            trg_tensor = torch.cat((trg_tensor, pred_token), dim=-1)

            # trg_tensor = [batch_size, cur trg len], cur trg len increased by 1
        
        src_tensor = src_tensor.detach().cpu().numpy()
        trg_tensor = trg_tensor.detach().cpu().numpy()
        attention = attention.detach().cpu().numpy()

        pred_words = []
        pred_sentences = []
        pred_attention = []
        for src_indexes, trg_indexes, attn in zip(src_tensor, trg_tensor, attention):
            # trg_indexes = [trg len = max len (filled with eos if max len not needed)]
            # src_indexes = [src len = len of longest sentence (padded if not longest)]
            # indexes where first eos tokens appear
            src_eosi = np.where(src_indexes == self.src_lang.EOS_token)[0][0]
            _trg_eosi_arr = np.where(trg_indexes == self.trg_lang.EOS_token)[0]
            if len(_trg_eosi_arr) > 0:  # check that an eos token exists in trg
                trg_eosi = _trg_eosi_arr[0]
            else:
                trg_eosi = len(trg_indexes)

            # cut target indexes up to first eos token and also exclude sos token
            trg_indexes = trg_indexes[1:trg_eosi]
            # attn = [n heads, trg len=max len, src len=max len of sentence in batch]
            # we want to keep n heads, but we'll cut trg len and src len up to
            # their first eos token
            attn = attn[:, :trg_eosi, :src_eosi]  # cut attention for trg eos tokens

        
            words = [self.trg_lang.index2word[index] for index in trg_indexes]
            
            sentence = "".join(words)
            pred_words.append(words)
            pred_sentences.append(sentence)
            pred_attention.append(attn)

        # pred_sentences = [batch_size]
        # pred_words = [batch_size, trg len]
        # attention = [batch size, n heads, trg len (varies), src len (varies)]

        return pred_sentences, pred_words, pred_attention