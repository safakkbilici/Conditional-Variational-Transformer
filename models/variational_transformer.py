import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers import *
from models.encoder_decoder import *

class Transformer(nn.Module):
    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj', enc_max_seq_len = 512, latent_size = 16):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)
        
        self.enc_max_seq_len = enc_max_seq_len
        self.vae_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(),
        )

        self.vae_decoder = nn.Sequential(
            nn.Linear(latent_size, d_model // 2),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(d_model // 2, d_model),
            nn.GELU(),
            nn.Dropout(),
        )
        
        self.context2mean = nn.Linear(d_model // 2, latent_size)
        self.context2std = nn.Linear(d_model // 2, latent_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq, prior, debug = False):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
        if debug:
            print(f"src_seq shape: {src_seq.shape}")
            print(f"trg_seq shape: {trg_seq.shape}")
            print(f"src_mask shape: {src_mask.shape}")
            print(f"trg_mask shape: {trg_mask.shape}")
            print(f"prior shape: {prior.shape}")
            

        enc_output, *_ = self.encoder(src_seq, src_mask)
        if debug:
            print(f"enc_output shape: {enc_output}")
        #print(enc_output.shape)
        cls_token = enc_output[:,0,:]
        latent = self.vae_encoder(cls_token)
        mean = self.context2mean(latent)
        std = self.context2std(latent)
        z = prior * torch.exp(0.5 * std) + mean
        to_dec = self.vae_decoder(z)
        
        to_dec = to_dec.repeat(1,self.enc_max_seq_len).reshape(src_seq.size(0), self.enc_max_seq_len, -1)
        
        KL_loss = (-0.5 * torch.sum(1 + std - mean.pow(2) - std.exp())) / src_seq.size(0)
        #print(to_dec.shape)
        
        
        dec_output, *_ = self.decoder(trg_seq, trg_mask, to_dec, src_mask = None) #enc_output
        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2)), KL_loss
    
    def generate(self, trg_seq, trg_mask, prior, src_mask = None):
        to_dec = self.vae_decoder(prior)
        to_dec = to_dec.repeat(prior.size(0),self.enc_max_seq_len).reshape(prior.size(0), self.enc_max_seq_len, -1)
        #print(to_dec.shape)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, to_dec, src_mask = None)
        return dec_output
