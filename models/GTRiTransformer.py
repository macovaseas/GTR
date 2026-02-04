import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class GTR(nn.Module):
    def __init__(self, d_series, period_len=24):
        
        super(GTR, self).__init__()
        self.agg = False
        self.period_len = period_len
        self.linear = nn.Linear(d_series, d_series)
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 1 + 2 * (self.period_len // 2)),
                                stride=1, padding=(0, self.period_len // 2), padding_mode="zeros", bias=False)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, q):
        _, C, S = x.shape
        # Step 1: Mapping
        global_query = self.linear(q)  # (B, C, S)

        # Step 2: 融合
        out = torch.stack([x, global_query], dim=2) # (B, C, 2, S)
        out = out.reshape(-1, 1, 2, S)  # (B*C, 1, 2, S)
        conv_out = self.conv2d(out)  # (B*C, 1, 1, S)
        conv_out = conv_out.reshape(-1, C, S)  # (B, C, S)

        return self.dropout(conv_out)
    

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_revin
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        
        ## GTR part
        self.cycle_len = configs.cycle
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.Q = torch.nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)
        self.GTR = GTR(d_series=self.seq_len)

    def forecast(self, x_enc, cycle_index, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Applying GTR module
        x_enc = x_enc.permute(0, 2, 1)
        gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
        query_input = self.Q[gather_index].permute(0, 2, 1)  # (b, c, s)
        global_information = self.GTR(x_enc, query_input)
        x_enc = x_enc + global_information
        x_enc = x_enc.permute(0, 2, 1)

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, cycle_index, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, cycle_index, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]