import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class AnnualEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.h_output_layer = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, historical_inputs):  # [B, context_len, 365, enc_in]
        B, context_len, seq_len, enc_in = historical_inputs.shape
        historical_inputs = historical_inputs.view(B*context_len, seq_len, enc_in)  # [B*context_len, 365, enc_in]
        _, historical_hidden = self.gru(historical_inputs)

        historical_hidden = torch.cat([historical_hidden[-2], historical_hidden[-1]], dim=1)  # [B*context_len, hidden_dim*2]
        historical_hidden = self.h_output_layer(historical_hidden) # [B*context_len, hidden_dim]
        historical_hidden = historical_hidden.view(B, context_len, -1)  # [B, context_len, hidden_dim]
        return historical_hidden


class DailyDecoder(nn.Module):
    def __init__(self, daily_input_dim, daily_hidden_dim, output_dim=1):
        super(DailyDecoder, self).__init__()
        self.lstm = nn.LSTM(daily_input_dim, daily_hidden_dim, num_layers=3, batch_first=True)

        self.historical_weight = nn.Parameter(torch.randn(1))

        self.head = nn.Sequential(
            nn.Linear(daily_hidden_dim, daily_hidden_dim * 4),
            nn.GELU(),
            nn.Linear(daily_hidden_dim * 4, output_dim)
        )

    def forward(self, x_current, historical_hidden):
        lstm_out, _ = self.lstm(x_current)  # [B, 365, enc_in]
        historical_hidden = historical_hidden.expand_as(lstm_out)  # [B, 365, H]
        lstm_out = (lstm_out + self.historical_weight * historical_hidden) / (1 + torch.abs(self.historical_weight))

        prediction = self.head(lstm_out).squeeze(-1)

        return prediction # prediction:[B, 365]


class LSTM_Embedding(nn.Module):
    def __init__(self, config):
        super(LSTM_Embedding, self).__init__()
        self.config = config
        self.multihead_attn = nn.MultiheadAttention(config.hidden_size, config.num_heads, dropout=config.dropout, batch_first=True)

        self.projection = nn.Linear(config.seq_len * config.enc_in, config.hidden_size)

        self.encoder = AnnualEncoder(config.enc_in, config.hidden_size, config.e_layers)
        self.decoder = DailyDecoder(config.enc_in, config.hidden_size, config.output_size)

    def forward(self, x_current, historical_inputs, index, y):
        '''
        x_current: current year input data [B, 365, enc_in]
        historical_inputs: historical year input data [B, context_len, 365, enc_in]
        '''
        historical_hidden = self.encoder(historical_inputs) # [B, context_len, hidden_size]
        
        B, context_len, seq_len, enc_in = historical_inputs.size()
        q = self.projection(x_current.view(B, 1, -1)) # [B, 1, hidden_size]
        k = self.projection(historical_inputs.view(B, context_len, -1)) # [B, context_len, hidden_size]

        attn_output, attn_weights = self.multihead_attn(
            query=q,
            key=k,
            value=historical_hidden
        ) # attn_output: [B, 1, hidden_size], attn_weights: [B, 1, context_len]

        prediction = self.decoder(x_current, attn_output)
        
        return prediction, prediction