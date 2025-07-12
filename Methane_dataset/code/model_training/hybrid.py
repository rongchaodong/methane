import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import Tuple

class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM, self).__init__()
        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers
        self.lstm = nn.LSTM(configs.input_size, configs.hidden_size, configs.num_layers, batch_first=True)
        self.fc = nn.Linear(configs.hidden_size, configs.output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)  
        final_out = self.fc(out)  
        return out, final_out.squeeze()

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(config.input_size, config.hidden_size)  
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=4), 
            num_layers=config.num_layers
        )
        self.fc_out = nn.Linear(config.hidden_size, config.output_size) 

    def forward(self, src):
        # print(src.shape)
        src_emb = self.embedding(src) 
        transformer_out = self.transformer(src_emb)
        output = self.fc_out(transformer_out) 
        # print(output.shape) 
        
        return transformer_out, output.squeeze(2)
    
class LSTM_Transformer(nn.Module):
    def __init__(self, config):
        super(LSTM_Transformer, self).__init__()
        self.lstm = nn.LSTM(config.input_size, config.hidden_size, config.num_layers, batch_first=True)

        # positional embedding
        self.positional_embedding = nn.Parameter(torch.randn(1, config.seq_len, config.hidden_size))

        self.trans_encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.trans_encoder_layer, num_layers=config.num_layers)

        self.fc_out = nn.Linear(config.hidden_size, config.output_size) 
    
    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        lstm_out = lstm_out + self.positional_embedding
        transformer_out = self.transformer(lstm_out)
        output = self.fc_out(transformer_out)
        return output, output.squeeze(-1)
