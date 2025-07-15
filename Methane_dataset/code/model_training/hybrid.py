import torch
import torch.nn as nn
    
class LSTM_Transformer(nn.Module):
    def __init__(self, config):
        super(LSTM_Transformer, self).__init__()
        self.lstm = nn.LSTM(config.input_size, config.hidden_size, num_layers=3, batch_first=True)

        # Fixed positional embedding
        pe = torch.zeros(config.seq_len, config.hidden_size)
        position = torch.arange(0, config.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.hidden_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / config.hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=1,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.fc_out = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (B, T, H)
        x = lstm_out + self.positional_encoding[:, :x.size(1), :]  # Add PE
        x = self.norm(x)
        residual = x
        x = self.transformer(x)
        x = x + residual  # residual connection
        x = self.dropout(x)
        out = self.fc_out(x).squeeze(-1)  # (B, T)
        return out, out

