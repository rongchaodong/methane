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
            nhead=4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.fc_out = nn.Linear(config.hidden_size, config.output_size)

    def make_mask(self, src):
        seq_len = src.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.bool()
        return mask

    def forward(self, x, year_index = False):
        lstm_out, _ = self.lstm(x)  # (B, T, H)
        x = lstm_out + self.positional_encoding[:, :x.size(1), :]  # Add PE
        x = self.norm(x)
        residual = x
        mask = self.make_mask(x).to(x.device)
        x = self.transformer(x, mask=mask)
        x = x + residual  # residual connection
        x = self.dropout(x)
        if year_index == True:
            out = x # (B, T, H) --> (128, 365, 8)
        else:
            out = self.fc_out(x).squeeze(-1)  # (B, T)
        return out, out



class Contextual_LSTM_Transformer(nn.Module):
    """
    Improved inter-year dependency: use year embeddings to initialize LSTM hidden state,
    so the base LSTM dynamics adapt per-year without requiring a separate Transformer for fusion.
    """
    def __init__(self, config):
        super(Contextual_LSTM_Transformer, self).__init__()
        self.config = config

        # Embedding for each global year index
        self.year_embedding = nn.Embedding(
            num_embeddings=config.max_num_years,
            embedding_dim=config.hidden_size
        )

        # Base LSTM for daily time series
        self.base_model = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True
        )

        # Final output layer
        self.fc_out = nn.Linear(config.hidden_size, config.output_size)

        # Normalization & dropout
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, year_indices, with_trans: bool = False):
        # x: (B, T, input_size)
        B, T, _ = x.size()

        # Lookup year embedding and project to initialize LSTM states
        year_emb = self.year_embedding(year_indices)        # (B, H)
        # Create initial hidden and cell states: (num_layers, B, H)
        h0 = year_emb.unsqueeze(0).repeat(self.config.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)

        # Base LSTM with custom init
        daily_features, _ = self.base_model(x, (h0.to(x.device), c0.to(x.device)))  # (B, T, H)
        daily_features = self.norm(daily_features)

        # Direct output
        out = self.fc_out(self.dropout(daily_features)).squeeze(-1)  # (B, T)

        return out, out

