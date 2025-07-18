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

    def forward(self, x, year_index = False):
        lstm_out, _ = self.lstm(x)  # (B, T, H)
        x = lstm_out + self.positional_encoding[:, :x.size(1), :]  # Add PE
        x = self.norm(x)
        residual = x
        x = self.transformer(x)
        x = x + residual  # residual connection
        x = self.dropout(x)
        if year_index == True:
            out = x # (B, T, H) --> (128, 365, 8)
        else:
            out = self.fc_out(x).squeeze(-1)  # (B, T)
        return out, out


class Contextual_LSTM_Transformer(nn.Module):
    def __init__(self, config):
        super(Contextual_LSTM_Transformer, self).__init__()
        self.config = config

        # self.base_lstm = nn.LSTM(config.input_size, config.hidden_size, num_layers=3, batch_first=True)
        self.base_model = LSTM_Transformer(config)

        # max_num_years 
        self.year_embedding = nn.Embedding(num_embeddings=config.max_num_years, embedding_dim=config.hidden_size)
        self.context_tokens = nn.Parameter(torch.randn(4, config.hidden_size))  # (M, H)

        # attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size, 
            num_heads=1,
            batch_first=True
        )

        self.fusion_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=1,
                batch_first=True,
                norm_first=True
            ),
            num_layers=1
        )

        self.fc_out = nn.Linear(config.hidden_size *2, config.output_size)
        self.fc_out_trans = nn.Linear(config.hidden_size, config.output_size)
        
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, year_indices, with_trans: bool = False):
        # x shape: (B, T, H_in) -> (128, 365, 15)
        # year_indices shape: (B) -> (128)
        
        daily_features, _ = self.base_model(x, True) # daily_features shape: (B, T, H)
        daily_features = self.norm(daily_features)

        # query_vector = self.year_embedding(year_indices) # Shape: (B, H)
        # query_vector = query_vector.unsqueeze(1) # Shape: (B, 1, H)

        # all_year_indices = torch.arange(0, self.config.max_num_years).to(self.config.device)
        # memory_bank = self.year_embedding(all_year_indices) # Shape: (max_years, H)
        # memory_bank = memory_bank.unsqueeze(0).expand(x.size(0), -1, -1) # Shape: (B, max_years, H)
        # Q: query_vector, K: memory_bank, V: memory_bank
        # context_vector, _ = self.attention(
        #     query=query_vector,
        #     key=memory_bank,
        #     value=memory_bank
        # ) # context_vector shape: (B, 1, H)

        query_vector = daily_features.mean(dim=1, keepdim=True)  # (B, 1, H)
        context = self.context_tokens.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, M, H)
        context_vector, _ = self.attention(query=query_vector, key=context, value=context)

        context_expanded = context_vector.expand(-1, x.size(1), -1) # Shape: (B, T, H)

        enhanced_features = torch.cat([daily_features, context_expanded], dim=2) # Shape: (B, T, 2*H)

        if with_trans:
            enhanced_features = self.fusion_proj(enhanced_features)
            enhanced_features = self.fusion_transformer(enhanced_features)
            out = self.fc_out_trans(self.dropout(enhanced_features)).squeeze(-1) # (B, T)
        else:
            out = self.fc_out(self.dropout(enhanced_features)).squeeze(-1) # (B, T)
        
        return out, out