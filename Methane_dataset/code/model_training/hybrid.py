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

    def forward(self, x, year_index = None):
        lstm_out, _ = self.lstm(x)  # (B, T, H)
        x = lstm_out + self.positional_encoding[:, :x.size(1), :]  # Add PE
        x = self.norm(x)
        residual = x
        x = self.transformer(x)
        x = x + residual  # residual connection
        x = self.dropout(x)
        out = self.fc_out(x).squeeze(-1)  # (B, T)
        return out, out


class Contextual_LSTM_Transformer(nn.Module):
    def __init__(self, config):
        super(Contextual_LSTM_Transformer, self).__init__()
        self.config = config

        # --- 1. 基础模型 (用于处理年内日度数据) ---
        self.base_lstm = nn.LSTM(config.input_size, config.hidden_size, num_layers=3, batch_first=True)

        # --- 2. 全局年份嵌入 (我们的"全局年鉴"或"内存库") ---
        # max_num_years 应该大于你数据集中所有年份的总数
        self.year_embedding = nn.Embedding(num_embeddings=config.max_num_years, embedding_dim=config.hidden_size)

        # --- 3. 上下文查询模块 (注意力机制) ---
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size, 
            num_heads=config.meta_n_heads, # 复用config里的参数
            batch_first=True
        )

        # --- 4. 最终输出层 ---
        # 因为我们将日度特征和上下文特征拼接，所以输入维度*2
        self.fc_out = nn.Linear(config.hidden_size * 2, config.output_size)
        
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, year_indices):
        # x shape: (B, T, H_in) -> (128, 365, 15)
        # year_indices shape: (B) -> (128)
        
        # --- Step 1: 基础模型处理，得到日度特征 ---
        daily_features, _ = self.base_lstm(x) # daily_features shape: (B, T, H)
        daily_features = self.norm(daily_features)

        # --- Step 2: 生成用于查询的向量 ---
        # 我们使用LSTM最后一个隐藏层的状态作为当前批次的"摘要"
        query_vector = self.year_embedding(year_indices) # Shape: (B, H)
        # 为Attention机制增加一个序列长度维度
        query_vector = query_vector.unsqueeze(1) # Shape: (B, 1, H)

        # --- Step 3: 准备全局年鉴作为Key和Value ---
        # 创建一个包含所有年份索引的张量
        all_year_indices = torch.arange(0, self.config.max_num_years).to(self.config.device)
        # 获取所有年份的嵌入向量
        memory_bank = self.year_embedding(all_year_indices) # Shape: (max_years, H)
        # 为Attention机制增加一个batch维度
        memory_bank = memory_bank.unsqueeze(0).expand(x.size(0), -1, -1) # Shape: (B, max_years, H)
        
        # --- Step 4: 查询全局上下文 ---
        # Q: query_vector, K: memory_bank, V: memory_bank
        context_vector, _ = self.attention(
            query=query_vector,
            key=memory_bank,
            value=memory_bank
        ) # context_vector shape: (B, 1, H)

        # --- Step 5: 上下文注入 ---
        # 移除多余的序列维度
        context_vector = context_vector.squeeze(1) # Shape: (B, H)
        # 将上下文向量复制T次(365次)，使其能与日度特征对齐
        context_expanded = context_vector.unsqueeze(1).expand(-1, x.size(1), -1) # Shape: (B, T, H)

        # 拼接日度特征和宏观上下文特征
        enhanced_features = torch.cat([daily_features, context_expanded], dim=2) # Shape: (B, T, 2*H)

        # --- Step 6: 最终预测 ---
        out = self.fc_out(self.dropout(enhanced_features)).squeeze(-1) # (B, T)
        
        # 为了与你的训练循环兼容
        return out, out