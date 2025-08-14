import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from config import Config

config = Config()

class HierarchicalTemporalDataset(Dataset):
    """
    1. x_daily: fast features in data: (B, T=365, K_fast=5), has 5 features
    2. x_yearly: slow features: (B, num_context_years, K_slow=10), 10 features
    3. y_target: the y_batch shaped in (B, T=365)
    """
    def __init__(self, target_years, config, all_available_years_range, data_source:str):
        self.target_years = list(target_years)
        self.config = config
        self.all_available_years = list(all_available_years_range)
        
        self.K_slow = 10 
        self.K_fast = 5 

        if data_source == "full":
            self.input_files = {year: f"../../processed_data/TEM-MDM/full/temporal/input_{year}.npy" for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/full/temporal/output_{year}.npy" for year in self.target_years}
        elif data_source == "seq":
            self.input_files = {year: f"../../processed_data/TEM-MDM/sequence/temporal/input_{year}.npy" for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/sequence/temporal/output_{year}.npy" for year in self.target_years}
        elif data_source == "sparse":
            self.input_files = {year: f"../../processed_data/TEM-MDM/sparse/temporal/input_{year}.npy" for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/sparse/temporal/output_{year}.npy" for year in self.target_years}
        elif data_source == "1percent":
            self.input_files = {year: f"../../processed_data/TEM-MDM/1percent/temporal/input_{year}.npy" for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/1percent/temporal/output_{year}.npy" for year in self.target_years}

        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')

        self.yearly_features_cache = self._precompute_yearly_features()

        self.data_lengths = [np.load(self.input_files[year], mmap_mode='r').shape[0] for year in self.target_years]
        self.cumsum_lengths = np.cumsum([0] + self.data_lengths)

    def _precompute_yearly_features(self):
        cache = {}
        mean_slow = self.mean_vals[:self.K_slow]
        std_slow = self.std_vals[:self.K_slow]

        for year in self.all_available_years:
            full_year_data = np.load(self.input_files[year], mmap_mode='r')
            slow_features_for_year = full_year_data[:, 0, :self.K_slow]
            slow_features_for_year = torch.tensor(slow_features_for_year, dtype=torch.float32)
            
            normalized_slow_features = (slow_features_for_year - mean_slow) / std_slow
            cache[year] = normalized_slow_features.detach()
        return cache

    def __len__(self):
        return self.cumsum_lengths[-1]

    def __getitem__(self, idx):
        year_list_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        file_idx = idx - self.cumsum_lengths[year_list_idx]
        actual_year = self.target_years[year_list_idx]

        full_input_sample = np.load(self.input_files[actual_year], mmap_mode='r')[file_idx]
        normalized_input = (full_input_sample - self.mean_vals) / self.std_vals
        x_daily = normalized_input[:, self.K_slow:]
        
        y_target = np.load(self.output_files[actual_year], mmap_mode='r')[file_idx] / 23

        context_vectors = []
        for i in range(1, self.config.num_context_years + 1):
            context_year = actual_year - i
            
            if context_year in self.yearly_features_cache:
                vec = self.yearly_features_cache[context_year][file_idx]
            else:
                earliest_year = self.all_available_years[0]
                vec = self.yearly_features_cache[earliest_year][file_idx]
            context_vectors.append(vec)
        
        x_yearly = torch.stack(context_vectors)

        return (
            torch.tensor(x_daily, dtype=torch.float32).to(config.device),
            x_yearly.to(config.device),
            torch.tensor(y_target, dtype=torch.float32).to(config.device)
        )


class MultiYearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.year_encoders = nn.ModuleList([
            nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
            for _ in range(10)
        ])
        
    def forward(self, x_yearly):  # [B, context_len, K_slow]
        B, context_len, K_slow = x_yearly.shape
        year_embeddings = []
        
        for i in range(context_len):
            year_data = x_yearly[:, i:i+1, :]  # [B, 1, K_slow]
            encoder_idx = min(i, len(self.year_encoders) - 1) 
            
            gru_out, _ = self.year_encoders[encoder_idx](year_data)  # [B, 1, 2H]
            year_embedding = gru_out.squeeze(1)  # [B, 2H]
            year_embeddings.append(year_embedding)
        
        multi_year_embeddings = torch.stack(year_embeddings, dim=1)  # [B, context_len, 2H]
        return multi_year_embeddings


class YearAttention(nn.Module):
    def __init__(self, hidden_dim, daily_input_dim=5, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 将日度特征投影到年度特征空间
        self.daily_to_annual_proj = nn.Linear(daily_input_dim, hidden_dim * 2)
        
        self.current_year_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.context_year_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.weight_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, current_year_features, context_year_embeddings):
        # current_year_features: [B, 365, K_fast] - 当前年份的日度特征
        # context_year_embeddings: [B, context_len, 2H] - 历史年份的embedding
        
        B, T, K_fast = current_year_features.shape
        B, context_len, H = context_year_embeddings.shape
        
        # 将当前年份的日度特征聚合为年度特征
        current_year_agg = current_year_features.mean(dim=1)  # [B, K_fast]
        
        # 先将日度特征投影到年度特征空间
        current_year_annual = self.daily_to_annual_proj(current_year_agg)  # [B, 2H]
        
        # 投影到相同的维度空间
        current_year_proj = self.current_year_proj(current_year_annual).unsqueeze(1)  # [B, 1, H]
        context_year_proj = self.context_year_proj(context_year_embeddings)  # [B, context_len, H]
        
        # 计算注意力权重
        attn_output, attn_weights = self.attention(
            query=current_year_proj,
            key=context_year_proj,
            value=context_year_proj
        )  # attn_output: [B, 1, H], attn_weights: [B, 1, context_len]
        
        # 计算每个context年份的重要性权重
        importance_weights = torch.softmax(
            self.weight_proj(context_year_proj).squeeze(-1), dim=1
        )  # [B, context_len]
        
        # 结合注意力权重和重要性权重
        final_weights = (attn_weights.squeeze(1) + importance_weights) / 2  # [B, context_len]
        final_weights = torch.softmax(final_weights, dim=1)  # 重新归一化
        
        return final_weights, attn_weights.squeeze(1)


class WeightedContextDecoder(nn.Module):
    """
    根据权重对context年份信息进行加权融合
    """
    def __init__(self, daily_input_dim, daily_hidden_dim, context_dim, output_dim=1, num_heads=4, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(daily_input_dim, daily_hidden_dim, num_layers=3, batch_first=True)
        
        # 将context信息投影到日度特征空间
        self.context_proj = nn.Linear(context_dim, daily_hidden_dim)
        
        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(daily_hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(daily_hidden_dim)
        
        # 输出层
        self.fc = nn.Linear(daily_hidden_dim, output_dim)
        
    def forward(self, x_daily, weighted_context):
        # x_daily: [B, 365, K_fast]
        # weighted_context: [B, context_dim] - 加权后的context信息
        
        lstm_out, _ = self.lstm(x_daily)  # [B, 365, H]
        
        # 将加权context投影到日度特征空间
        context_proj = self.context_proj(weighted_context).unsqueeze(1)  # [B, 1, H]
        
        # 使用多头注意力机制融合信息
        attn_output, attn_weights = self.multihead_attn(
            query=lstm_out,
            key=context_proj.expand(-1, lstm_out.size(1), -1),
            value=context_proj.expand(-1, lstm_out.size(1), -1)
        )
        
        # 残差连接和层归一化
        combined = self.norm(lstm_out + attn_output)
        
        # 预测
        prediction = self.fc(combined).squeeze(-1)  # [B, 365]
        
        return prediction, attn_weights.mean(dim=1)


class HierarchicalMultiEmbedding(nn.Module):
    def __init__(self, annual_input_dim, annual_hidden_dim, daily_input_dim, daily_hidden_dim, output_dim=1):
        super().__init__()
        self.encoder = MultiYearEncoder(annual_input_dim, annual_hidden_dim)
        self.attention = YearAttention(annual_hidden_dim, daily_input_dim)
        self.decoder = WeightedContextDecoder(daily_input_dim, daily_hidden_dim, annual_hidden_dim * 2, output_dim)
        
    def forward(self, x_daily, x_yearly):
        # x_daily: [B, 365, K_fast]
        # x_yearly: [B, context_len, K_slow]
        
        # 1. 为每个context年份生成独立的embedding
        context_year_embeddings = self.encoder(x_yearly)  # [B, context_len, 2H]
        
        # 2. 计算当前年份与历史年份的相关性权重
        year_weights, attn_weights = self.attention(x_daily, context_year_embeddings)  # [B, context_len]
        
        # 3. 根据权重对context年份信息进行加权融合
        weighted_context = (context_year_embeddings * year_weights.unsqueeze(-1)).sum(dim=1)  # [B, 2H]
        
        prediction, daily_attn_weights = self.decoder(x_daily, weighted_context)
        
        return prediction, prediction