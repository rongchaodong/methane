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
            self.output_files = {year: f"../../processed_data/TEM-MDM/full/temporal/output_{year}.npy" for year in self.all_available_years}
        elif data_source == "seq":
            self.input_files = {year: f"../../processed_data/TEM-MDM/sequence/temporal/input_{year}.npy" for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/sequence/temporal/output_{year}.npy" for year in self.all_available_years}
        elif data_source == "sparse":
            self.input_files = {year: f"../../processed_data/TEM-MDM/sparse/temporal/input_{year}.npy" for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/sparse/temporal/output_{year}.npy" for year in self.all_available_years}
        elif data_source == "1percent":
            self.input_files = {year: f"../../processed_data/TEM-MDM/1percent/temporal/input_{year}.npy" for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/1percent/temporal/output_{year}.npy" for year in self.all_available_years}

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
            
            # 加载对应的output数据
            output_data = np.load(self.output_files[year], mmap_mode='r') / 23  # 归一化
            output_features = torch.tensor(output_data, dtype=torch.float32)
            
            # 计算output的均值作为年度特征 [B, 1]
            output_mean = output_features.mean(dim=1, keepdim=True)
            
            # 归一化slow features
            normalized_slow_features = (slow_features_for_year - mean_slow) / std_slow
            
            # 组合input和output特征 [B, K_slow + 1]
            combined_features = torch.cat([normalized_slow_features, output_mean], dim=1)
            
            cache[year] = combined_features.detach()
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


class EnhancedMultiYearEncoder(nn.Module):
    """
    增强的多年编码器，处理包含input和output信息的特征
    """
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        # 分离input和output特征的编码器
        self.input_encoder = nn.GRU(input_dim - 1, hidden_dim // 2, num_layers=num_layers, 
                                   batch_first=True, bidirectional=True)
        self.output_encoder = nn.GRU(1, hidden_dim // 2, num_layers=num_layers, 
                                    batch_first=True, bidirectional=True)
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 年份特定的编码器
        self.year_encoders = nn.ModuleList([
            nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
            for _ in range(10)
        ])
        
    def forward(self, x_yearly):  # [B, context_len, K_slow + 1]
        B, context_len, total_dim = x_yearly.shape
        
        # 分离input和output特征
        input_features = x_yearly[:, :, :total_dim-1]  # [B, context_len, K_slow]
        output_features = x_yearly[:, :, total_dim-1:]  # [B, context_len, 1]
        
        year_embeddings = []
        
        for i in range(context_len):
            year_input = input_features[:, i:i+1, :]  # [B, 1, K_slow]
            year_output = output_features[:, i:i+1, :]  # [B, 1, 1]
            
            # 分别编码input和output特征
            input_encoded, _ = self.input_encoder(year_input)  # [B, 1, H//2*2]
            output_encoded, _ = self.output_encoder(year_output)  # [B, 1, H//2*2]
            
            # 融合特征
            combined = input_encoded + output_encoded  # [B, 1, H]
            fused = self.fusion_layer(combined.squeeze(1))  # [B, H]
            
            # 使用年份特定的编码器
            encoder_idx = min(i, len(self.year_encoders) - 1)
            year_specific_input = fused.unsqueeze(1)  # [B, 1, H]
            year_encoded, _ = self.year_encoders[encoder_idx](year_specific_input)  # [B, 1, 2H]
            
            year_embeddings.append(year_encoded.squeeze(1))  # [B, 2H]
        
        multi_year_embeddings = torch.stack(year_embeddings, dim=1)  # [B, context_len, 2H]
        return multi_year_embeddings


class EnhancedYearAttention(nn.Module):
    """
    增强的年份注意力机制，考虑input和output的联合信息
    """
    def __init__(self, hidden_dim, daily_input_dim=5, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 将日度特征投影到年度特征空间
        self.daily_to_annual_proj = nn.Linear(daily_input_dim, hidden_dim * 2)
        
        # 当前年份投影
        self.current_year_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 上下文年份投影
        self.context_year_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # 相似性计算层
        self.similarity_layer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 权重投影
        self.weight_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, current_year_features, context_year_embeddings):
        # current_year_features: [B, 365, K_fast]
        # context_year_embeddings: [B, context_len, 2H]
        
        B, T, K_fast = current_year_features.shape
        B, context_len, H = context_year_embeddings.shape
        
        # 将当前年份的日度特征聚合为年度特征
        current_year_agg = current_year_features.mean(dim=1)  # [B, K_fast]
        
        # 投影到年度特征空间
        current_year_annual = self.daily_to_annual_proj(current_year_agg)  # [B, 2H]
        
        # 投影到相同的维度空间
        current_year_proj = self.current_year_proj(current_year_annual).unsqueeze(1)  # [B, 1, H]
        context_year_proj = self.context_year_proj(context_year_embeddings)  # [B, context_len, H]
        
        # 计算注意力权重
        attn_output, attn_weights = self.attention(
            query=current_year_proj,
            key=context_year_proj,
            value=context_year_proj
        )
        
        # 计算相似性权重（基于input-output联合特征）
        similarity_input = torch.cat([current_year_annual.unsqueeze(1).expand(-1, context_len, -1), 
                                     context_year_embeddings], dim=-1)  # [B, context_len, 4H]
        similarity_weights = torch.sigmoid(self.similarity_layer(similarity_input).squeeze(-1))  # [B, context_len]
        
        # 计算重要性权重
        importance_weights = torch.softmax(
            self.weight_proj(context_year_proj).squeeze(-1), dim=1
        )
        
        # 组合三种权重
        final_weights = (attn_weights.squeeze(1) + similarity_weights + importance_weights) / 3
        final_weights = torch.softmax(final_weights, dim=1)
        
        return final_weights, attn_weights.squeeze(1)


class EnhancedContextDecoder(nn.Module):
    """
    增强的上下文解码器，更好地利用input-output联合信息
    """
    def __init__(self, daily_input_dim, daily_hidden_dim, context_dim, output_dim=1, num_heads=4, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(daily_input_dim, daily_hidden_dim, num_layers=3, batch_first=True, dropout=dropout)
        
        # 上下文信息投影
        self.context_proj = nn.Linear(context_dim, daily_hidden_dim)
        
        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(daily_hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(daily_hidden_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(daily_hidden_dim, daily_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(daily_hidden_dim * 2, daily_hidden_dim)
        )
        self.norm2 = nn.LayerNorm(daily_hidden_dim)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(daily_hidden_dim, daily_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(daily_hidden_dim // 2, output_dim)
        )
        
    def forward(self, x_daily, weighted_context):
        # x_daily: [B, 365, K_fast]
        # weighted_context: [B, context_dim]
        
        lstm_out, _ = self.lstm(x_daily)  # [B, 365, H]
        
        # 将加权context投影到日度特征空间
        context_proj = self.context_proj(weighted_context).unsqueeze(1)  # [B, 1, H]
        
        # 多头注意力机制
        attn_output, attn_weights = self.multihead_attn(
            query=lstm_out,
            key=context_proj.expand(-1, lstm_out.size(1), -1),
            value=context_proj.expand(-1, lstm_out.size(1), -1)
        )
        
        # 第一个残差连接和层归一化
        combined = self.norm1(lstm_out + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(combined)
        
        # 第二个残差连接和层归一化
        final_features = self.norm2(combined + ffn_output)
        
        # 预测
        prediction = self.output_layer(final_features).squeeze(-1)  # [B, 365]
        
        return prediction, attn_weights.mean(dim=1)


class EnhancedHierarchicalMultiEmbedding(nn.Module):
    """
    增强的分层多嵌入模型，整合input和output信息
    """
    def __init__(self, annual_input_dim, annual_hidden_dim, daily_input_dim, daily_hidden_dim, output_dim=1):
        super().__init__()
        self.encoder = EnhancedMultiYearEncoder(annual_input_dim, annual_hidden_dim)
        self.attention = EnhancedYearAttention(annual_hidden_dim, daily_input_dim)
        self.decoder = EnhancedContextDecoder(daily_input_dim, daily_hidden_dim, annual_hidden_dim * 2, output_dim)
        
    def forward(self, x_daily, x_yearly):
        # x_daily: [B, 365, K_fast]
        # x_yearly: [B, context_len, K_slow + 1] - 现在包含input和output信息
        
        # 1. 为每个context年份生成包含input-output信息的embedding
        context_year_embeddings = self.encoder(x_yearly)  # [B, context_len, 2H]
        
        # 2. 计算当前年份与历史年份的相关性权重
        year_weights, attn_weights = self.attention(x_daily, context_year_embeddings)  # [B, context_len]
        
        # 3. 根据权重对context年份信息进行加权融合
        weighted_context = (context_year_embeddings * year_weights.unsqueeze(-1)).sum(dim=1)  # [B, 2H]
        
        # 4. 解码预测
        prediction, daily_attn_weights = self.decoder(x_daily, weighted_context)
        
        return prediction, prediction