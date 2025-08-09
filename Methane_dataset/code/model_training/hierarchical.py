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
    def __init__(self, target_years, config, all_available_years_range):
        self.target_years = list(target_years)
        self.config = config
        self.all_available_years = list(all_available_years_range)
        
        self.K_slow = 10 
        self.K_fast = 5 

        self.input_files = {year: f"../../processed_data/TEM-MDM/sequence/temporal/input_{year}.npy" for year in self.all_available_years}
        self.output_files = {year: f"../../processed_data/TEM-MDM/sequence/temporal/output_{year}.npy" for year in self.target_years}
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
        # print("year_list_idx", year_list_idx, " year:", actual_year, " data length: ", self.data_lengths[year_list_idx])
        # print("file idx:", file_idx)

        full_input_sample = np.load(self.input_files[actual_year], mmap_mode='r')[file_idx]
        # print("this year data:", full_input_sample[0])
        
        normalized_input = (full_input_sample - self.mean_vals) / self.std_vals
        # print("this year data:", normalized_input[0][:10])
        
        x_daily = normalized_input[:, self.K_slow:]
        
        y_target = np.load(self.output_files[actual_year], mmap_mode='r')[file_idx] / 23

        context_vectors = []
        for i in range(1, self.config.num_context_years + 1):
            context_year = actual_year - i
            # print(self.mean_vals.shape)
            # print(self.std_vals.shape)
            # print("context year", context_year, " context year length", self.data_lengths[year_list_idx - i])
            # print("context year true data:", np.load(self.input_files[context_year], mmap_mode='r')[file_idx][0][:10])
            
            if context_year in self.yearly_features_cache:
                vec = self.yearly_features_cache[context_year][file_idx]
            else:
                earliest_year = self.all_available_years[0]
                vec = self.yearly_features_cache[earliest_year][file_idx]
            # print("context vec of :", context_year, " ", vec)
            context_vectors.append(vec)
        
        x_yearly = torch.stack(context_vectors)
        # print("target", y_target.shape)
        # print("x_daily", x_daily.shape)
        # print("year_daily", x_yearly.shape)

        return (
            torch.tensor(x_daily, dtype=torch.float32).to(config.device),
            x_yearly.to(config.device),
            torch.tensor(y_target, dtype=torch.float32).to(config.device)
        )

class HierarchicalSpatialDataset(Dataset):
    """
    1. x_daily: fast features in data: (B, T=365, K_fast=5), has 5 features
    2. x_yearly: slow features: (B, num_context_years, K_slow=10), 10 features
    3. y_target: the y_batch shaped in (B, T=365)
    """
    def __init__(self, target_years, config, all_available_years_range, folds):
        self.target_years = list(target_years)
        self.config = config
        self.all_available_years = list(all_available_years_range)
        self.folds = folds
        
        self.K_slow = 10 
        self.K_fast = 5 
        """
        input/output_files structure:
        input_1979_0, input_1979_1, input_1979_2...
        """
        self.input_files = {f"{year}_{fold}": f"../../processed_data/TEM-MDM/sequence/spatial/input_{year}_{fold}.npy" for year in self.all_available_years for fold in folds}
        self.output_files = {f"{year}_{fold}": f"../../processed_data/TEM-MDM/sequence/spatial/output_{year}_{fold}.npy" for year in target_years for fold in folds}
        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')

        self.yearly_caches = {}
        for f in folds:
            yearly_features_cache = self._precompute_yearly_features(f)
            self.yearly_caches[f] = yearly_features_cache

        self.data_lengths = [np.load(self.input_files[key], mmap_mode='r').shape[0] for key in self.input_files.keys()]
        self.cumsum_lengths = np.cumsum([0] + self.data_lengths)

    def _precompute_yearly_features(self, fold: int):
        cache = {}
        mean_slow = self.mean_vals[:self.K_slow]
        std_slow = self.std_vals[:self.K_slow]

        for year in self.all_available_years:
            key = f"{year}_{fold}"
            # print(key)
            full_year_data = np.load(self.input_files[key], mmap_mode='r')
            slow_features_for_year = full_year_data[:, 0, :self.K_slow]
            slow_features_for_year = torch.tensor(slow_features_for_year, dtype=torch.float32)
            
            normalized_slow_features = (slow_features_for_year - mean_slow) / std_slow
            # print(normalized_slow_features)
            cache[key] = normalized_slow_features.detach()
        return cache

    def __len__(self):
        return self.cumsum_lengths[-1]

    def __getitem__(self, idx):
        folded_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        file_idx = idx - self.cumsum_lengths[folded_idx]
        # print(self.folds)
        year = folded_idx // len(self.folds)
        fold = self.folds[folded_idx % len(self.folds)]
        # print("folded_idx", folded_idx, "year", year, "fold", fold, " data length: ", self.data_lengths[folded_idx])
        actual_year = self.target_years[year]
        # print("file idx:", file_idx, " actual year", actual_year)

        key = f"{actual_year}_{fold}"
        full_input_sample = np.load(self.input_files[key], mmap_mode='r')[file_idx]
        # print("this year data:", full_input_sample[0])
        
        normalized_input = (full_input_sample - self.mean_vals) / self.std_vals
        # print("this year data:", actual_year, " ", normalized_input[0][:10])
        
        x_daily = normalized_input[:, self.K_slow:]
        
        y_target = np.load(self.output_files[key], mmap_mode='r')[file_idx] / 23

        context_vectors = []
        for i in range(1, self.config.num_context_years + 1):
            context_year = actual_year - i
            # print(self.mean_vals.shape)
            # print(self.std_vals.shape)
            # print("context year", context_year, " context year length", self.data_lengths[folded_idx - i])
            # print("context year true data:", np.load(self.input_files[context_year], mmap_mode='r')[file_idx][0][:10])
            
            key = f"{context_year}_{fold}"
            if key in self.yearly_caches[fold]:
                vec = self.yearly_caches[fold][key][file_idx]
            else:
                earliest_year = self.all_available_years[0]
                key = f"{earliest_year}_{fold}"
                vec = self.yearly_caches[fold][key][file_idx]
            # print("context vec of :", context_year, " ", vec)
            context_vectors.append(vec)
        
        x_yearly = torch.stack(context_vectors)
        # print("target", y_target.shape)
        # print("x_daily", x_daily.shape)
        # print("year_daily", x_yearly.shape)
        # exit()

        return (
            torch.tensor(x_daily, dtype=torch.float32).to(config.device),
            x_yearly.to(config.device),
            torch.tensor(y_target, dtype=torch.float32).to(config.device)
        )

# class AnnualEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers=1):
#         super(AnnualEncoder, self).__init__()
#         self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

#     def forward(self, x_yearly):
#         _, h_n = self.gru(x_yearly)
#         # print("GRU shape:", h_n.shape, h_n.squeeze(0).shape)
#         return h_n.squeeze(0)

class AnnualEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_yearly):  # [B, context_len, K_slow]
        gru_out, _ = self.gru(x_yearly)  # [B, context_len, 2H]
        attn_weights = torch.softmax(self.attn(gru_out), dim=1)  # [B, context_len, 1]
        context = (attn_weights * gru_out).sum(dim=1)  # [B, 2H]
        return context


class CrossAttention(nn.Module):
    def __init__(self, daily_hidden_dim, annual_context_dim):
        super(CrossAttention, self).__init__()
        self.query_layer = nn.Linear(daily_hidden_dim, annual_context_dim)

    def forward(self, daily_hidden, annual_context):
        # daily_hidden: [B, 365, H_d]
        # annual_context: [B, H_y]
        B, T, H_d = daily_hidden.shape
        annual_context = annual_context.unsqueeze(1)  # [B, 1, H_y]
        query = self.query_layer(daily_hidden)        # [B, 365, H_y]

        # Dot-product attention score: [B, 365, 1]
        energy = torch.bmm(query, annual_context.transpose(1, 2)) / (H_d ** 0.5)
        attention_weights = F.softmax(energy, dim=1)  # [B, 365, 1]

        # context = weighted sum of annual_context (broadcasted): [B, 365, H_y]
        context = attention_weights * annual_context

        return context, attention_weights.squeeze(-1)  # [B, 365, H_y], [B, 365]


class DailyDecoder(nn.Module):
    def __init__(self, daily_input_dim, daily_hidden_dim, annual_context_dim, output_dim=1, num_heads=4, dropout=0.1):
        super(DailyDecoder, self).__init__()
        self.lstm = nn.LSTM(daily_input_dim, daily_hidden_dim, num_layers=3, batch_first=True)
        
        self.attn_proj = nn.Linear(annual_context_dim*2, daily_hidden_dim)  # project to same dim
        # cross attention
        self.multihead_attn = nn.MultiheadAttention(daily_hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(daily_hidden_dim)

        self.fc = nn.Linear(daily_hidden_dim, output_dim)

    def forward(self, x_daily, annual_context):
        # x_daily: [B, 365, K_fast]
        # annual_context: [B, annual_context_dim]

        lstm_out, _ = self.lstm(x_daily)  # [B, 365, H=8]

        annual_context_proj = self.attn_proj(annual_context).unsqueeze(1)  # [B, 1, H]

        attn_output, attn_weights = self.multihead_attn(
            query=lstm_out,
            key=annual_context_proj.expand(-1, lstm_out.size(1), -1),  # [B, 365, H]
            value=annual_context_proj.expand(-1, lstm_out.size(1), -1) # [B, 365, H]
        )  # attn_output: [B, 365, H], attn_weights: [B, 365, 365]

        combined = self.norm(lstm_out + attn_output)  # [B, 365, H]

        prediction = self.fc(combined).squeeze(-1)

        return prediction, attn_weights.mean(dim=1)  # prediction:[B, 365]


class Hierarchical(nn.Module):
    def __init__(self, annual_input_dim, annual_hidden_dim, daily_input_dim, daily_hidden_dim, output_dim=1):
        super(Hierarchical, self).__init__()
        self.encoder = AnnualEncoder(annual_input_dim, annual_hidden_dim)
        self.decoder = DailyDecoder(daily_input_dim, daily_hidden_dim, annual_hidden_dim, output_dim)

    def forward(self, x_daily, x_yearly):
        # print("input shape: daily:", x_daily.shape, "yearly:", x_yearly.shape)
        annual_context = self.encoder(x_yearly)
        # print("annaual context shape:", annual_context.shape)
        # context_norm = torch.norm(annual_context, dim=1).mean().item()
        # print(f"Context L2 norm (context={6}):", context_norm)

        prediction, attention_weights = self.decoder(x_daily, annual_context)
        # print(attention_weights)
        # exit()
        
        return prediction, prediction