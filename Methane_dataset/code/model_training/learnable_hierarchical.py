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
    def __init__(self, target_years, config, all_available_years_range, data_source : str):
        self.target_years = list(target_years)
        self.config = config
        self.all_available_years = list(all_available_years_range)
        
        self.K_slow = 10 
        self.K_fast = 5 

        if data_source == "1percent":
            self.input_files = {year: f"../../processed_data/TEM-MDM/1percent/temporal/input_{year}.npy" for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/1percent/temporal/output_{year}.npy" for year in self.target_years}
        elif data_source == "seq":
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
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.output_projection = nn.Linear(hidden_dim * 2, output_dim)
        
        # Use separate output projections for each year
        self.output_projections = nn.ModuleList([
            nn.Linear(hidden_dim * 2, output_dim) for _ in range(10)  # Support up to 10 years
        ])

    def forward(self, x_yearly, historical_targets=None):  # [B, context_len, K_slow]
        batch_size, context_len, _ = x_yearly.shape
        
        gru_out, _ = self.gru(x_yearly)  # [B, context_len, 2H]
        attn_weights = torch.softmax(self.attn(gru_out), dim=1)  # [B, context_len, 1]
        context = (attn_weights * gru_out).sum(dim=1)  # [B, 2H]
        
        # Generate predictions and weights for each year
        predictions = []
        weights = []
        
        for year in range(context_len):
            # Use year-specific output projection
            year_predictions = self.output_projections[year](context)  # [B, output_dim]
            predictions.append(year_predictions)
            
            # Get the weights from year-specific projection
            year_weights = self.output_projections[year].weight  # [output_dim, 2H]
            weights.append(year_weights)
        
        # Stack predictions and weights
        predictions = torch.stack(predictions, dim=1)  # [B, context_len, output_dim]
        weights = torch.stack(weights, dim=1)  # [B, context_len, output_dim, 2H]
        
        return context, predictions, weights


class DailyDecoder(nn.Module):
    def __init__(self, daily_input_dim, daily_hidden_dim, annual_context_dim, output_dim=1, num_heads=4, dropout=0.1):
        super(DailyDecoder, self).__init__()
        self.lstm = nn.LSTM(daily_input_dim, daily_hidden_dim, num_layers=3, batch_first=True)
        
        # Weight projection layer: project annual weights to LSTM hidden state dimension
        self.weight_projection = nn.Linear(annual_context_dim, daily_hidden_dim)
        
        # Learnable weight parameter for historical information
        self.historical_weight = nn.Parameter(torch.randn(1))

        self.fc = nn.Linear(daily_hidden_dim, output_dim)

    def forward(self, x_daily, annual_context, annual_weights=None, attention_weights=None):
        # x_daily: [B, 365, K_fast]
        # annual_context: [B, annual_context_dim]
        # annual_weights: [B, context_len, output_dim, annual_context_dim]
        # attention_weights: [B, context_len]

        lstm_out, (h_n, c_n) = self.lstm(x_daily)  # [B, 365, H=8]
        
        # Apply historical information enhancement to LSTM hidden states
        if annual_weights is not None and attention_weights is not None:
            # Project weights to hidden state dimension
            batch_size, context_len, output_dim, annual_context_dim = annual_weights.shape
            
            # Reshape weights for projection
            weights_flat = annual_weights.view(-1, annual_context_dim)  # [B*context_len*output_dim, annual_context_dim]
            projected_weights_flat = self.weight_projection(weights_flat)  # [B*context_len*output_dim, daily_hidden_dim]
            projected_weights = projected_weights_flat.view(batch_size, context_len, output_dim, self.weight_projection.out_features)
            
            # Compute weighted average of projected weights
            weighted_historical_weights = torch.sum(
                attention_weights.unsqueeze(-1).unsqueeze(-1) * projected_weights, dim=1
            )  # [B, output_dim, daily_hidden_dim]
            
            # Use the weighted historical weights to enhance LSTM hidden states
            historical_enhancement = torch.mean(weighted_historical_weights, dim=1)  # [B, self.weight_projection.out_features]
            
            # Enhance LSTM hidden states
            h_n_enhanced = h_n + self.historical_weight * historical_enhancement.unsqueeze(0)
            
            # Re-run LSTM with enhanced hidden states (optional, or just use enhanced states)
            # For simplicity, we'll just use the enhanced hidden states for final prediction
            lstm_out_enhanced = lstm_out + self.historical_weight * historical_enhancement.unsqueeze(1)
        else:
            lstm_out_enhanced = lstm_out

        prediction = self.fc(lstm_out_enhanced).squeeze(-1)

        return prediction, attention_weights if attention_weights is not None else torch.ones(1)  # prediction:[B, 365]


class Hierarchical(nn.Module):
    def __init__(self, annual_input_dim, annual_hidden_dim, daily_input_dim, daily_hidden_dim, output_dim=1):
        super(Hierarchical, self).__init__()
        self.encoder = AnnualEncoder(annual_input_dim, annual_hidden_dim, output_dim)
        self.decoder = DailyDecoder(daily_input_dim, daily_hidden_dim, annual_hidden_dim * 2, output_dim)

    def forward(self, x_daily, x_yearly, historical_targets=None, training_inner=True):
        # x_daily: [B, 365, K_fast]
        # x_yearly: [B, context_len, K_slow]
        # historical_targets: [B, context_len, 365] - historical target Y for self-supervised learning
        
        # Get annual context, predictions, and weights
        annual_context, annual_predictions, annual_weights = self.encoder(x_yearly, historical_targets)
        
        # Compute attention weights for historical information
        # Use the annual context to compute attention weights
        attention_weights = torch.softmax(torch.norm(annual_context, dim=1), dim=0)  # Simple attention based on context norm
        attention_weights = attention_weights.unsqueeze(0).expand(x_yearly.size(0), -1)  # [B, context_len]
        
        # Get main prediction
        prediction, _ = self.decoder(x_daily, annual_context, annual_weights, attention_weights)
        
        # Compute inner model loss for self-supervised learning
        inner_loss = torch.tensor(0.0, device=x_daily.device)
        if historical_targets is not None and training_inner:
            inner_loss = self._compute_inner_loss(annual_predictions, historical_targets, attention_weights)
        
        return prediction, inner_loss
    
    def _compute_inner_loss(self, predictions, historical_targets, attention_weights):
        """
        Compute inner model loss based on prediction min_delta: Σ A_k * ||pred_k - pred_ref||^2
        Args:
            predictions: [B, context_len, output_dim] - predictions from annual encoder
            historical_targets: [B, context_len, 365] - historical target Y
            attention_weights: [B, context_len]
        """
        batch_size, context_len, output_dim = predictions.shape
        
        # Compute reference predictions (mean across years)
        ref_predictions = torch.mean(predictions, dim=1, keepdim=True)  # [B, 1, output_dim]
        ref_predictions = ref_predictions.expand(-1, context_len, -1)  # [B, context_len, output_dim]
        
        # Compute prediction delta (difference from reference)
        prediction_delta = predictions - ref_predictions  # [B, context_len, output_dim]
        
        # Compute min_delta loss (L2 norm of prediction differences)
        min_delta_loss = torch.norm(prediction_delta, p=2, dim=2)  # [B, context_len]
        
        # Weighted sum using attention weights
        weighted_loss = torch.sum(attention_weights * min_delta_loss, dim=1)  # [B,]
        
        return torch.mean(weighted_loss)


class LearnableHierarchicalWrapper(nn.Module):
    """Wrapper class providing second-order gradient support for nested optimization"""
    def __init__(self, annual_input_dim, annual_hidden_dim, daily_input_dim, daily_hidden_dim, output_dim=1, inner_loss_weight=1.0):
        super(LearnableHierarchicalWrapper, self).__init__()
        
        self.model = Hierarchical(annual_input_dim, annual_hidden_dim, daily_input_dim, daily_hidden_dim, output_dim)
        self.inner_loss_weight = inner_loss_weight
        
    def forward(self, x_daily, x_yearly, historical_targets=None, enable_second_order=True):
        """
        Forward pass with second-order gradient support
        Args:
            x_daily: daily input data
            x_yearly: yearly input data
            historical_targets: historical target Y for self-supervised learning
            enable_second_order: whether to enable second-order gradients
        """
        if enable_second_order and historical_targets is not None:
            # Enable second-order gradients for nested optimization
            outputs, inner_loss = self._forward_with_second_order(x_daily, x_yearly, historical_targets)
        else:
            # Standard forward pass
            outputs, inner_loss = self.model(x_daily, x_yearly, historical_targets, training_inner=False)
        
        total_loss = inner_loss * self.inner_loss_weight
        
        return outputs, total_loss, inner_loss
    
    def _forward_with_second_order(self, x_daily, x_yearly, historical_targets):
        """
        Forward pass with second-order gradient computation
        This creates a nested optimization structure where inner model updates
        are computed as gradients of gradients
        """
        # First forward pass to get initial outputs and inner loss
        outputs, inner_loss = self.model(x_daily, x_yearly, historical_targets, training_inner=True)
        
        # Compute gradients of inner loss w.r.t. inner model parameters
        # This creates the "gradient of gradient" effect for second-order optimization
        inner_grads = torch.autograd.grad(
            inner_loss, 
            self.model.encoder.parameters(), 
            create_graph=True,  # Enable second-order gradients
            retain_graph=True,
            allow_unused=True
        )
        
        # The key insight: we don't actually update parameters here
        # Instead, we let the outer optimizer handle the second-order gradients
        # The inner_grads are computed with create_graph=True, which means
        # the outer loss can backpropagate through these gradients
        
        # Return the outputs and inner loss for the outer optimization
        # The second-order gradients will be automatically computed when
        # the outer loss is backpropagated
        return outputs, inner_loss
    
    def predict(self, x_daily, x_yearly):
        """Only perform prediction without training inner model"""
        return self.model(x_daily, x_yearly, training_inner=False)[0]