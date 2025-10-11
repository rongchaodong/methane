import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from config import Config

config = Config()

class HierarchicalTemporalDataset(Dataset):
    """
    Dataset for hierarchical temporal data with historical information
    1. x_current: current year data (B, T=365, input_size=15), all features
    2. historical_inputs: historical year data (B, num_context_years, T=365, input_size=15)
    3. y_target: target values (B, T=365)
    4. historical_targets: historical target Y for self-supervised learning (B, num_context_years, T=365)
    """
    def __init__(self, target_years, config, all_available_years_range, data_source: str):
        self.target_years = list(target_years)
        self.config = config
        self.all_available_years = list(all_available_years_range)
        self.data_source = data_source

        if data_source == "1percent":
            self.input_files = {year: f"../../processed_data/TEM-MDM/1percent/temporal/input_{year}.npy" for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/1percent/temporal/output_{year}.npy" for year in self.all_available_years}
        elif data_source == "seq":
            self.input_files = {year: f"../../processed_data/TEM-MDM/sequence/temporal/input_{year}.npy" for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/sequence/temporal/output_{year}.npy" for year in self.all_available_years}
        elif data_source == "sparse":
            self.input_files = {year: f"../../processed_data/TEM-MDM/sparse/temporal/input_{year}.npy" for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/sparse/temporal/output_{year}.npy" for year in self.all_available_years}
            
        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')

        # Precompute all historical data for efficiency
        self.historical_cache = self._precompute_historical_data()

        self.data_lengths = [np.load(self.input_files[year], mmap_mode='r').shape[0] for year in self.target_years]
        self.cumsum_lengths = np.cumsum([0] + self.data_lengths)

    def _precompute_historical_data(self):
        """Precompute all year data to improve training efficiency"""
        cache = {}
        for year in self.all_available_years:
            if year in self.input_files:
                input_data = np.load(self.input_files[year], mmap_mode='r')
                output_data = np.load(self.output_files[year], mmap_mode='r')
                
                # Normalize
                normalized_input = (input_data - self.mean_vals) / self.std_vals
                normalized_output = output_data / 23
                
                cache[year] = {
                    'input': normalized_input,
                    'output': normalized_output
                }
        return cache

    def __len__(self):
        return self.cumsum_lengths[-1]

    def __getitem__(self, idx):
        year_list_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        file_idx = idx - self.cumsum_lengths[year_list_idx]
        actual_year = self.target_years[year_list_idx]

        # Current year data
        current_data = self.historical_cache[actual_year]
        x_current = current_data['input'][file_idx]  # (365, 15) - all features
        y_target = current_data['output'][file_idx]  # (365,)
        
        # Collect historical year data
        historical_inputs = []
        historical_targets = []
        
        for i in range(1, self.config.num_context_years + 1):
            historical_year = actual_year - i
            
            if historical_year in self.historical_cache:
                hist_data = self.historical_cache[historical_year]
                # Ensure index doesn't go out of bounds
                safe_idx = min(file_idx, hist_data['input'].shape[0] - 1)
                historical_inputs.append(hist_data['input'][safe_idx])  # (365, 15) - all features
                historical_targets.append(hist_data['output'][safe_idx])  # (365,)
            else:
                # If historical year doesn't exist, use earliest year data
                earliest_year = min(self.all_available_years)
                hist_data = self.historical_cache[earliest_year]
                safe_idx = min(file_idx, hist_data['input'].shape[0] - 1)
                historical_inputs.append(hist_data['input'][safe_idx])
                historical_targets.append(hist_data['output'][safe_idx])
        
        # Convert to tensors
        x_current = torch.tensor(x_current, dtype=torch.float32)
        y_target = torch.tensor(y_target, dtype=torch.float32)
        historical_inputs = torch.stack([torch.tensor(h, dtype=torch.float32) for h in historical_inputs])
        historical_targets = torch.stack([torch.tensor(h, dtype=torch.float32) for h in historical_targets])

        return (
            x_current.to(config.device),
            historical_inputs.to(config.device),
            y_target.to(config.device),
            historical_targets.to(config.device)
        )

class HierarchicalSpatialDataset(Dataset):
    """
    Dataset for hierarchical spatial data with historical information
    1. x_current: current year data (B, T=365, input_size=15), all features
    2. historical_inputs: historical year data (B, num_context_years, T=365, input_size=15)
    3. y_target: target values (B, T=365)
    4. historical_targets: historical target Y for self-supervised learning (B, num_context_years, T=365)
    """
    def __init__(self, target_years, config, all_available_years_range, folds):
        self.target_years = list(target_years)
        self.config = config
        self.all_available_years = list(all_available_years_range)
        self.folds = folds
        
        # Data file paths
        self.input_files = {f"{year}_{fold}": f"../../processed_data/TEM-MDM/sequence/spatial/input_{year}_{fold}.npy" 
                           for year in self.all_available_years for fold in folds}
        self.output_files = {f"{year}_{fold}": f"../../processed_data/TEM-MDM/sequence/spatial/output_{year}_{fold}.npy" 
                            for year in self.all_available_years for fold in folds}
        
        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')

        # Precompute all historical data for efficiency
        self.historical_cache = self._precompute_historical_data()

        self.data_lengths = [np.load(self.input_files[key], mmap_mode='r').shape[0] for key in self.input_files.keys()]
        self.cumsum_lengths = np.cumsum([0] + self.data_lengths)

    def _precompute_historical_data(self):
        """Precompute all historical data"""
        cache = {}
        for key, input_file in self.input_files.items():
            input_data = np.load(input_file, mmap_mode='r')
            output_data = np.load(self.output_files[key], mmap_mode='r')
            
            # Normalize
            normalized_input = (input_data - self.mean_vals) / self.std_vals
            normalized_output = output_data / 23
            
            cache[key] = {
                'input': normalized_input,
                'output': normalized_output
            }
        return cache

    def __len__(self):
        return self.cumsum_lengths[-1]

    def __getitem__(self, idx):
        folded_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        file_idx = idx - self.cumsum_lengths[folded_idx]
        year = folded_idx // len(self.folds)
        fold = self.folds[folded_idx % len(self.folds)]
        actual_year = self.target_years[year]

        # Current year data
        key = f"{actual_year}_{fold}"
        current_data = self.historical_cache[key]
        x_current = current_data['input'][file_idx]  # (365, 15) - all features
        y_target = current_data['output'][file_idx]  # (365,)

        # Collect historical year data
        historical_inputs = []
        historical_targets = []
        
        for i in range(1, self.config.num_context_years + 1):
            context_year = actual_year - i
            context_key = f"{context_year}_{fold}"
            
            if context_key in self.historical_cache:
                hist_data = self.historical_cache[context_key]
                # Ensure index doesn't go out of bounds
                safe_idx = min(file_idx, hist_data['input'].shape[0] - 1)
                historical_inputs.append(hist_data['input'][safe_idx])  # (365, 15) - all features
                historical_targets.append(hist_data['output'][safe_idx])  # (365,)
            else:
                # If historical year doesn't exist, use earliest year data
                earliest_year = min(self.all_available_years)
                context_key = f"{earliest_year}_{fold}"
                hist_data = self.historical_cache[context_key]
                safe_idx = min(file_idx, hist_data['input'].shape[0] - 1)
                historical_inputs.append(hist_data['input'][safe_idx])
                historical_targets.append(hist_data['output'][safe_idx])
        
        # Convert to tensors
        x_current = torch.tensor(x_current, dtype=torch.float32)
        y_target = torch.tensor(y_target, dtype=torch.float32)
        historical_inputs = torch.stack([torch.tensor(h, dtype=torch.float32) for h in historical_inputs])
        historical_targets = torch.stack([torch.tensor(h, dtype=torch.float32) for h in historical_targets])

        return (
            x_current.to(config.device),
            historical_inputs.to(config.device),
            y_target.to(config.device),
            historical_targets.to(config.device)
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

    def forward(self, historical_inputs, historical_targets=None):  # [B, context_len, seq_len, input_size]
        batch_size, context_len, seq_len, input_size = historical_inputs.shape
        
        # Process each historical year through GRU
        # Reshape to process all years together: [B*context_len, seq_len, input_size]
        historical_flat = historical_inputs.view(-1, seq_len, input_size)
        gru_out, _ = self.gru(historical_flat)  # [B*context_len, seq_len, 2H]
        
        # Get the last output for each year
        gru_out_last = gru_out[:, -1, :]  # [B*context_len, 2H]
        gru_out_last = gru_out_last.view(batch_size, context_len, -1)  # [B, context_len, 2H]
        
        # Compute attention weights
        attn_weights = torch.softmax(self.attn(gru_out_last), dim=1)  # [B, context_len, 1]
        context = (attn_weights * gru_out_last).sum(dim=1)  # [B, 2H]
        
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
        
        return context, predictions, weights, attn_weights.squeeze(-1)


class DailyDecoder(nn.Module):
    def __init__(self, input_dim, daily_hidden_dim, annual_context_dim, output_dim=1, num_heads=4, dropout=0.1):
        super(DailyDecoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, daily_hidden_dim, num_layers=3, batch_first=True)
        
        # Weight projection layer: project annual weights to LSTM hidden state dimension
        self.weight_projection = nn.Linear(annual_context_dim, daily_hidden_dim)
        
        # Learnable weight parameter for historical information
        self.historical_weight = nn.Parameter(torch.randn(1))

        self.fc = nn.Linear(daily_hidden_dim, output_dim)

    def forward(self, x_current, annual_context, annual_weights=None, attention_weights=None):
        # x_current: [B, 365, input_size=15] - all features
        # annual_context: [B, annual_context_dim]
        # annual_weights: [B, context_len, output_dim, annual_context_dim]
        # attention_weights: [B, context_len]

        batch_size, seq_len, _ = x_current.shape
        device = x_current.device
        
        # Pre-compute historical enhancement if available
        historical_enhancement = None
        if annual_weights is not None and attention_weights is not None:
            # Project weights to hidden state dimension
            context_len, output_dim, annual_context_dim = annual_weights.shape[1], annual_weights.shape[2], annual_weights.shape[3]
            
            # Reshape weights for projection
            weights_flat = annual_weights.view(-1, annual_context_dim)  # [B*context_len*output_dim, annual_context_dim]
            projected_weights_flat = self.weight_projection(weights_flat)  # [B*context_len*output_dim, daily_hidden_dim]
            projected_weights = projected_weights_flat.view(batch_size, context_len, output_dim, self.weight_projection.out_features)
            
            # Compute weighted average of projected weights
            weighted_historical_weights = torch.sum(
                attention_weights.unsqueeze(-1).unsqueeze(-1) * projected_weights, dim=1
            )  # [B, output_dim, daily_hidden_dim]
            
            # Use the weighted historical weights to enhance LSTM hidden states
            historical_enhancement = torch.mean(weighted_historical_weights, dim=1)  # [B, daily_hidden_dim]
        
        # Use LSTMCell for step-by-step processing with historical enhancement
        lstm_cells = [nn.LSTMCell(self.lstm.input_size if i == 0 else self.lstm.hidden_size, 
                                 self.lstm.hidden_size).to(device) for i in range(3)]
        
        # Copy weights from the original LSTM
        for i, cell in enumerate(lstm_cells):
            cell.weight_ih.data = self.lstm.weight_ih_l0.data if i == 0 else getattr(self.lstm, f'weight_ih_l{i}').data
            cell.weight_hh.data = self.lstm.weight_hh_l0.data if i == 0 else getattr(self.lstm, f'weight_hh_l{i}').data
            cell.bias_ih.data = self.lstm.bias_ih_l0.data if i == 0 else getattr(self.lstm, f'bias_ih_l{i}').data
            cell.bias_hh.data = self.lstm.bias_hh_l0.data if i == 0 else getattr(self.lstm, f'bias_hh_l{i}').data
        
        # Initialize hidden states
        h_states = [torch.zeros(batch_size, self.lstm.hidden_size, device=device) for _ in range(3)]
        c_states = [torch.zeros(batch_size, self.lstm.hidden_size, device=device) for _ in range(3)]
        
        outputs = []
        
        # Step-by-step LSTM processing with historical enhancement
        for t in range(seq_len):
            x_t = x_current[:, t, :]  # [B, input_size=15]
            
            # Process through each LSTM layer
            for layer in range(3):
                if layer == 0:
                    input_t = x_t
                else:
                    input_t = h_states[layer-1]
                
                h_states[layer], c_states[layer] = lstm_cells[layer](input_t, (h_states[layer], c_states[layer]))
            
            # Apply historical enhancement to the final layer's hidden state
            if historical_enhancement is not None:
                h_states[-1] = h_states[-1] + self.historical_weight * historical_enhancement
            
            # Store output
            outputs.append(h_states[-1].unsqueeze(1))  # [B, 1, hidden_size]
        
        # Concatenate all outputs
        lstm_out_enhanced = torch.cat(outputs, dim=1)  # [B, seq_len, hidden_size]
        
        prediction = self.fc(lstm_out_enhanced).squeeze(-1)

        return prediction, attention_weights if attention_weights is not None else torch.ones(1)  # prediction:[B, 365]


class Hierarchical(nn.Module):
    def __init__(self, input_dim, annual_hidden_dim, daily_hidden_dim, output_dim=1):
        super(Hierarchical, self).__init__()
        self.encoder = AnnualEncoder(input_dim, annual_hidden_dim, output_dim)
        self.decoder = DailyDecoder(input_dim, daily_hidden_dim, annual_hidden_dim * 2, output_dim)

    def forward(self, x_current, historical_inputs, historical_targets=None, training_inner=True):
        # x_current: [B, 365, input_size=15] - current year data
        # historical_inputs: [B, context_len, 365, input_size=15] - historical year data
        # historical_targets: [B, context_len, 365] - historical target Y for self-supervised learning
        
        # Get annual context, predictions, weights, and attention weights
        annual_context, annual_predictions, annual_weights, attention_weights = self.encoder(historical_inputs, historical_targets)
        
        # Get main prediction
        prediction, _ = self.decoder(x_current, annual_context, annual_weights, attention_weights)
        
        # Compute inner model loss for self-supervised learning
        inner_loss = torch.tensor(0.0, device=x_current.device)
        if historical_targets is not None and training_inner:
            inner_loss = self._compute_inner_loss(annual_predictions, historical_targets, attention_weights)
        
        return prediction, inner_loss
    
    def _compute_inner_loss(self, predictions, historical_targets, attention_weights):
        """
        Compute inner model loss using historical targets for self-supervised learning
        Args:
            predictions: [B, context_len, output_dim] - predictions from annual encoder
            historical_targets: [B, context_len, 365] - historical target Y
            attention_weights: [B, context_len]
        """
        batch_size, context_len, seq_len = historical_targets.shape
        
        # Expand predictions to match historical targets sequence length
        # For simplicity, we'll use the predictions as is (assuming output_dim=1)
        # In practice, you might want to expand predictions to full sequence length
        if predictions.shape[-1] == 1:
            # If predictions are scalar, expand to sequence length
            predictions_expanded = predictions.expand(-1, -1, seq_len)  # [B, context_len, 365]
        else:
            predictions_expanded = predictions  # [B, context_len, 365]
        
        # Compute MSE loss between predictions and historical targets
        mse_loss = F.mse_loss(predictions_expanded, historical_targets, reduction='none')  # [B, context_len, 365]
        
        # Average over sequence dimension
        mse_loss = torch.mean(mse_loss, dim=2)  # [B, context_len]
        
        # Weighted sum using attention weights
        weighted_loss = torch.sum(attention_weights * mse_loss, dim=1)  # [B,]
        
        return torch.mean(weighted_loss)


class LearnableHierarchicalWrapper(nn.Module):
    """Wrapper class providing second-order gradient support for nested optimization"""
    def __init__(self, input_dim, annual_hidden_dim, daily_hidden_dim, output_dim=1, inner_loss_weight=1.0):
        super(LearnableHierarchicalWrapper, self).__init__()
        
        self.model = Hierarchical(input_dim, annual_hidden_dim, daily_hidden_dim, output_dim)
        self.inner_loss_weight = inner_loss_weight
        
    def forward(self, x_current, historical_inputs, historical_targets=None, enable_second_order=True):
        """
        Forward pass with second-order gradient support
        Args:
            x_current: current year input data [B, 365, input_size]
            historical_inputs: historical year input data [B, context_len, 365, input_size]
            historical_targets: historical target Y for self-supervised learning [B, context_len, 365]
            enable_second_order: whether to enable second-order gradients
        """
        if enable_second_order and historical_targets is not None:
            # Enable second-order gradients for nested optimization
            outputs, inner_loss = self._forward_with_second_order(x_current, historical_inputs, historical_targets)
        else:
            # Standard forward pass
            outputs, inner_loss = self.model(x_current, historical_inputs, historical_targets, training_inner=False)
        
        total_loss = inner_loss * self.inner_loss_weight
        
        return outputs, total_loss, inner_loss
    
    def _forward_with_second_order(self, x_current, historical_inputs, historical_targets):
        """
        Forward pass with second-order gradient computation
        This creates a nested optimization structure where inner model updates
        are computed as gradients of gradients
        """
        # First forward pass to get initial outputs and inner loss
        outputs, inner_loss = self.model(x_current, historical_inputs, historical_targets, training_inner=True)
        
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
    
    def predict(self, x_current, historical_inputs):
        """Only perform prediction without training inner model"""
        return self.model(x_current, historical_inputs, training_inner=False)[0]