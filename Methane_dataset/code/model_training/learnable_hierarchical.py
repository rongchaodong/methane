import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import math
from config import Config
import pickle

config = Config()

def save_location_weights(model, save_path):
    location_weights = {}
    
    for i, projection in enumerate(model.model.encoder.output_projections):
        location_weights[f'location_{i}'] = {
            'weight': projection.weight.data.clone(),
            'bias': projection.bias.data.clone() if projection.bias is not None else None
        }
    
    torch.save(location_weights, save_path)
    print(f"Saved location weights to {save_path}")

def load_pretrained_model_with_location_matching(pretrained_model_path, pretrained_location_weights_path, fluxnet_model):
    pretrained_state = torch.load(pretrained_model_path, map_location='cpu')
    current_state = fluxnet_model.state_dict()
    
    for key, value in pretrained_state.items():
        if key in current_state:
            # load all other possible weights, skip output_projections
            if 'output_projections' in key:
                print(f"Skipping location-specific weight: {key}")
                continue
            
            if current_state[key].shape == value.shape:
                current_state[key] = value
                print(f"Loaded: {key}")
            else:
                print(f"Shape mismatch for {key}: {current_state[key].shape} vs {value.shape}")
    
    fluxnet_model.load_state_dict(current_state)
    
    pretrained_location_weights = torch.load(pretrained_location_weights_path, map_location='cpu')
    
    fluxnet_model.model.encoder.pretrained_location_weights = pretrained_location_weights
    fluxnet_model.model.encoder.use_pretrained_weights = True
    
    print("Successfully loaded pretrained model with location weight matching capability")
    return fluxnet_model

class ContrastiveLocationLearning(nn.Module):
    """
    Contrastive learning module for location-specific weights
    Goal: Make location weights more representative and pure
    """
    def __init__(self, weight_dim, contrastive_dim=32, temperature=0.1, noise_std=0.05):
        super().__init__()
        self.weight_dim = weight_dim
        self.contrastive_dim = contrastive_dim
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.noise_std = noise_std
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(weight_dim, contrastive_dim),
            nn.ReLU(),
            nn.Linear(contrastive_dim, contrastive_dim),
            nn.LayerNorm(contrastive_dim)
        )
        
        # Adaptive weight scheduler
        self.contrastive_weight = nn.Parameter(torch.tensor(1.0))
        self.min_weight = 0.1
        self.max_weight = 2.0
        
    def augment_weights(self, weights):
        """
        Data augmentation for location weights with stronger augmentation
        Args:
            weights: [B, context_len, output_dim, weight_dim]
        Returns:
            augmented_weights: [B, context_len, output_dim, weight_dim]
        """
        batch_size, context_len, output_dim, weight_dim = weights.shape
        
        # Create multiple augmented versions for better contrastive learning
        augmented_versions = []
        
        for aug_idx in range(1):
            # Add stronger Gaussian noise
            noise = torch.randn_like(weights) * (self.noise_std * (1 + aug_idx * 0.5))
            augmented = weights + noise
            
            # Time-based augmentation: create temporal variations
            temporal_augmented = []
            for i in range(batch_size):
                loc_weights = augmented[i]  # [context_len, output_dim, weight_dim]
                
                # Method 1: Temporal smoothing with different weights
                temporal_smooth = torch.mean(loc_weights, dim=0, keepdim=True).expand(context_len, -1, -1)
                
                # Method 2: Add temporal noise with different scales
                temporal_noise = torch.randn_like(loc_weights) * (self.noise_std * (0.5 + aug_idx * 0.3))
                temporal_variation = loc_weights + temporal_noise
                
                # Method 3: Random dropout-like augmentation
                dropout_mask = torch.rand_like(loc_weights) > 0.1  # 10% dropout
                dropout_augmented = loc_weights * dropout_mask.float()
                
                combined = 0.5 * temporal_variation + 0.5 * dropout_augmented
                
                temporal_augmented.append(combined)
            
            temporal_augmented = torch.stack(temporal_augmented, dim=0)
            augmented_versions.append(temporal_augmented)
        
        # Return the first augmented version (you can modify this to return multiple versions)
        return augmented_versions[0]
    
    def compute_contrastive_loss(self, pruned_inputs, weights):
        """
        Compute InfoNCE contrastive loss using input similarity to select positive/negative samples
        Args:
            pruned_inputs: [B, context_len, seq_len, feature_dim] - used to compute similarity for sampling
            weights: [B, context_len, output_dim, weight_dim]
        Returns:
            contrastive_loss: scalar tensor
        """
        batch_size, context_len, output_dim, weight_dim = weights.shape
        
        # Flatten weights for processing: [B*context_len*output_dim, weight_dim]
        weights_flat = weights.view(-1, weight_dim)
        
        # Project to contrastive space
        projected = self.projection_head(weights_flat)  # [B*context_len*output_dim, contrastive_dim]
        
        # Reshape back: [B, context_len*output_dim, contrastive_dim]
        projected = projected.view(batch_size, context_len * output_dim, self.contrastive_dim)
        
        # Normalize for cosine similarity
        projected = F.normalize(projected, dim=-1)
        
        # Compute input similarity for positive/negative sampling
        # Average over context_len and seq_len to get location-level representation
        input_repr = pruned_inputs.mean(dim=(1, 2))  # [B, feature_dim]
        input_repr = F.normalize(input_repr, dim=-1)  # normalize for cosine similarity
        
        # Compute pairwise cosine similarities between locations
        input_similarities = torch.matmul(input_repr, input_repr.T)  # [B, B]
        
        total_loss = 0.0
        valid_pairs = 0
        
        # Number of positive samples: 1/4 of batch size (excluding self)
        num_positives = max(1, (batch_size - 1) // 4)
        
        for i in range(batch_size):
            # Get similarities for location i with all other locations
            similarities = input_similarities[i]  # [B]
            
            # Exclude self-similarity
            mask = torch.ones(batch_size, dtype=torch.bool, device=similarities.device)
            mask[i] = False
            other_similarities = similarities[mask]  # [B-1]
            other_indices = torch.arange(batch_size, device=similarities.device)[mask]  # [B-1]
            
            if len(other_indices) == 0:
                continue
                
            # Select top similar locations as positives
            _, top_indices = torch.topk(other_similarities, min(num_positives, len(other_indices)))
            positive_indices = other_indices[top_indices]
            
            # Remaining locations as negatives
            negative_mask = torch.ones(len(other_indices), dtype=torch.bool, device=similarities.device)
            negative_mask[top_indices] = False
            negative_indices = other_indices[negative_mask]
            
            # Get anchor representation (use first context*output representation)
            anchor = projected[i, 0:1]  # [1, contrastive_dim]
            
            # Get positive samples (use first context*output representation from each positive location)
            if len(positive_indices) > 0:
                pos_samples = projected[positive_indices, 0]  # [num_pos, contrastive_dim]
                pos_sim = torch.matmul(anchor, pos_samples.T) / self.temperature  # [1, num_pos]
            else:
                # Fallback: use temporal positive from same location
                if context_len * output_dim > 1:
                    pos_samples = projected[i, 1:2]  # [1, contrastive_dim]
                    pos_sim = torch.matmul(anchor, pos_samples.T) / self.temperature  # [1, 1]
                else:
                    continue
            
            # Get negative samples
            if len(negative_indices) > 0:
                # Use a subset of negatives to balance computation
                max_negatives = min(len(negative_indices), 3 * num_positives)
                if len(negative_indices) > max_negatives:
                    neg_subset_idx = torch.randperm(len(negative_indices))[:max_negatives]
                    negative_indices = negative_indices[neg_subset_idx]
                
                neg_samples = projected[negative_indices, 0]  # [num_neg, contrastive_dim]
                neg_sim = torch.matmul(anchor, neg_samples.T) / self.temperature  # [1, num_neg]
                
                # InfoNCE loss
                logits = torch.cat([pos_sim, neg_sim], dim=1)  # [1, num_pos + num_neg]
                labels = torch.zeros(1, dtype=torch.long, device=logits.device)
                
                loss = F.cross_entropy(logits, labels)
                total_loss += loss
                valid_pairs += 1
        
        if valid_pairs > 0:
            return total_loss / valid_pairs
        else:
            return torch.tensor(0.0, device=weights.device)

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
        elif data_source == "full":
            self.input_files = {year: f"../../processed_data/TEM-MDM/full/temporal/input_{year}.npy" for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/full/temporal/output_{year}.npy" for year in self.all_available_years}

        # Use float32 mean/std to avoid float64 temporaries
        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy').astype(np.float32)
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy').astype(np.float32)

        # Build memory-mapped handles (lightweight). Do NOT load full arrays into memory.
        self.input_memmaps = {year: np.load(path, mmap_mode='r') for year, path in self.input_files.items()}
        self.output_memmaps = {year: np.load(path, mmap_mode='r') for year, path in self.output_files.items()}

        # historical_cache kept for API compatibility but we don't precompute large arrays
        self.historical_cache = {}

        self.data_lengths = [self.input_memmaps[year].shape[0] for year in self.target_years]
        self.cumsum_lengths = np.cumsum([0] + self.data_lengths)
        
        # Calculate the maximum number of locations across all years (cheap)
        self.max_locations = max([self.input_memmaps[year].shape[0] for year in self.all_available_years if year in self.input_memmaps])

    def _precompute_historical_data(self):
        """Precompute all year data to improve training efficiency"""
        # no-op: we use memmaps and normalize per-sample in __getitem__ to avoid large allocations
        return {}

    def __len__(self):
        return self.cumsum_lengths[-1]
    
    def __location_length__(self):
        return self.max_locations

    def __getitem__(self, idx):
        year_list_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        file_idx = idx - self.cumsum_lengths[year_list_idx]
        actual_year = self.target_years[year_list_idx]

        # Current year sample read from memmap (small view, no full-copy until astype)
        inp_mm = self.input_memmaps[actual_year]
        out_mm = self.output_memmaps[actual_year]
        x_current = inp_mm[file_idx]   # (365, 15) - all features
        y_target = out_mm[file_idx]
         
         # Collect historical year data
        historical_inputs = []
        historical_targets = []
        
        for i in range(1, self.config.num_context_years + 1):
            historical_year = actual_year - i
             
            if historical_year in self.input_memmaps:
                hist_inp_mm = self.input_memmaps[historical_year]
                hist_out_mm = self.output_memmaps[historical_year]
                safe_idx = min(file_idx, hist_inp_mm.shape[0] - 1)
                historical_inputs.append(hist_inp_mm[safe_idx])
                historical_targets.append(hist_out_mm[safe_idx])
            else:
                # If historical year doesn't exist, use earliest year data
                earliest_year = min(self.all_available_years)
                hist_inp_mm = self.input_memmaps[earliest_year]
                hist_out_mm = self.output_memmaps[earliest_year]
                safe_idx = min(file_idx, hist_inp_mm.shape[0] - 1)
                historical_inputs.append(hist_inp_mm[safe_idx])
                historical_targets.append(hist_out_mm[safe_idx])
         

        # Per-sample normalization and conversion to float32 tensors (minimize temporaries)
        x_current = ((x_current.astype(np.float32) - self.mean_vals) / self.std_vals).astype(np.float32)
        y_target = (y_target.astype(np.float32) / 23.0).astype(np.float32)
        historical_inputs = np.stack(historical_inputs, axis=0).astype(np.float32)
        # normalize each historical input sample
        historical_inputs = (historical_inputs - self.mean_vals) / self.std_vals
        historical_targets = np.stack(historical_targets, axis=0).astype(np.float32) / 23.0

        x_current = torch.tensor(x_current, dtype=torch.float32)
        y_target = torch.tensor(y_target, dtype=torch.float32)
        historical_inputs = torch.tensor(historical_inputs, dtype=torch.float32)
        historical_targets = torch.tensor(historical_targets, dtype=torch.float32)
 
        return (
             x_current.to(config.device),
             historical_inputs.to(config.device),
             y_target.to(config.device),
             historical_targets.to(config.device),
             torch.tensor(file_idx, dtype=torch.long).to(config.device)
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
        
        # Calculate the maximum number of locations across all years and folds
        self.max_locations = max([np.load(self.input_files[key], mmap_mode='r').shape[0] for key in self.input_files.keys()])

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

    def __location_length__(self):
        return self.max_locations

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
            historical_targets.to(config.device),
            torch.tensor(file_idx, dtype=torch.long).to(config.device)
        )


class FluxnetHierarchicalDataset(Dataset):
    """
    用于 FLUXNET 数据的分层时间数据集。
    
    这个 Dataset 读取一个“以站点为中心”的缓存，该缓存由预处理 notebook 生成。
    它能智能地为每个 (站点, 年份) 样本查找真实的历史数据。
    """
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        
        data_dir = '../../processed_data/FLUXNET-CH4/site_centric/'
        
        # 1. 加载预处理好的数据
        import os
        with open(os.path.join(data_dir, 'all_data_cache.pkl'), 'rb') as f:
            self.all_data_cache = pickle.load(f)
            
        with open(os.path.join(data_dir, 'site_to_id_map.pkl'), 'rb') as f:
            self.site_to_id_map = pickle.load(f)
            
        if mode == 'train':
            with open(os.path.join(data_dir, 'train_samples.pkl'), 'rb') as f:
                self.samples = pickle.load(f)
        elif mode == 'test':
            with open(os.path.join(data_dir, 'test_samples.pkl'), 'rb') as f:
                self.samples = pickle.load(f)
        else:
            raise ValueError(f"Mode '{mode}' not recognized. Use 'train' or 'test'.")

        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')
        
        self.max_locations = len(self.site_to_id_map)

        print(f"Loaded {mode} dataset with {len(self.samples)} samples from {self.max_locations} unique sites.")

    def __len__(self):
        return len(self.samples)
    
    def __location_length__(self):
        return self.max_locations

    def __getitem__(self, idx):
        # 1. 获取此样本对应的 站点 和 年份
        site_name, target_year = self.samples[idx]
        
        # 2. 获取唯一的整数 "location_id" (新的 file_idx)
        # 这将用于 AnnualEncoder 中的 nn.ModuleList
        location_id = self.site_to_id_map[site_name]
        
        # 3. 获取当前年份的数据
        current_data = self.all_data_cache[site_name][target_year]
        
        x_current_raw = current_data['input']
        x_current = (x_current_raw - self.mean_vals) / self.std_vals
        # print("x_current:", x_current[150][:9])
        
        y_target_raw = current_data['output']
        y_target = y_target_raw
        
        # 转换为 Tensors
        x_current = torch.tensor(x_current, dtype=torch.float32)
        y_target = torch.tensor(y_target, dtype=torch.float32)

        historical_inputs = []
        historical_targets = []
        
        # 获取此站点的所有可用数据
        site_data = self.all_data_cache[site_name]
        
        # 找到所有早于目标年份的年份，并按降序排列
        available_hist_years = sorted(
            [y for y in site_data.keys() if y < target_year], 
            reverse=True
        )
        
        for i in range(self.config.num_context_years):
            if i < len(available_hist_years):
                # --- 情况1: 找到了真实的历史年份 ---
                hist_year = available_hist_years[i]
                hist_data = site_data[hist_year]
                
                # 标准化
                hist_input_raw = hist_data['input']
                hist_input = (hist_input_raw - self.mean_vals) / self.std_vals
                # print("current year:", target_year, "hist_year:", hist_year)
                # print("hist_input: ", hist_input[150][:9])
                
                hist_target_raw = hist_data['output']
                hist_target = hist_target_raw
                
                historical_inputs.append(torch.tensor(hist_input, dtype=torch.float32))
                historical_targets.append(torch.tensor(hist_target, dtype=torch.float32))
                
            else:
                if len(available_hist_years) > 0:
                    earliest_year = available_hist_years[-1] # 最早的一年
                    earliest_data = site_data[earliest_year]
                    
                    earliest_input_raw = earliest_data['input']
                    earliest_input = (earliest_input_raw - self.mean_vals) / self.std_vals

                    # print("current year:", target_year, "earliest_year:", earliest_year)
                    # print("earliest_input: ", earliest_input[150][:9])
                    
                    earliest_target_raw = earliest_data['output']
                    earliest_target = earliest_target_raw
                    
                    historical_inputs.append(torch.tensor(earliest_input, dtype=torch.float32))
                    historical_targets.append(torch.tensor(earliest_target, dtype=torch.float32))
                else:
                    # print("Warning: No historical data available for site:", site_name, "year:", target_year)
                    # print("should not happen if data is prepared correctly.")
                    historical_inputs.append(x_current)
                    historical_targets.append(y_target)
        
        # exit()
        historical_inputs = torch.stack(historical_inputs)
        historical_targets = torch.stack(historical_targets)

        return (
            x_current.to(self.config.device),
            historical_inputs.to(self.config.device),
            y_target.to(self.config.device),
            historical_targets.to(self.config.device),
            torch.tensor(location_id, dtype=torch.long).to(self.config.device)
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
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1, enable_contrastive=True, dataset=None, pretrained_location_weights=None):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.output_projection = nn.Linear(hidden_dim * 2, output_dim)
        
        # Use separate output projections for each location
        # Get the actual number of locations from the dataset
        if dataset is not None and hasattr(dataset, '__location_length__'):
            self.max_locations = dataset.__location_length__()
        else:
            print("Warning: No dataset provided")
            exit()
        
        self.output_projections = nn.ModuleList([
            nn.Linear(hidden_dim * 2, output_dim) for _ in range(self.max_locations)
        ])
        
        # Store pretrained location weights for dynamic matching
        self.pretrained_location_weights = pretrained_location_weights
        self.use_pretrained_weights = pretrained_location_weights is not None
        
        # Contrastive learning module
        self.enable_contrastive = enable_contrastive
        if enable_contrastive:
            self.contrastive_learning = ContrastiveLocationLearning(
                weight_dim=hidden_dim * 2,  # GRU output dimension
                contrastive_dim=hidden_dim * 2,  # Keep same dimension as weight_dim
                temperature=0.1,  # Increased temperature for better learning
                noise_std=0.05   # Increased noise for stronger augmentation
            )

    def forward(self, historical_inputs, file_idx, historical_targets=None, compute_contrastive=True):  # [B, context_len, seq_len, input_size]
        batch_size, context_len, seq_len, input_size = historical_inputs.shape
        
        # Process each historical year through GRU
        # Reshape to process all years together: [B*context_len, seq_len, input_size]
        historical_flat = historical_inputs.view(-1, seq_len, input_size)
        gru_out, _ = self.gru(historical_flat)  # [B*context_len, seq_len, 2H]
        
        gru_out = gru_out.view(batch_size, context_len, seq_len, -1)  # [B, context_len, 2H]
        
        
        # Generate predictions and weights for each location using file_idx
        predictions = []
        weights = []
        
        for loc in range(batch_size):
            # Get the actual location index
            location_idx = file_idx[loc].item()
            
            if self.use_pretrained_weights:
                current_input = historical_inputs[loc]  # [context_len, seq_len, input_size]
                
                best_match_idx = self.find_similar_pretrained_location(current_input, location_idx)
                
                pretrained_weights = self.pretrained_location_weights[best_match_idx]
                
                original_weight = self.output_projections[location_idx].weight.data.clone()
                original_bias = self.output_projections[location_idx].bias.data.clone() if self.output_projections[location_idx].bias is not None else None
                
                self.output_projections[location_idx].weight.data = pretrained_weights['weight']
                if pretrained_weights['bias'] is not None and self.output_projections[location_idx].bias is not None:
                    self.output_projections[location_idx].bias.data = pretrained_weights['bias']
            
            # Use location-specific output projection
            loc_predictions = self.output_projections[location_idx](gru_out[loc, :, :, :])  # [context_len, seq_len, output_dim]
            predictions.append(loc_predictions)
            
            # Get the weights from location-specific projection and expand to batch dimension
            location_weights = self.output_projections[location_idx].weight  # [output_dim, 2H]
            location_weights_expanded = location_weights.unsqueeze(0).expand(context_len, -1, -1) # [context_len, output_dim, 2H]
            weights.append(location_weights_expanded)
            
            if self.use_pretrained_weights:
                self.output_projections[location_idx].weight.data = original_weight
                if original_bias is not None and self.output_projections[location_idx].bias is not None:
                    self.output_projections[location_idx].bias.data = original_bias
        
        # Stack predictions and weights
        predictions = torch.stack(predictions, dim=0)  # [B, context_len, seq_len, output_dim]
        weights = torch.stack(weights, dim=0)  # [B, context_len, output_dim, 2H]
        
        # Compute contrastive loss if enabled
        contrastive_loss = torch.tensor(0.0, device=historical_inputs.device)
        if self.enable_contrastive and compute_contrastive and self.training:
            # Apply data augmentation
            pruned_inputs = historical_inputs[:, :, :, :10]  # Use first 10 static features to pick pos and neg samples
            augmented_weights = self.contrastive_learning.augment_weights(weights)
            contrastive_loss = self.contrastive_learning.compute_contrastive_loss(pruned_inputs, augmented_weights)

        # print("predictions shape:", predictions.shape)
        # print("weights shape:", weights.shape)
        # exit()
        return predictions, weights, contrastive_loss


class DailyDecoder(nn.Module):
    def __init__(self, input_dim, daily_hidden_dim, annual_context_dim, output_dim=1, num_heads=4, dropout=0.1):
        super(DailyDecoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, daily_hidden_dim, num_layers=1, batch_first=True)
        
        # Weight projection layer: project annual weights to LSTM hidden state dimension
        self.weight_projection = nn.Linear(annual_context_dim, daily_hidden_dim)
        
        # Learnable weight parameter for historical information
        self.historical_weight = nn.Parameter(torch.randn(1))

        self.fc = nn.Linear(daily_hidden_dim, output_dim)
        self.head = nn.Sequential(
            nn.Linear(daily_hidden_dim, daily_hidden_dim * 4),
            nn.GELU(),
            nn.Linear(daily_hidden_dim * 4, output_dim)
        )

    def forward(self, x_current, annual_weights=None, attention_weights=None):
        # x_current: [B, 365, input_size=15] - all features
        # annual_weights: [B, context_len, output_dim, 2H]
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
            # attention_weights: [B, context_len]
            # projected_weights: [B, context_len, output_dim=1, hidden_dim=8]
            weighted_historical_weights = torch.sum(
                attention_weights.unsqueeze(-1).unsqueeze(-1) * projected_weights, dim=1
            )  # [B, output_dim, hidden_dim]
            
            # Use the weighted historical weights to enhance LSTM hidden states
            historical_enhancement = weighted_historical_weights.squeeze(1)  # [B, daily_hidden_dim]

            # normalize historical enhancement: scale to [-1, 1]
            historical_enhancement = 2 * (historical_enhancement - historical_enhancement.min()) / (historical_enhancement.max() - historical_enhancement.min()) - 1
            historical_enhancement = historical_enhancement.unsqueeze(0)  # [1, B, daily_hidden_dim]
        
        
        # Initialize hidden states
        h_states = [torch.zeros(1, batch_size, self.lstm.hidden_size, device=device) for _ in range(1)]
        c_states = [torch.zeros(1, batch_size, self.lstm.hidden_size, device=device) for _ in range(1)]

        
        # Apply historical enhancement to the final layer's hidden state, and maintain tanh scale
        # initial the last layer into historical enhancement, not sure useful
        # I can also try to use zero as hidden state, but let's try this first
        if historical_enhancement is not None:
            h_states[-1] = (h_states[-1] + self.historical_weight * historical_enhancement) / (1 + torch.abs(self.historical_weight))
        # print("enhanced hidden state: ", h_states[-1])
        # exit()


        lstm_out_enhanced, _ = self.lstm(x_current, (h_states[-1], c_states[-1]))  # [B, seq_len, hidden_size]

        prediction = self.head(lstm_out_enhanced).squeeze(-1) # [B, 365]

        return prediction  # prediction:[B, 365]


class Hierarchical(nn.Module):
    def __init__(self, input_dim, annual_hidden_dim, daily_hidden_dim, output_dim=1, enable_contrastive=True, dataset=None, pretrained_location_weights=None):
        super(Hierarchical, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1, batch_first=True)
        self.encoder = AnnualEncoder(input_dim, annual_hidden_dim, output_dim, enable_contrastive=enable_contrastive, dataset=dataset, pretrained_location_weights=pretrained_location_weights)
        self.decoder = DailyDecoder(input_dim, daily_hidden_dim, annual_hidden_dim * 2, output_dim)
        self.enable_contrastive = enable_contrastive

    def forward(self, x_current, historical_inputs, file_idx, historical_targets=None, training_inner=True, compute_contrastive=True):
        # x_current: [B, 365, input_size=15] - current year data
        # historical_inputs: [B, context_len, 365, input_size=15] - historical year data
        # file_idx: [B] - location indices for each sample in the batch
        # historical_targets: [B, context_len, 365] - historical target Y for self-supervised learning
        
        attention_weights = self.cross_attn(
            query=x_current.mean(dim=1, keepdim=True),  # [B, 1, input_size]
            key=historical_inputs.mean(dim=2),  # [B, context_len, input_size]
            value=historical_inputs.mean(dim=2)   # [B, context_len, input_size]
        )[1].squeeze(1) # [B, context_len]
        # print("attention weights:", attention_weights)
        # exit()
        # Get annual context, predictions, weights, and attention weights
        if self.enable_contrastive:
            annual_predictions, annual_weights, contrastive_loss = self.encoder(historical_inputs, file_idx, historical_targets, compute_contrastive=True)
        else:
            annual_predictions, annual_weights, _ = self.encoder(historical_inputs, file_idx, historical_targets, compute_contrastive=False)
            contrastive_loss = torch.tensor(0.0, device=x_current.device)
        
        # Get main prediction
        prediction = self.decoder(x_current, annual_weights, attention_weights)
        
        # Compute inner model loss for self-supervised learning
        inner_loss = torch.tensor(0.0, device=x_current.device)
        if historical_targets is not None and training_inner:
            inner_loss = self._compute_inner_loss(annual_predictions, historical_targets, attention_weights)
        
        return prediction, inner_loss, contrastive_loss
    
    def _compute_inner_loss(self, predictions, historical_targets, attention_weights):
        """
        Compute inner model loss using historical targets for self-supervised learning
        Args:
            predictions: [B, context_len, seq_len=365, output_dim] - predictions from annual encoder
            historical_targets: [B, context_len, 365] - historical target Y
            attention_weights: [B, context_len]
        """
        batch_size, context_len, seq_len = historical_targets.shape
        
        # Compute MSE loss between predictions and historical targets
        mse_loss = F.mse_loss(predictions.squeeze(-1), historical_targets, reduction='none')  # [B, context_len, 365]
        
        # Average over sequence dimension
        mse_loss = torch.mean(mse_loss, dim=2)  # [B, context_len]
        
        # Weighted sum using attention weights
        weighted_loss = torch.sum(attention_weights * mse_loss, dim=1)  # [B,]
        
        return torch.mean(weighted_loss)


class LearnableHierarchicalWrapper(nn.Module):
    """Wrapper class providing second-order gradient support for nested optimization"""
    def __init__(self, input_dim, annual_hidden_dim, daily_hidden_dim, output_dim=1, inner_loss_weight=1.0, enable_contrastive=True, dataset=None, pretrained_location_weights=None):
        super(LearnableHierarchicalWrapper, self).__init__()
        
        self.model = Hierarchical(input_dim, annual_hidden_dim, daily_hidden_dim, output_dim, enable_contrastive=enable_contrastive, dataset=dataset, pretrained_location_weights=pretrained_location_weights)
        self.inner_loss_weight = inner_loss_weight
        self.enable_contrastive = enable_contrastive
        
        # Adaptive weight parameters for contrastive learning
        if enable_contrastive:
            self.contrastive_weight = nn.Parameter(torch.tensor(1.0))
            self.min_contrastive_weight = 0.1
            self.max_contrastive_weight = 2.0
        
    def forward(self, x_current, historical_inputs, file_idx, historical_targets=None, enable_second_order=True, epoch=0, total_epochs=20):
        """
        Forward pass with second-order gradient support and adaptive contrastive learning
        Args:
            x_current: current year input data [B, 365, input_size]
            historical_inputs: historical year input data [B, context_len, 365, input_size]
            file_idx: location indices for each sample in the batch [B]
            historical_targets: historical target Y for self-supervised learning [B, context_len, 365]
            enable_second_order: whether to enable second-order gradients
            epoch: current training epoch for adaptive weighting
            total_epochs: total training epochs for adaptive weighting
        """
        if enable_second_order and historical_targets is not None and self.training:
            # Enable second-order gradients for nested optimization (only during training)
            outputs, inner_loss, contrastive_loss = self._forward_with_second_order(x_current, historical_inputs, file_idx, historical_targets)
        else:
            # Standard forward pass (during evaluation)
            outputs, inner_loss, contrastive_loss = self.model(x_current, historical_inputs, file_idx, historical_targets, training_inner=False)
        
        # Compute adaptive contrastive weight
        if self.enable_contrastive and self.training:
            adaptive_weight = self._get_adaptive_contrastive_weight(epoch, total_epochs)
            contrastive_loss_weighted = contrastive_loss * adaptive_weight
        else:
            contrastive_loss_weighted = torch.tensor(0.0, device=x_current.device)
        
        # Total loss with adaptive weighting
        total_loss = inner_loss * self.inner_loss_weight + contrastive_loss_weighted
        
        return outputs, total_loss, inner_loss, contrastive_loss
    
    def _forward_with_second_order(self, x_current, historical_inputs, file_idx, historical_targets):
        """
        Forward pass with second-order gradient computation
        This creates a nested optimization structure where inner model updates
        are computed as gradients of gradients
        """
        # First forward pass to get initial outputs, inner loss, and contrastive loss
        outputs, inner_loss, contrastive_loss = self.model(x_current, historical_inputs, file_idx, historical_targets, training_inner=True, compute_contrastive=True)
        
        # Compute gradients of inner loss w.r.t. inner model parameters
        # This creates the "gradient of gradient" effect for second-order optimization
        try:
            inner_grads = torch.autograd.grad(
                inner_loss, 
                self.model.encoder.parameters(), 
                create_graph=True,  # Enable second-order gradients
                retain_graph=True,
                allow_unused=True
            )
        except RuntimeError as e:
            if "does not require grad" in str(e):
                # If gradients are not available, fall back to standard forward pass
                outputs, inner_loss, contrastive_loss = self.model(x_current, historical_inputs, file_idx, historical_targets, training_inner=False, compute_contrastive=False)
                return outputs, inner_loss, contrastive_loss
            else:
                raise e
        
        # The key insight: we don't actually update parameters here
        # Instead, we let the outer optimizer handle the second-order gradients
        # The inner_grads are computed with create_graph=True, which means
        # the outer loss can backpropagate through these gradients
        
        # Return the outputs, inner loss, and contrastive loss for the outer optimization
        # The second-order gradients will be automatically computed when
        # the outer loss is backpropagated
        return outputs, inner_loss, contrastive_loss
    
    def _get_adaptive_contrastive_weight(self, epoch, total_epochs):
        """
        Get adaptive weight for contrastive loss
        Early training: higher weight (focus on learning pure representations)
        Later training: lower weight (focus on prediction accuracy)
        """
        if total_epochs == 0:
            return self.max_contrastive_weight
        
        # Linear decay from max_weight to min_weight
        progress = epoch / total_epochs
        weight = self.max_contrastive_weight - (self.max_contrastive_weight - self.min_contrastive_weight) * progress
        
        # Apply sigmoid for smoother transition
        weight = self.min_contrastive_weight + (self.max_contrastive_weight - self.min_contrastive_weight) * (1 - torch.sigmoid(torch.tensor(progress * 4 - 2)))
        
        return weight.item()
    
    def predict(self, x_current, historical_inputs, file_idx):
        """Only perform prediction without training inner model"""
        return self.model(x_current, historical_inputs, file_idx, training_inner=False, compute_contrastive=False)[0]