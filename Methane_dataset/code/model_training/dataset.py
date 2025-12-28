from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pickle

class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TemporalDataset(Dataset):
    def __init__(self, years, config):
        self.config = config
        self.years = years
        self.year_to_idx = {year: i for i, year in enumerate(range(1979, 1979 + config.max_num_years))}
        if config.data_type == "seq":
            if config.data_source == "full":
                self.input_files = [f"../../processed_data/TEM-MDM/full/temporal/input_{year}.npy" for year in years]
                self.output_files = [f"../../processed_data/TEM-MDM/full/temporal/output_{year}.npy" for year in years]
            elif config.data_source == "seq":
                self.input_files = [f"../../processed_data/TEM-MDM/sequence/temporal/input_{year}.npy" for year in years]
                self.output_files = [f"../../processed_data/TEM-MDM/sequence/temporal/output_{year}.npy" for year in years]
        elif config.data_type == "random":
            self.input_files = [f"../../processed_data/TEM-MDM/temporal/input_{year}.npy" for year in years]
            self.output_files = [f"../../processed_data/TEM-MDM/temporal/output_{year}.npy" for year in years]
        self.data_lengths = [np.load(f, mmap_mode='r').shape[0] for f in self.input_files] 
        self.cumsum_lengths = np.cumsum([0] + self.data_lengths)  
        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')

    def __len__(self):
        return self.cumsum_lengths[-1]  

    def __getitem__(self, idx):
        year_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        file_idx = idx - self.cumsum_lengths[year_idx]

        # actual_year = self.years[year_idx]
        # global_year_idx = self.year_to_idx[actual_year]

        input_data = (np.load(self.input_files[year_idx], mmap_mode='r')[file_idx] - self.mean_vals) / self.std_vals 
        output_data = np.load(self.output_files[year_idx], mmap_mode='r')[file_idx] / 23

        return torch.tensor(input_data, dtype=torch.float32).to(self.config.device), torch.tensor(output_data, dtype=torch.float32).to(self.config.device)
    

class HierarchicalTemporalDataset(Dataset):
    """
    Dataset for hierarchical temporal data with historical information
    1. x_current: current year data (B, T=365, input_size=15), all features
    2. historical_inputs: historical year data (B, num_context_years, T=365, input_size=15)
    3. y_target: target values (B, T=365)
    4. historical_targets: historical target Y for self-supervised learning (B, num_context_years, T=365)
    """
    def __init__(self, target_years, config, all_available_years_range):
        self.target_years = list(target_years)
        self.config = config
        self.all_available_years = list(all_available_years_range)
        self.data_source = config.data_source

        if self.data_source == "1percent":
            self.input_files = {year: f"../../processed_data/TEM-MDM/1percent/temporal/input_{year}.npy" for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/1percent/temporal/output_{year}.npy" for year in self.all_available_years}
        elif self.data_source == "seq":
            self.input_files = {year: f"../../processed_data/TEM-MDM/sequence/temporal/input_{year}.npy" for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/sequence/temporal/output_{year}.npy" for year in self.all_available_years}
        elif self.data_source == "sparse":
            self.input_files = {year: f"../../processed_data/TEM-MDM/sparse/temporal/input_{year}.npy" for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/sparse/temporal/output_{year}.npy" for year in self.all_available_years}
        elif self.data_source == "full":
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
             x_current.to(self.config.device),
             historical_inputs.to(self.config.device),
             y_target.to(self.config.device),
             historical_targets.to(self.config.device),
             torch.tensor(file_idx, dtype=torch.long).to(self.config.device)
         )
    


### FLUXNET DATASET

class FluxnetBaselineDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        
        data_dir = '../../processed_data/FLUXNET-CH4/site_centric/'
        
        with open(os.path.join(data_dir, 'all_data_cache.pkl'), 'rb') as f:
            self.all_data_cache = pickle.load(f)
            
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
        
        print(f"Loaded baseline {mode} dataset with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        site_name, target_year = self.samples[idx]
        
        data = self.all_data_cache[site_name][target_year]
        x_raw = data['input']
        y_raw = data['output']
        
        x_normalized = (x_raw - self.mean_vals) / self.std_vals
        y_normalized = y_raw
        
        x_tensor = torch.tensor(x_normalized, dtype=torch.float32)
        y_tensor = torch.tensor(y_normalized, dtype=torch.float32)
        
        return x_tensor.to(self.config.device), y_tensor.to(self.config.device)

class FluxnetHierarchicalDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        
        data_dir = '../../processed_data/FLUXNET-CH4/site_centric/'

        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')
        self.y_mean = None
        self.y_std = None
        
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
        
        self.max_locations = len(self.site_to_id_map)
        
        self._get_normalization()

        print(f"Loaded {mode} dataset with {len(self.samples)} samples from {self.max_locations} unique sites.")

    def _get_normalization(self):
        if self.mode != 'train':
            return
        # std method: mean and std
        # all_y_data_list = []
        # for site_name, target_year in self.samples:
        #     data = self.all_data_cache[site_name][target_year]
        #     all_y_data_list.append(data['output']) 
        
        # all_y_data = np.concatenate(all_y_data_list, axis=0)

        # self.y_mean = all_y_data.mean(axis=0)
        # self.y_std = all_y_data.std(axis=0)

        # from sklearn.preprocessing import power_transform
        # all_y_data_list = []
        # for site_name, target_year in self.samples:
        #     data = self.all_data_cache[site_name][target_year]
        #     all_y_data_list.append(data['output']) 
        
        # all_y_data = np.concatenate(all_y_data_list, axis=0)
        # all_y_data_transformed = power_transform(all_y_data.reshape(-1, 1), method='yeo-johnson', standardize=True).flatten()

        # log method
        all_y_data_list = []
        for site_name, target_year in self.samples:
            data = self.all_data_cache[site_name][target_year]
            all_y_data_list.append(data['output'])
        all_y_data = np.concatenate(all_y_data_list, axis=0)
        all_y_data_logged = np.sign(all_y_data) * np.log1p(np.abs(all_y_data))

        self.y_mean = all_y_data_logged.mean(axis=0)
        self.y_std = all_y_data_logged.std(axis=0)

    def __len__(self):
        return len(self.samples)
    
    def __location_length__(self):
        return self.max_locations

    def __getitem__(self, idx):
        site_name, target_year = self.samples[idx]
        
        location_id = self.site_to_id_map[site_name]
        
        current_data = self.all_data_cache[site_name][target_year]
        
        x_current_raw = current_data['input']
        x_current = (x_current_raw - self.mean_vals) / self.std_vals
        
        y_target_raw = current_data['output']

        if self.mode == 'train':
            y_target_raw = np.sign(y_target_raw) * np.log1p(np.abs(y_target_raw))
            y_target = (y_target_raw - self.y_mean) / self.y_std
        else:
            y_target = y_target_raw
        
        x_current = torch.tensor(x_current, dtype=torch.float32)
        y_target = torch.tensor(y_target, dtype=torch.float32)

        historical_inputs = []
        historical_targets = []
        
        site_data = self.all_data_cache[site_name]
        
        available_hist_years = sorted(
            [y for y in site_data.keys() if y < target_year], 
            reverse=True
        )
        
        for i in range(self.config.num_context_years):
            if i < len(available_hist_years):
                hist_year = available_hist_years[i]
                hist_data = site_data[hist_year]
                
                hist_input_raw = hist_data['input']
                hist_input = (hist_input_raw - self.mean_vals) / self.std_vals
                # print("current year:", target_year, "hist_year:", hist_year)
                # print("hist_input: ", hist_input[150][:9])
                
                hist_target_raw = hist_data['output']
                if self.mode == 'train':
                    hist_target_raw = np.sign(hist_target_raw) * np.log1p(np.abs(hist_target_raw))
                    hist_target = (hist_target_raw - self.y_mean) / self.y_std
                else:
                    hist_target = hist_target_raw

                historical_inputs.append(torch.tensor(hist_input, dtype=torch.float32))
                historical_targets.append(torch.tensor(hist_target, dtype=torch.float32))
                
            else:
                if len(available_hist_years) > 0:
                    earliest_year = available_hist_years[-1] 
                    earliest_data = site_data[earliest_year]
                    
                    earliest_input_raw = earliest_data['input']
                    earliest_input = (earliest_input_raw - self.mean_vals) / self.std_vals

                    # print("current year:", target_year, "earliest_year:", earliest_year)
                    # print("earliest_input: ", earliest_input[150][:9])
                    
                    earliest_target_raw = earliest_data['output']
                    if self.mode == 'train':
                        earliest_target_raw = np.sign(earliest_target_raw) * np.log1p(np.abs(earliest_target_raw))
                        earliest_target = (earliest_target_raw - self.y_mean) / self.y_std
                    else:
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