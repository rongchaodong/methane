import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Dataset
import math
import logging
from sklearn.metrics import mean_squared_error, r2_score
from config import Config
from learnable_hidden_state_lstm import LearnableHiddenStateLSTMWrapper

config = Config()
criterion = nn.MSELoss()

parser = argparse.ArgumentParser(description="training learnable hidden state LSTM for methane dataset")
parser.add_argument("--type", type=str, default="temporal", help="please choose temporal or spatial")
parser.add_argument("--simulated", action="store_true", help="training simulated data or observed data")
parser.add_argument("--data_type", type=str, default="seq", help="Model selection: random, or seq")
parser.add_argument("--epoch", type=int, default=100, help="Training epoches")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--hidden_size", type=int, default=8, help="Hidden size for LSTM")
parser.add_argument("--historical_years", type=int, default=2, help="Number of historical years")
parser.add_argument("--inner_model_type", type=str, default="lstm", help="Inner model type: lstm, linear, mlp")
parser.add_argument("--inner_loss_weight", type=float, default=1.0, help="Weight for inner model loss")
parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers")
args = parser.parse_args()

class LearnableHiddenTemporalDataset(Dataset):
    def __init__(self, target_years, all_available_years_range, data_source: str, historical_years=2):
        self.target_years = list(target_years)
        self.all_available_years = list(all_available_years_range)
        self.historical_years = historical_years
        self.data_source = data_source
        
        # Data file paths
        if data_source == "1percent":
            self.input_files = {year: f"../../processed_data/TEM-MDM/1percent/temporal/input_{year}.npy" 
                              for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/1percent/temporal/output_{year}.npy" 
                               for year in self.all_available_years}
        elif data_source == "seq":
            self.input_files = {year: f"../../processed_data/TEM-MDM/sequence/temporal/input_{year}.npy" 
                              for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/sequence/temporal/output_{year}.npy" 
                               for year in self.all_available_years}
        elif data_source == "sparse":
            self.input_files = {year: f"../../processed_data/TEM-MDM/sparse/temporal/input_{year}.npy" 
                              for year in self.all_available_years}
            self.output_files = {year: f"../../processed_data/TEM-MDM/sparse/temporal/output_{year}.npy" 
                               for year in self.all_available_years}
        
        # Normalization parameters
        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')
        
        # Calculate data lengths
        self.data_lengths = [np.load(self.input_files[year], mmap_mode='r').shape[0] 
                           for year in self.target_years]
        self.cumsum_lengths = np.cumsum([0] + self.data_lengths)
        
        # Precompute historical data cache
        self.historical_cache = self._precompute_historical_data()
    
    def _precompute_historical_data(self):
        """Precompute all year data to improve training efficiency"""
        cache = {}
        for year in self.all_available_years:
            if year in self.input_files:
                input_data = np.load(self.input_files[year], mmap_mode='r')
                output_data = np.load(self.output_files.get(year, self.input_files[year].replace('input', 'output')), mmap_mode='r')
                
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
        # Find corresponding year and sample index
        year_list_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        file_idx = idx - self.cumsum_lengths[year_list_idx]
        actual_year = self.target_years[year_list_idx]
        
        # Current year data
        current_data = self.historical_cache[actual_year]
        x_current = current_data['input'][file_idx]  # (365, 15)
        y_current = current_data['output'][file_idx]  # (365, 1)
        
        # Collect historical year data
        historical_inputs = []
        historical_outputs = []
        
        for i in range(1, self.historical_years + 1):
            historical_year = actual_year - i
            if historical_year in self.historical_cache:
                hist_data = self.historical_cache[historical_year]
                # Ensure index doesn't go out of bounds
                safe_idx = min(file_idx, hist_data['input'].shape[0] - 1)
                historical_inputs.append(hist_data['input'][safe_idx])
                historical_outputs.append(hist_data['output'][safe_idx])
            else:
                # If historical year doesn't exist, use earliest year data
                earliest_year = min(self.all_available_years)
                hist_data = self.historical_cache[earliest_year]
                safe_idx = min(file_idx, hist_data['input'].shape[0] - 1)
                historical_inputs.append(hist_data['input'][safe_idx])
                historical_outputs.append(hist_data['output'][safe_idx])
        
        # Convert to tensors
        x_current = torch.tensor(x_current, dtype=torch.float32)
        y_current = torch.tensor(y_current, dtype=torch.float32)
        historical_inputs = torch.stack([torch.tensor(h, dtype=torch.float32) for h in historical_inputs])
        historical_outputs = torch.stack([torch.tensor(h, dtype=torch.float32) for h in historical_outputs])
        
        return x_current, historical_inputs, y_current, historical_outputs

class LearnableHiddenSpatialDataset(Dataset):
    def __init__(self, years, folds, historical_years=5):
        self.years = years
        self.folds = folds
        self.historical_years = historical_years
        
        # Data file paths
        self.input_files = {}
        self.output_files = {}
        
        for year in years:
            for fold in folds:
                key = f"{year}_{fold}"
                self.input_files[key] = f"../../processed_data/TEM-MDM/sequence/spatial/input_{year}_{fold}.npy"
                self.output_files[key] = f"../../processed_data/TEM-MDM/sequence/spatial/output_{year}_{fold}.npy"
        
        # Normalization parameters
        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')
        
        # Calculate data lengths
        self.data_lengths = [np.load(f, mmap_mode='r').shape[0] for f in self.input_files.values()]
        self.cumsum_lengths = np.cumsum([0] + self.data_lengths)
        
        # Precompute historical data
        self.historical_cache = self._precompute_historical_data()
    
    def _precompute_historical_data(self):
        """Precompute historical data"""
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
        # Find corresponding file and sample index
        file_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        sample_idx = idx - self.cumsum_lengths[file_idx]
        
        # Get current data
        current_key = list(self.historical_cache.keys())[file_idx]
        current_data = self.historical_cache[current_key]
        
        x_current = current_data['input'][sample_idx]
        y_current = current_data['output'][sample_idx]
        
        # Collect historical data (simplified processing, use other folds of same year as history)
        historical_inputs = []
        historical_outputs = []
        
        for i, (key, data) in enumerate(self.historical_cache.items()):
            if i < self.historical_years:
                safe_idx = min(sample_idx, data['input'].shape[0] - 1)
                historical_inputs.append(data['input'][safe_idx])
                historical_outputs.append(data['output'][safe_idx])
        
        # If historical data is insufficient, reuse existing data
        while len(historical_inputs) < self.historical_years:
            historical_inputs.append(historical_inputs[-1])
            historical_outputs.append(historical_outputs[-1])
        
        # Convert to tensors
        x_current = torch.tensor(x_current, dtype=torch.float32)
        y_current = torch.tensor(y_current, dtype=torch.float32)
        historical_inputs = torch.stack([torch.tensor(h, dtype=torch.float32) for h in historical_inputs])
        historical_outputs = torch.stack([torch.tensor(h, dtype=torch.float32) for h in historical_outputs])
        
        return x_current, historical_inputs, y_current, historical_outputs

# Setup logging
logging.basicConfig(
    filename=f"log/learnable_hidden_{args.data_type}_{args.type}_{args.inner_model_type}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()

# Create model
model = LearnableHiddenStateLSTMWrapper(
    input_size=config.input_size,
    hidden_size=args.hidden_size,
    output_size=config.output_size,
    historical_years=args.historical_years,
    inner_model_type=args.inner_model_type,
    num_layers=args.num_layers,
    inner_loss_weight=args.inner_loss_weight
).to(config.device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train_learnable_hidden_lstm(epochs: int, train_dataloader: DataLoader, test_dataloader: DataLoader):
    patience = 10 if args.simulated else 100
    min_delta = 1e-4 if args.simulated else 1e-5
    wait = 0
    r2_max = -10.0
    
    for epoch in range(epochs):
        logger.info(f'Epoch {epoch}: Training start.')
        running_loss = 0.0
        running_inner_loss = 0.0
        model.train()
        
        for batch_x, batch_historical_input, batch_y, batch_historical_output in train_dataloader:
            if torch.isnan(batch_x).any():
                continue
            
            # Forward pass
            predictions, total_loss, inner_loss = model(
                batch_x, 
                batch_historical_input, 
                batch_historical_output
            )
            
            # Compute main task loss
            main_loss = criterion(predictions, batch_y)
            total_loss = main_loss + inner_loss * args.inner_loss_weight
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += main_loss.item()
            running_inner_loss += inner_loss.item()
        
        avg_main_loss = math.sqrt(running_loss / len(train_dataloader))
        avg_inner_loss = running_inner_loss / len(train_dataloader)
        logger.info(f'Epoch {epoch}: total loss {(avg_main_loss*23 + avg_inner_loss):.4f}, Main loss {avg_main_loss*23:.4f}, Inner loss {avg_inner_loss:.4f}')
        
        model.eval()
        logger.info(f'Epoch {epoch}: Eval start.')
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_historical_input, batch_y, batch_historical_output in test_dataloader:
                predictions, _, _ = model(
                    batch_x, 
                    batch_historical_input, 
                    batch_historical_output
                )
                all_predictions.append(predictions.cpu().numpy().reshape(-1))
                all_targets.append(batch_y.cpu().numpy().reshape(-1))
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        mask = ~np.isnan(all_targets)
        
        rmse = np.sqrt(mean_squared_error(all_targets[mask], all_predictions[mask]))
        nrmse = rmse / np.mean(all_targets[mask])
        r2 = r2_score(all_targets[mask], all_predictions[mask])
        
        logger.info(f'Testing RMSE {rmse:.4f}, NRMSE {nrmse:.4f}, R2 score {r2:.4f}')
        
        # Early stopping check
        if r2 > r2_max + min_delta:
            r2_max = r2
            wait = 0
            record_param = model.state_dict()
            best_nrmse = nrmse
            best_r2 = r2
            
            if args.simulated:
                torch.save(record_param, f'../model_save/learnable_hidden_lstm/base_{args.inner_model_type}.pth')
                logger.info(f"New best model saved at epoch {epoch} with R2 = {r2:.4f}")
            else:
                torch.save(record_param, f'../model_save/learnable_hidden_lstm/base_obs_{args.inner_model_type}.pth')
                logger.info(f"New best model saved at epoch {epoch} with R2 = {r2:.4f}")
        else:
            wait += 1
            logger.info(f"No improvement in R2 for {wait} epoch(s)")
        
        if wait >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    return best_nrmse, best_r2

def main():
    if args.simulated:
        if args.type == "temporal":
            logger.info(f'Training learnable hidden LSTM with {args.historical_years} historical years')
            all_years = range(1979, 2019)
            train_years = range(1979, 1999)
            test_years = range(1999, 2019)
            
            train_dataset = LearnableHiddenTemporalDataset(
                train_years, all_years, args.data_type, args.historical_years
            )
            train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            
            test_dataset = LearnableHiddenTemporalDataset(
                test_years, all_years, args.data_type, args.historical_years
            )
            test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            
            train_learnable_hidden_lstm(args.epoch, train_dataloader, test_dataloader)
            
        else:  # spatial
            years = range(1979, 2019)
            train_folds = [1, 2, 3, 4]
            test_folds = [0]
            
            train_dataset = LearnableHiddenSpatialDataset(years, train_folds, args.historical_years)
            train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            
            test_dataset = LearnableHiddenSpatialDataset(years, test_folds, args.historical_years)
            test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            
            train_learnable_hidden_lstm(args.epoch, train_dataloader, test_dataloader)
    
    else:  # FLUXNET data
        if args.type == "temporal":
            # 加载FLUXNET数据
            test_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_x.npy'), dtype=torch.float32)
            test_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_y.npy'), dtype=torch.float32)
            train_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/train_data_x.npy'), dtype=torch.float32)
            train_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/train_data_y.npy'), dtype=torch.float32)
            
            # 标准化
            mean = torch.tensor(np.load('../../data/TEM-MDM/mean_vals.npy'))
            std = torch.tensor(np.load('../../data/TEM-MDM/std_vals.npy'))
            test_x = (test_x - mean) / std
            train_x = (train_x - mean) / std
            
            # Create simplified dataset (FLUXNET has no multi-year historical data, use current data as history)
            class FLUXNETDataset(Dataset):
                def __init__(self, x, y, historical_years=5):
                    self.x = x
                    self.y = y
                    self.historical_years = historical_years
                
                def __len__(self):
                    return len(self.y)
                
                def __getitem__(self, idx):
                    # Use current data as historical data (simplified processing)
                    historical_inputs = torch.stack([self.x[idx] for _ in range(self.historical_years)])
                    historical_outputs = torch.stack([self.y[idx] for _ in range(self.historical_years)])
                    return self.x[idx], historical_inputs, self.y[idx], historical_outputs
            
            train_dataset = FLUXNETDataset(train_x.to(config.device), train_y.to(config.device), args.historical_years)
            test_dataset = FLUXNETDataset(test_x.to(config.device), test_y.to(config.device), args.historical_years)
            
            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
            
            train_learnable_hidden_lstm(args.epoch, train_dataloader, test_dataloader)
        
        else:  # spatial FLUXNET
            # 类似的空间数据处理
            test = 4
            train_x_list, train_y_list = [], []
            
            for i in range(5):
                if test == i:
                    test_x = torch.tensor(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_x_{i}.npy"), dtype=torch.float32)
                    test_y = torch.tensor(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_y_{i}.npy"), dtype=torch.float32)
                else:
                    train_x_list.append(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_x_{i}.npy"))
                    train_y_list.append(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_y_{i}.npy"))
            
            train_x = torch.tensor(np.concatenate(train_x_list, axis=0), dtype=torch.float32)
            train_y = torch.tensor(np.concatenate(train_y_list, axis=0), dtype=torch.float32)
            
            # Normalize
            mean = torch.tensor(np.load('../../data/TEM-MDM/mean_vals.npy'))
            std = torch.tensor(np.load('../../data/TEM-MDM/std_vals.npy'))
            test_x = (test_x - mean) / std
            train_x = (train_x - mean) / std
            
            # Create dataset
            train_dataset = FLUXNETDataset(train_x.to(config.device), train_y.to(config.device), args.historical_years)
            test_dataset = FLUXNETDataset(test_x.to(config.device), test_y.to(config.device), args.historical_years)
            
            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
            
            train_learnable_hidden_lstm(args.epoch, train_dataloader, test_dataloader)

if __name__ == "__main__":
    main()
