import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Dataset
import math
import logging
from sklearn.metrics import mean_squared_error, r2_score
from config import Config
from learnable_hierarchical import LearnableHierarchicalWrapper, HierarchicalTemporalDataset, HierarchicalSpatialDataset

config = Config()
criterion = nn.MSELoss()

parser = argparse.ArgumentParser(description="training learnable hierarchical model for methane dataset")
parser.add_argument("--type", type=str, default="temporal", help="please choose temporal or spatial")
parser.add_argument("--simulated", action="store_true", help="training simulated data or observed data")
parser.add_argument("--data_type", type=str, default="seq", help="Model selection: random, or seq")
parser.add_argument("--epoch", type=int, default=100, help="Training epoches")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--annual_hidden_size", type=int, default=16, help="Hidden size for annual encoder")
parser.add_argument("--daily_hidden_size", type=int, default=8, help="Hidden size for daily decoder")
parser.add_argument("--inner_loss_weight", type=float, default=1.0, help="Weight for inner model loss")
parser.add_argument("--context_years", type=int, default=4, help="Number of context years for hierarchical model")
args = parser.parse_args()

config.num_context_years = args.context_years
# Setup logging
logging.basicConfig(
    filename=f"log/learnable_hierarchical_{args.data_type}_{args.type}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()

# Create model
model = LearnableHierarchicalWrapper(
    input_dim=config.input_size,  # 15 features
    annual_hidden_dim=args.annual_hidden_size,
    daily_hidden_dim=args.daily_hidden_size,
    output_dim=config.output_size,
    inner_loss_weight=args.inner_loss_weight
).to(config.device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

def train_learnable_hierarchical(epochs: int, train_dataloader: DataLoader, test_dataloader: DataLoader):
    patience = 10 if args.simulated else 100
    min_delta = 1e-4 if args.simulated else 1e-5
    wait = 0
    r2_max = -10.0
    
    for epoch in range(epochs):
        logger.info(f'Epoch {epoch}: Training start.')
        running_loss = 0.0
        running_inner_loss = 0.0
        model.train()
        
        for batch_x_current, batch_historical_inputs, batch_y, batch_historical_targets in train_dataloader:
            if torch.isnan(batch_x_current).any():
                continue
            
            # Forward pass
            predictions, total_loss, inner_loss = model(
                batch_x_current, 
                batch_historical_inputs, 
                batch_historical_targets
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
            for batch_x_current, batch_historical_inputs, batch_y, batch_historical_targets in test_dataloader:
                predictions, _, _ = model(
                    batch_x_current, 
                    batch_historical_inputs, 
                    batch_historical_targets
                )
                all_predictions.append(predictions.cpu().numpy().reshape(-1))
                all_targets.append(batch_y.cpu().numpy().reshape(-1))
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        mask = ~np.isnan(all_targets)
        
        rmse = np.sqrt(mean_squared_error(all_targets[mask], all_predictions[mask]))
        nrmse = rmse / np.mean(all_targets[mask])
        r2 = r2_score(all_targets[mask], all_predictions[mask])

        scheduler.step(r2)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f'LR {current_lr} Testing RMSE {rmse:.4f}, NRMSE {nrmse:.4f}, R2 score {r2:.4f}')
        
        # Early stopping check
        if r2 > r2_max + min_delta:
            r2_max = r2
            wait = 0
            record_param = model.state_dict()
            best_nrmse = nrmse
            best_r2 = r2
            
            if args.simulated:
                torch.save(record_param, f'../model_save/hierarchical/base_model-{args.context_years}.pth')
                logger.info(f"New best model saved at epoch {epoch} with R2 = {r2:.4f}")
            else:
                torch.save(record_param, f'../model_save/hierarchical/base_obs.pth')
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
            logger.info(f'Training learnable hierarchical model with {config.num_context_years} context years')
            all_years = range(1979, 2019)
            train_years = range(1979 + config.num_context_years, 1999)
            test_years = range(1999, 2019)
            
            train_dataset = HierarchicalTemporalDataset(
                train_years, config, all_years, args.data_type
            )

            if args.data_type == "1percent":
                train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            else:
                train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

            test_dataset = HierarchicalTemporalDataset(
                test_years, config, all_years, args.data_type
            )

            if args.data_type == "1percent":
                test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            else:
                test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

            train_learnable_hierarchical(args.epoch, train_dataloader, test_dataloader)
            
        else:  # spatial
            years = range(1979, 2019)
            train_folds = [1, 2, 3, 4]
            test_folds = [0]
            
            train_dataset = HierarchicalSpatialDataset(years, config, years, train_folds)
            train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            
            test_dataset = HierarchicalSpatialDataset(years, config, years, test_folds)
            test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            
            train_learnable_hierarchical(args.epoch, train_dataloader, test_dataloader)
    
    else:  # FLUXNET data
        if args.type == "temporal":
            # Load FLUXNET data
            test_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_x.npy'), dtype=torch.float32)
            test_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_y.npy'), dtype=torch.float32)
            train_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/train_data_x.npy'), dtype=torch.float32)
            train_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/train_data_y.npy'), dtype=torch.float32)
            
            # Normalize
            mean = torch.tensor(np.load('../../data/TEM-MDM/mean_vals.npy'))
            std = torch.tensor(np.load('../../data/TEM-MDM/std_vals.npy'))
            test_x = (test_x - mean) / std
            train_x = (train_x - mean) / std
            
            # Create simplified dataset (FLUXNET has no multi-year historical data, use current data as history)
            class FLUXNETHierarchicalDataset(Dataset):
                def __init__(self, x, y, config):
                    self.x = x
                    self.y = y
                    self.config = config
                
                def __len__(self):
                    return len(self.y)
                
                def __getitem__(self, idx):
                    # Use current data as historical data (simplified processing)
                    x_current = self.x[idx]  # (365, 15) - all features
                    historical_inputs = self.x[idx].unsqueeze(0).repeat(config.num_context_years, 1, 1)  # (num_context_years, 365, 15)
                    historical_targets = self.y[idx].unsqueeze(0).repeat(config.num_context_years, 1)  # (num_context_years, 365)
                    
                    return x_current, historical_inputs, self.y[idx], historical_targets
            
            train_dataset = FLUXNETHierarchicalDataset(train_x.to(config.device), train_y.to(config.device), config)
            test_dataset = FLUXNETHierarchicalDataset(test_x.to(config.device), test_y.to(config.device), config)
            
            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
            
            train_learnable_hierarchical(args.epoch, train_dataloader, test_dataloader)
        
        else:  # spatial FLUXNET
            # Similar spatial data processing
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
            train_dataset = FLUXNETHierarchicalDataset(train_x.to(config.device), train_y.to(config.device), config)
            test_dataset = FLUXNETHierarchicalDataset(test_x.to(config.device), test_y.to(config.device), config)
            
            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
            
            train_learnable_hierarchical(args.epoch, train_dataloader, test_dataloader)

if __name__ == "__main__":
    main()
