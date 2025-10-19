import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Dataset
import math
from torch.utils.data import Dataset
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, r2_score
from config import Config
from learn_fluxnet import LearnableHierarchicalWrapper

config = Config()
criterion = nn.MSELoss()

parser = argparse.ArgumentParser(description="training FLUXNET with pretrained model and location weight matching")
parser.add_argument("--type", type=str, default="temporal", help="please choose temporal or spatial")
parser.add_argument("--pretrained_model", action="store_true", help="Use pretrained model for fine-tuning")
parser.add_argument("--epoch", type=int, default=100, help="Training epoches")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--annual_hidden_size", type=int, default=16, help="Hidden size for annual encoder")
parser.add_argument("--daily_hidden_size", type=int, default=8, help="Hidden size for daily decoder")
parser.add_argument("--context_years", type=int, default=4, help="Number of context years for hierarchical model")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
args = parser.parse_args()

config.num_context_years = args.context_years
config.batch_size = args.batch_size

# Setup logging
log_filename = f"log/fluxnet_pretrained_{args.type}.log" if args.pretrained_model else f"log/fluxnet_from_scratch_{args.type}.log"
logging.basicConfig(
    filename=log_filename,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()

def save_location_weights(model, save_path):
    location_weights = {}

    for i, projection in enumerate(model.model.decoder.output_projections):
        location_weights[f'location_{i}'] = {
            'weight': projection.weight.data.clone(),
            'bias': projection.bias.data.clone() if projection.bias is not None else None
        }
    
    torch.save(location_weights, save_path)
    print(f"Saved location weights to {save_path}")

def load_pretrained_model_with_location_matching(pretrained_model_path, pretrained_weights, fluxnet_model):
    pretrained_state = torch.load(pretrained_model_path, map_location='cpu', weights_only=True)
    current_state = fluxnet_model.state_dict()
    
    for key, value in pretrained_state.items():
        if key in current_state:
            print(f"Loading weight: {key}")
            if current_state[key].shape == value.shape:
                current_state[key] = value
                print(f"Loaded: {key}")
                if 'historical_weight' in key:
                    print(value)
            else:
                print(f"Shape mismatch for {key}: {current_state[key].shape} vs {value.shape}")
    
    fluxnet_model.load_state_dict(current_state)
    fluxnet_model.model.decoder.pretrained_location_weights = pretrained_weights
    print("Successfully loaded pretrained model with location weight matching capability")

    return fluxnet_model

def train_fluxnet_model(epochs: int, train_dataloader: DataLoader, test_dataloader: DataLoader, model, optimizer, scheduler):
    patience = 10  # Early stopping patience
    min_delta = 1e-4
    wait = 0
    r2_max = -10.0
    
    logger.info(f"model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(epochs):
        logger.info(f'Epoch {epoch}: Training start.')
        running_loss = 0.0
        running_inner_loss = 0.0
        running_contrastive_loss = 0.0
        model.train()
        
        for batch_x_current, batch_y, batch_file_idx in train_dataloader:
            if torch.isnan(batch_x_current).any():
                continue
            
            # Forward pass
            predictions = model(
                batch_x_current, 
                batch_file_idx
            )
            
            # Compute main task loss
            # print("predictions shape:", predictions.shape)
            # print("batch_y shape:", batch_y.shape)
            # exit()
            main_loss = criterion(predictions, batch_y)
            
            optimizer.zero_grad()
            main_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += main_loss.item()

        avg_main_loss = math.sqrt(running_loss / len(train_dataloader))
        logger.info(f'Epoch {epoch}: total loss {avg_main_loss*23:.4f}')

        model.eval()
        logger.info(f'Epoch {epoch}: Eval start.')
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x_current, batch_y, batch_file_idx in test_dataloader:
                predictions = model(
                    batch_x_current, 
                    batch_file_idx
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
            
            # Save model
            model_suffix = "pretrained" if args.pretrained_model else "from_scratch"
            torch.save(record_param, f'../model_save/hierarchical/fluxnet_{model_suffix}_{args.type}.pth')
            save_location_weights(model, f'../model_save/hierarchical/fluxnet_location_weights_{model_suffix}_{args.type}.pth')
            logger.info(f"New best model saved at epoch {epoch} with R2 = {r2:.4f}")
        else:
            wait += 1
            logger.info(f"No improvement in R2 for {wait} epoch(s)")
        
        if wait >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    return best_nrmse, best_r2

def main():
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
        
        # Create simplified dataset for FLUXNET (no historical data needed)
        class FLUXNETHierarchicalDataset(Dataset):
            def __init__(self, x, y, config):
                self.x = x
                self.y = y
                self.config = config
                # For FLUXNET, the number of locations is the number of samples
                self.max_locations = len(self.y)
            
            def __len__(self):
                return len(self.y)
            
            def __location_length__(self):
                return self.max_locations
            
            def __getitem__(self, idx):
                # For FLUXNET, we don't use historical data, just current data
                x_current = self.x[idx]  # (365, 15) - all features
                # Create dummy historical inputs (not used in FLUXNET training)
                # historical_inputs = torch.zeros(config.num_context_years, 365, 15, device=config.device)
                # historical_targets = torch.zeros(config.num_context_years, 365, device=config.device)
                
                return x_current, self.y[idx], torch.tensor(idx, dtype=torch.long)
        
        train_dataset = FLUXNETHierarchicalDataset(train_x.to(config.device), train_y.to(config.device), config)
        test_dataset = FLUXNETHierarchicalDataset(test_x.to(config.device), test_y.to(config.device), config)
        
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        # Create model with the actual dataset
        model = LearnableHierarchicalWrapper(
            input_dim=config.input_size,  # 15 features
            annual_hidden_dim=args.annual_hidden_size,
            daily_hidden_dim=args.daily_hidden_size,
            output_dim=config.output_size,
            dataset=train_dataset
        ).to(config.device)
        
        # Load pretrained model if specified
        if args.pretrained_model:
            logger.info("Loading pretrained model with location weight matching...")
            pretrained_model_path = f'../model_save/hierarchical/base_model-seq-{args.context_years}.pth'
            pretrained_location_weights_path = f'../model_save/hierarchical/location_weights-1percent-{args.context_years}.pth'
            
            try:
                pretrained_location_weights = torch.load(pretrained_location_weights_path, map_location='cpu', weights_only=True)
                model = load_pretrained_model_with_location_matching(
                    pretrained_model_path, 
                    pretrained_location_weights,
                    model
                )
                logger.info("Pretrained model loaded successfully with location weight matching")
            except Exception as e:
                logger.error(f"Failed to load pretrained model: {e}")
                logger.info("Continuing with randomly initialized model")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        train_fluxnet_model(args.epoch, train_dataloader, test_dataloader, model, optimizer, scheduler)
    
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
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create model with the actual dataset
        model = LearnableHierarchicalWrapper(
            input_dim=config.input_size,  # 15 features
            annual_hidden_dim=args.annual_hidden_size,
            daily_hidden_dim=args.daily_hidden_size,
            output_dim=config.output_size,
            inner_loss_weight=args.inner_loss_weight,
            enable_contrastive=True,
            dataset=train_dataset
        ).to(config.device)
        
        # Load pretrained model if specified
        if args.pretrained_model:
            logger.info("Loading pretrained model with location weight matching...")
            pretrained_model_path = f'../model_save/hierarchical/base_model-{args.context_years}.pth'
            pretrained_location_weights_path = f'../model_save/hierarchical/location_weights-{args.context_years}.pth'
            
            try:
                model = load_pretrained_model_with_location_matching(
                    pretrained_model_path, 
                    pretrained_location_weights_path, 
                    model
                )
                logger.info("Pretrained model loaded successfully with location weight matching")
            except Exception as e:
                logger.error(f"Failed to load pretrained model: {e}")
                logger.info("Continuing with randomly initialized model")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        train_fluxnet_model(args.epoch, train_dataloader, test_dataloader, model, optimizer, scheduler)

if __name__ == "__main__":
    main()
