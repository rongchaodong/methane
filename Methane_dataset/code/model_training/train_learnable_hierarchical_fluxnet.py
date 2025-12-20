import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Dataset
import math
import logging
from sklearn.metrics import mean_squared_error, r2_score
from config import Config
from learnable_hierarchical import LearnableHierarchicalWrapper, FluxnetHierarchicalDataset
from model import LSTM
import os, pickle

config = Config()
criterion = nn.MSELoss()

parser = argparse.ArgumentParser(description="training learnable hierarchical model for methane dataset")
parser.add_argument("--type", type=str, default="temporal", help="please choose temporal or spatial")
parser.add_argument("--model", type=str, default="learnable_hierarchical", help="please choose model type")
parser.add_argument("--epoch", type=int, default=1000, help="Training epoches")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--annual_hidden_size", type=int, default=16, help="Hidden size for annual encoder")
parser.add_argument("--daily_hidden_size", type=int, default=8, help="Hidden size for daily decoder")
parser.add_argument("--inner_loss_weight", type=float, default=1.0, help="Weight for inner model loss")
parser.add_argument("--context_years", type=int, default=1, help="Number of context years for hierarchical model")
parser.add_argument("--pretrained", action="store_true", help="Use pretrained model for fine-tuning")
args = parser.parse_args()

config.num_context_years = args.context_years
# Setup logging
logging.basicConfig(
    filename=f"log/learnable_pretrained_{args.type}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()

# Create model - will be initialized with dataset later
model = None
optimizer = None
scheduler = None

def load_pretrained_model(pretrained_model_path, fluxnet_model):
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
    print("Successfully loaded pretrained model with location weight matching capability")

    return fluxnet_model

class FluxnetBaselineDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        
        # 数据缓存目录
        data_dir = '../../processed_data/FLUXNET-CH4/site_centric/'
        
        # 1. 加载预处理好的数据
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

        # 2. 加载用于标准化的均值和标准差
        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')
        
        print(f"Loaded baseline {mode} dataset with {len(self.samples)} samples.")

    def __len__(self):
        """返回数据集中的样本总数。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引获取一个样本。
        对于 baseline 模型，只返回 (x, y)。
        """
        # 1. 从样本列表中获取站点名称和年份
        site_name, target_year = self.samples[idx]
        
        # 2. 从缓存中获取该站点该年份的原始数据
        data = self.all_data_cache[site_name][target_year]
        x_raw = data['input']
        y_raw = data['output']
        
        # 3. 对数据进行标准化
        x_normalized = (x_raw - self.mean_vals) / self.std_vals
        y_normalized = y_raw
        
        # 4. 转换为 PyTorch Tensors
        x_tensor = torch.tensor(x_normalized, dtype=torch.float32)
        y_tensor = torch.tensor(y_normalized, dtype=torch.float32)
        
        return x_tensor.to(self.config.device), y_tensor.to(self.config.device)

def train_learnable_hierarchical(epochs: int, train_dataloader: DataLoader, test_dataloader: DataLoader, model, optimizer, scheduler):
    patience = 100 if not args.pretrained else 100
    min_delta = 1e-5
    wait = 0
    r2_max = -10.0
    
    for epoch in range(epochs):
        logger.info(f'Epoch {epoch}: Training start.')
        running_loss = 0.0
        running_inner_loss = 0.0
        running_contrastive_loss = 0.0
        model.train()
        
        if args.model == "lstm":
            for batch_x, batch_y in train_dataloader:
                if torch.isnan(batch_x).any():
                    continue
                
                # Forward pass
                _, predictions = model(batch_x)
                # print("Predictions shape:", predictions.shape)
                # print("Batch y shape:", batch_y.shape)
                # exit()
                
                # Compute loss
                loss = criterion(predictions, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                running_loss += loss.item()
        else:
            for batch_x_current, batch_historical_inputs, batch_y, batch_historical_targets, batch_file_idx in train_dataloader:
                if torch.isnan(batch_x_current).any():
                    continue
                
                # Forward pass
                predictions, total_loss, inner_loss, contrastive_loss = model(
                    batch_x_current, 
                    batch_historical_inputs, 
                    batch_file_idx,
                    batch_historical_targets
                )
                
                # Compute main task loss
                main_loss = criterion(predictions, batch_y)
                total_loss = main_loss + total_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                running_loss += main_loss.item()
                running_inner_loss += inner_loss.item()
                running_contrastive_loss += contrastive_loss.item()

        avg_main_loss = math.sqrt(running_loss / len(train_dataloader))
        avg_inner_loss = running_inner_loss / len(train_dataloader)
        avg_contrastive_loss = running_contrastive_loss / len(train_dataloader)
        logger.info(f'Epoch {epoch}: total loss {(avg_main_loss*23 + avg_inner_loss):.4f}, Main loss {avg_main_loss*23:.4f}, Inner loss {avg_inner_loss:.4f}, Contrastive loss {avg_contrastive_loss:.4f}')

        model.eval()
        logger.info(f'Epoch {epoch}: Eval start.')
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            if args.model == "lstm":
                for batch_x, batch_y in test_dataloader:
                    _, predictions = model(batch_x)
                    all_predictions.append(predictions.cpu().numpy().reshape(-1))
                    all_targets.append(batch_y.cpu().numpy().reshape(-1))
            else:
                for batch_x_current, batch_historical_inputs, batch_y, batch_historical_targets, batch_file_idx in test_dataloader:
                    predictions, _, _, _ = model(
                        batch_x_current, 
                        batch_historical_inputs, 
                        batch_file_idx,
                        batch_historical_targets
                    )
                    # predictions = predictions * train_dataloader.dataset.y_std + train_dataloader.dataset.y_mean
                    predictions_log = predictions.cpu().numpy() * train_dataloader.dataset.y_std + train_dataloader.dataset.y_mean
                    predictions = np.sign(predictions_log) * (np.exp(np.abs(predictions_log)) - 1)
                    all_predictions.append(predictions.reshape(-1))
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
            import time
            if args.model == "lstm":
                torch.save(record_param, f'../model_save/hierarchical/base_lstm_norm.pth')
                # time.sleep(1)
            else:
                if args.pretrained:
                    torch.save(record_param, f'../model_save/hierarchical/base_fine_tuned.pth')
                else:
                    torch.save(record_param, f'../model_save/hierarchical/base_obs_norm_le.pth')
                # time.sleep(1)
            logger.info(f"New best model saved at epoch {epoch} with R2 = {r2:.4f}")
        else:
            wait += 1
            logger.info(f"No improvement in R2 for {wait} epoch(s)")
        
        if wait >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    return best_nrmse, best_r2

def main():
    global model, optimizer, scheduler
    
    # FLUXNET data
    if args.type == "temporal":
        
        if args.model == "lstm":
            model = LSTM(config).to(config.device)
            test_dataset = FluxnetBaselineDataset(config, mode='test')
            train_dataset = FluxnetBaselineDataset(config, mode='train')
        else:
            train_dataset = FluxnetHierarchicalDataset(config, mode='train')
            test_dataset = FluxnetHierarchicalDataset(config, mode='test')
            
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

            if args.pretrained and args.model != "lstm":
                pretrained_model_path = '../model_save/hierarchical/base_model-1percent-6.pth'
                model = load_pretrained_model(pretrained_model_path, model)
                logger.info("Loaded pretrained model for fine-tuning.")
                # exit()
        
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=100)
        
        train_learnable_hierarchical(args.epoch, train_dataloader, test_dataloader, model, optimizer, scheduler)
    
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
        train_dataset = FluxnetHierarchicalDataset(train_x.to(config.device), train_y.to(config.device), config)
        test_dataset = FluxnetHierarchicalDataset(test_x.to(config.device), test_y.to(config.device), config)
        
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
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
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=100)
        
        train_learnable_hierarchical(args.epoch, train_dataloader, test_dataloader, model, optimizer, scheduler)

if __name__ == "__main__":
    main()
