import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import math
import logging
import time
from sklearn.metrics import mean_squared_error, r2_score
from config import Config
from learnable_hierarchical import LearnableHierarchicalWrapper
from lstm_embedding import LSTM_Embedding
from dataset import FluxnetHierarchicalDataset

config = Config()
criterion = nn.MSELoss()

parser = argparse.ArgumentParser(description="training learnable hierarchical model for methane dataset")
parser.add_argument("--model", type=str, default="learnable_hierarchical", choices=["learnable_hierarchical", "lstm_embedding"], help="Model type to train")
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
    filename=f"log/learnable_h_fluxnet_{args.model}.log",
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
    
        for batch_x_current, batch_historical_inputs, batch_y, batch_historical_targets, batch_file_idx in train_dataloader:
            if torch.isnan(batch_x_current).any():
                continue
            
            if args.model == "lstm_embedding":
                predictions, _ = model(
                    batch_x_current, 
                    batch_historical_inputs, 
                    batch_file_idx,
                    batch_historical_targets
                )
                total_loss = torch.tensor(0.0).to(config.device)
                inner_loss = torch.tensor(0.0).to(config.device)
                contrastive_loss = torch.tensor(0.0).to(config.device)
            else:
                predictions, total_loss, inner_loss, contrastive_loss = model(
                    batch_x_current, 
                    batch_historical_inputs, 
                    batch_file_idx,
                    batch_historical_targets
                )
            
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
            for batch_x_current, batch_historical_inputs, batch_y, batch_historical_targets, batch_file_idx in test_dataloader:
                if args.model == "lstm_embedding":
                    predictions, _ = model(
                        batch_x_current, 
                        batch_historical_inputs, 
                        batch_file_idx,
                        batch_historical_targets
                    )
                else:
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
            output_dir = f'../model_save/{args.model}'
            os.makedirs(output_dir, exist_ok=True)
            time.sleep(1)
            if args.pretrained:
                torch.save(record_param, f'../model_save/{args.model}/base_fine_tuned.pth')
            else:
                torch.save(record_param, f'../model_save/{args.model}/base_obs_norm_le.pth')
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
    logger.info(f'Training learnable hierarchical fluxnet model with {config.num_context_years} context years')
    
    train_dataset = FluxnetHierarchicalDataset(config, mode='train')
    test_dataset = FluxnetHierarchicalDataset(config, mode='test')
    
    if args.model == "lstm_embedding":
        model = LSTM_Embedding(config).to(config.device)
    else:
        model = LearnableHierarchicalWrapper(
            input_dim=config.input_size,  # 15 features
            annual_hidden_dim=args.annual_hidden_size,
            daily_hidden_dim=args.daily_hidden_size,
            output_dim=config.output_size,
            inner_loss_weight=args.inner_loss_weight,
            enable_contrastive=True,
            dataset=train_dataset
        ).to(config.device)
        
    if args.pretrained:
        pretrained_model_path = '../model_save/learnable_hierarchical/base_model-4.pth'
        model = load_pretrained_model(pretrained_model_path, model)
        logger.info("Loaded pretrained model for fine-tuning.")
        # exit()
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=100)
    
    train_learnable_hierarchical(args.epoch, train_dataloader, test_dataloader, model, optimizer, scheduler)


if __name__ == "__main__":
    main()
