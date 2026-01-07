import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import math
import logging
from sklearn.metrics import mean_squared_error, r2_score
from config import Config
from learnable_hierarchical import LearnableHierarchicalWrapper, save_location_weights
from dataset import HierarchicalTemporalDataset
from lstm_embedding import LSTM_Embedding
import os

config = Config()
criterion = nn.MSELoss()

parser = argparse.ArgumentParser(description="training learnable hierarchical model for methane dataset")
parser.add_argument("--model", type=str, default="learnable_hierarchical", choices=["learnable_hierarchical", "lstm_embedding"], help="Model type to train")
parser.add_argument("--epoch", type=int, default=100, help="Training epoches")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--annual_hidden_size", type=int, default=16, help="Hidden size for annual encoder")
parser.add_argument("--daily_hidden_size", type=int, default=8, help="Hidden size for daily decoder")
parser.add_argument("--inner_loss_weight", type=float, default=1.0, help="Weight for inner model loss")
parser.add_argument("--context_years", type=int, default=4, help="Number of context years for hierarchical model")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--pretrained_model", action="store_true", help="Use pretrained model for fine-tuning")
parser.add_argument("--fine_tune_epochs", type=int, default=100, help="Number of epochs for fine-tuning on FLUXNET")
args = parser.parse_args()

config.num_context_years = args.context_years
config.batch_size = args.batch_size

# Setup logging
logging.basicConfig(
    filename=f"log/ablation_{args.model}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()

model = None
optimizer = None
scheduler = None

def train_learnable_hierarchical(epochs: int, train_dataloader: DataLoader, test_dataloader: DataLoader, model, optimizer, scheduler):
    patience = 10
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
        
        output_dir = f'../model_save/{args.model}'
        os.makedirs(output_dir, exist_ok=True)
        # Early stopping check
        if r2 > r2_max + min_delta:
            r2_max = r2
            wait = 0
            record_param = model.state_dict()
            best_nrmse = nrmse
            best_r2 = r2
            
            torch.save(record_param, f'../model_save/{args.model}/base_model-{args.context_years}.pth')
            logger.info(f"New best model and location weights saved at epoch {epoch} with R2 = {r2:.4f}")
        else:
            wait += 1
            logger.info(f"No improvement in R2 for {wait} epoch(s)")
        
        if wait >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    return best_nrmse, best_r2

def main():
    global model, optimizer, scheduler
    
    logger.info(f'Training learnable hierarchical model with {config.num_context_years} context years')
    all_years = range(1979, 2019)
    train_years = range(1979 + config.num_context_years, 1999)
    test_years = range(1999, 2019)
    
    train_dataset = HierarchicalTemporalDataset(train_years, config, all_years)
    logger.info(f"training with {train_dataset.__location_length__()} locations")

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataset = HierarchicalTemporalDataset(test_years, config, all_years)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    if args.model == "lstm_embedding":
        model = LSTM_Embedding(config).to(config.device)
        print("Using LSTM Embedding Model")
    else:
        model = LearnableHierarchicalWrapper(
            input_dim=config.input_size,  # 15 features
            annual_hidden_dim=args.annual_hidden_size,
            daily_hidden_dim=args.daily_hidden_size,
            output_dim=config.output_size,
            inner_loss_weight=args.inner_loss_weight,
            enable_contrastive=False,
            enable_attn=False,
            dataset=train_dataset
        ).to(config.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    train_learnable_hierarchical(args.epoch, train_dataloader, test_dataloader, model, optimizer, scheduler)

if __name__ == "__main__":
    main()
