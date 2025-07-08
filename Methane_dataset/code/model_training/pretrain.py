import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Dataset
import math
import json
import logging
from sklearn.metrics import mean_squared_error, r2_score

from config import Config
from model import LSTM, MyEALSTM, MultiTCN, Transformer, iTransformer, Pyraformer

class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TemporalDataset(Dataset):
    def __init__(self, years):
        self.years = years
        self.input_files = [f"../../processed_data/TEM-MDM/input_{year}.npy" for year in years]
        self.output_files = [f"../../processed_data/TEM-MDM/output_{year}.npy" for year in years]
        self.data_lengths = [np.load(f, mmap_mode='r').shape[0] for f in self.input_files] 
        self.cumsum_lengths = np.cumsum([0] + self.data_lengths)  
        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')

    def __len__(self):
        return self.cumsum_lengths[-1]  

    def __getitem__(self, idx):
        year_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        file_idx = idx - self.cumsum_lengths[year_idx]

        input_data = (np.load(self.input_files[year_idx], mmap_mode='r')[file_idx] - self.mean_vals) / self.std_vals 
        output_data = np.load(self.output_files[year_idx], mmap_mode='r')[file_idx] / 23

        return torch.tensor(input_data, dtype=torch.float32).to(config.device), torch.tensor(output_data, dtype=torch.float32).to(config.device)
    
class SpatialDataset(Dataset):
    def __init__(self, years, folds):
        self.years = years
        self.folds = folds 
        self.device = config.device
        self.input_files = [f"../../processed_data/TEM-MDM/spatial/input_{year}_{fold}.npy" for year in years for fold in folds]
        self.output_files = [f"../../processed_data/TEM-MDM/spatial/output_{year}_{fold}.npy" for year in years for fold in folds]
        self.data_lengths = [np.load(f, mmap_mode='r').shape[0] for f in self.input_files]
        self.cumsum_lengths = np.cumsum([0] + self.data_lengths)

        self.mean_vals = np.load('../../data/TEM-MDM/mean_vals.npy')
        self.std_vals = np.load('../../data/TEM-MDM/std_vals.npy')

    def __len__(self):
        return self.cumsum_lengths[-1]

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumsum_lengths, idx, side="right") - 1
        sample_idx = idx - self.cumsum_lengths[file_idx]

        input_data = (np.load(self.input_files[file_idx], mmap_mode='r')[sample_idx] - self.mean_vals) / self.std_vals
        output_data = np.load(self.output_files[file_idx], mmap_mode='r')[sample_idx] / 23

        return torch.tensor(input_data, dtype=torch.float32).to(self.device), torch.tensor(output_data, dtype=torch.float32).to(self.device)

parser = argparse.ArgumentParser(description="Pretraining script for methane dataset")
parser.add_argument("--valid_type", type=str, help="Use temporal dataset or spatial dataset")
parser.add_argument("--spatial_fold", type=int, default=0, help="The fold using as cross validation in spatial experiment")
parser.add_argument("--model", type=str, help="Model selection")
parser.add_argument("--id", type=str, help="Experiment identifier")
parser.add_argument("--epoch", type=int, default=10, help="Training epoches")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
args = parser.parse_args()

config = Config()
criterion = nn.MSELoss()

if args.valid_type == 'temporal':
    logging.basicConfig(
        filename=f"log/pretrain_{args.model}_{args.valid_type}_{args.id}.log", 
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO 
    )
else:
    logging.basicConfig(
        filename=f"log/pretrain_{args.model}_{args.valid_type}_fold{args.spatial_fold}.log", 
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO 
    )

logger = logging.getLogger()

logger.info("Command-line Arguments:")
logger.info(json.dumps(vars(args), indent=4)) 

logger.info("Model Configuration:")
logger.info(json.dumps(vars(config), indent=4))

if args.valid_type == 'temporal':
    train_years = range(1979, 1999)
    test_years = range(1999, 2019)
    train_dataset = TemporalDataset(train_years)
    test_dataset = TemporalDataset(test_years)
else:
    years = range(1979, 2019)
    train_folds = [i for i in range(5) if i not in [args.spatial_fold]]
    test_folds = [args.spatial_fold]
    train_dataset = SpatialDataset(years, train_folds)
    test_dataset = SpatialDataset(years, test_folds)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

if args.model == 'lstm':
    model = LSTM(config).to(config.device)
elif args.model == 'ealstm':
    model = MyEALSTM(config).to(config.device)
elif args.model == 'tcn':
    model = MultiTCN(config).to(config.device)
elif args.model == 'transformer':
    model = Transformer(config).to(config.device)
elif args.model == 'itransformer':
    model = iTransformer(config).to(config.device)
elif args.model == 'pyraformer':
    model = Pyraformer(config).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_loss_list, test_loss_list, r2_list = [], [], []
for epoch in range(args.epoch):
    running_loss = 0.0
    model.train()
    for batch_X, batch_y in train_dataloader:
        if torch.isnan(batch_X).any():
            continue
        _, prediction = model(batch_X)
        loss = criterion(batch_y, prediction)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
    avg_loss = math.sqrt(running_loss / len(train_dataloader))
    
    model.eval()  
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            if torch.isnan(batch_X).any():
                continue
            _, prediction = model(batch_X)
            all_predictions.append(prediction.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mask = ~np.isnan(all_targets)
    all_targets, all_predictions = all_targets[mask]*23, all_predictions[mask]*23

    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    nrmse = rmse/np.mean(all_targets)
    r2 = r2_score(all_targets, all_predictions)

    logger.info(f'Epoch {epoch}: Training loss {avg_loss*23}, testing rmse {rmse}, nrmse {nrmse}, r2 score {r2}')
    train_loss_list.append(avg_loss)
    test_loss_list.append(rmse)
    r2_list.append(r2)
    if args.valid_type == 'temporal':
        torch.save(model.state_dict(), f'../model_save/{args.model}/pretrain_{args.valid_type}_{args.id}.pth')
    else:
        torch.save(model.state_dict(), f'../model_save/{args.model}/pretrain_{args.valid_type}_fold{args.spatial_fold}.pth')