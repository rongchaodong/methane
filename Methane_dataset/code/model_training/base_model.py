import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Dataset
import math
import logging
from sklearn.metrics import mean_squared_error, r2_score

from config import Config
from model import LSTM, MyEALSTM, MultiTCN, Transformer, iTransformer, Pyraformer

from hybrid import LSTM_Transformer

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

        input_data = (np.load(self.input_files[year_idx], mmap_mode='r')[file_idx] - self.mean_vals) / self.std_vals 
        output_data = np.load(self.output_files[year_idx], mmap_mode='r')[file_idx] / 23

        return torch.tensor(input_data, dtype=torch.float32).to(config.device), torch.tensor(output_data, dtype=torch.float32).to(config.device)

parser = argparse.ArgumentParser(description="Pretraining script for methane dataset")
parser.add_argument("--model", type=str, help="Model selection")
parser.add_argument("--epoch", type=int, default=100, help="Training epoches")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
args = parser.parse_args()

config = Config()
criterion = nn.MSELoss()

logging.basicConfig(
    filename=f"log/base_{args.model}.log", 
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO 
)

logger = logging.getLogger()

train_years = range(1979, 1999)
train_dataset = TemporalDataset(train_years)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

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
elif args.model == 'hybrid':
    model = LSTM_Transformer(config).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_loss_list, test_loss_list, r2_list = [], [], []

test_reality = False
if test_reality:
    test_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_x.npy'), dtype=torch.float32)
    test_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_y.npy'), dtype=torch.float32) * 1.38
    mean = torch.tensor(np.load('../../data/TEM-MDM/mean_vals.npy'))
    # this file is missing, I copy from data/
    std = torch.tensor(np.load('../../processed_data/TEM-MDM/std_vals.npy'))

    # test_x = test_x[:, :, indices_to_keep]
    # mean = mean[:, :, indices_to_keep]
    # std = std[:, :, indices_to_keep]
    test_x = (test_x - mean) / std
    test_y = test_y + 100
    # print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = Dataset(test_x.to(device), test_y.to(device))
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
else:
    test_years = range(1999, 2019)
    test_dataset = TemporalDataset(test_years)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

patience = 5
min_delta = 1e-4
wait = 0
r2_max = -10.0

for epoch in range(args.epoch):
    logger.info(f'Epoch {epoch}: Training start.')
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
    logger.info(f'Epoch {epoch}: Training loss {avg_loss*23:.4f}')
    train_loss_list.append(avg_loss)

    model.eval()
    logger.info(f'Epoch {epoch}: eval start.')
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            _, prediction = model(batch_X)
            all_predictions.append(prediction.cpu().numpy().reshape(-1))
            all_targets.append(batch_y.cpu().numpy().reshape(-1))

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mask = ~np.isnan(all_targets)

    rmse = np.sqrt(mean_squared_error(all_targets[mask], all_predictions[mask]))
    nrmse = rmse / np.mean(all_targets[mask])
    r2 = r2_score(all_targets[mask], all_predictions[mask])

    logger.info(f'testing rmse {rmse:.4f}, nrmse {nrmse:.4f}, r2 score {r2:.4f}')

    # early stopping check
    if r2 > r2_max + min_delta:
        r2_max = r2
        wait = 0
        record_param = model.state_dict()
        torch.save(record_param, f'../model_save/{args.model}/base_model.pth')
        logger.info(f"New best model saved at epoch {epoch} with R2 = {r2:.4f}")
    else:
        wait += 1
        logger.info(f"No improvement in R2 for {wait} epoch(s)")

    if wait >= patience:
        logger.info(f"Early stopping triggered at epoch {epoch}")
        break
