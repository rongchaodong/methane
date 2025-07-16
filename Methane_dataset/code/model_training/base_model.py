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

from hybrid import LSTM_Transformer, Contextual_LSTM_Transformer

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
        self.year_to_idx = {year: i for i, year in enumerate(range(1979, 1979 + config.max_num_years))}
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

        # 获取真实的年份
        actual_year = self.years[year_idx]
        # 从年份到全局索引的转换
        global_year_idx = self.year_to_idx[actual_year]

        input_data = (np.load(self.input_files[year_idx], mmap_mode='r')[file_idx] - self.mean_vals) / self.std_vals 
        output_data = np.load(self.output_files[year_idx], mmap_mode='r')[file_idx] / 23

        return torch.tensor(input_data, dtype=torch.float32).to(config.device), torch.tensor(output_data, dtype=torch.float32).to(config.device), torch.tensor(global_year_idx, dtype=torch.long).to(config.device)

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

parser = argparse.ArgumentParser(description="training base model for methane dataset")
parser.add_argument("--type", type=str, default="temporal", help="please choose temporal or spatial")
parser.add_argument("--simulated", type=bool, default=True, help="training simulated data or observed data")
parser.add_argument("--model", type=str, help="Model selection")
parser.add_argument("--epoch", type=int, default=100, help="Training epoches")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
args = parser.parse_args()

config = Config()
criterion = nn.MSELoss()

logging.basicConfig(
    filename=f"log/base_{args.type}_{args.model}.log" if args.simulated else f"log/base_obs_{args.type}_{args.model}.log", 
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO 
)

logger = logging.getLogger()

#model selection
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
    # model = LSTM_Transformer(config).to(config.device)
    model = Contextual_LSTM_Transformer(config).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train(epoches : int, train_dataloader: DataLoader, test_dataloader: DataLoader):
    #hyperparameter
    patience = 5
    min_delta = 1e-4
    wait = 0
    r2_max = -10.0

    for epoch in range(epoches):
        logger.info(f'Epoch {epoch}: Training start.')
        running_loss = 0.0
        model.train()
        for batch_X, batch_y, batch_year_indices in train_dataloader:
            if torch.isnan(batch_X).any():
                continue
            _, prediction = model(batch_X, batch_year_indices)
            # print("prediction shape:", prediction.shape)
            # print("batch_y shape:", batch_y.shape)

            loss = criterion(prediction, batch_y)  # 保证 shape: [B, 1]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_loss = math.sqrt(running_loss / len(train_dataloader))
        logger.info(f'Epoch {epoch}: Training loss {avg_loss*23:.4f}')

        model.eval()
        logger.info(f'Epoch {epoch}: eval start.')
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y, batch_year_indices in test_dataloader:
                _, prediction = model(batch_X, batch_year_indices)
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
            best_nrmse = nrmse
            best_r2 = r2
            torch.save(record_param, f'../model_save/{args.model}/base_model.pth')
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
            train_years = range(1979, 1999)
            train_dataset = TemporalDataset(train_years)
            train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            test_years = range(1999, 2019)
            test_dataset = TemporalDataset(test_years)
            test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
            # model train
            train(args.epoch, train_dataloader, test_dataloader)         
        else: # simulated spatial dataset
            years = range(1979, 2019)
            nrmse_list = []
            r2_list = []
            for test in range(5): 
                train_folds = [i for i in range(5) if i not in [test]]
                test_folds = [test]
                train_dataset = SpatialDataset(years, train_folds)
                test_dataset = SpatialDataset(years, test_folds)
                train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
                test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
                nrmse, r2 = train(args.epoch, train_dataloader, test_dataloader)
                nrmse_list.append(nrmse)
                r2_list.append(r2)
            logger.info(f'testing average nrmse {sum(nrmse_list)/len(nrmse_list):.4f}, r2 score {sum(r2_list)/len(r2_list):.4f}')
    else: # flux
        if args.type == "temporal":
            test_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_x.npy'), dtype=torch.float32)
            test_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_y.npy'), dtype=torch.float32) * 1.38
            train_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/train_data_x.npy'), dtype=torch.float32)
            train_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/train_data_y.npy'), dtype=torch.float32) * 1.38
            mean = torch.tensor(np.load('../../data/TEM-MDM/mean_vals.npy'))
            # this file is missing, I copy from data/
            std = torch.tensor(np.load('../../processed_data/TEM-MDM/std_vals.npy'))
            test_x = (test_x - mean) / std
            test_y = test_y + 100
            train_x = (train_x - mean) / std
            train_y = train_y + 100
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_dataset = Dataset(test_x.to(device), test_y.to(device))
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
            train_dataset = Dataset(train_x.to(device), train_y.to(device))
            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)
            train(args.epoch, train_dataloader, test_dataloader)
        else:
            nrmse_list = []
            r2_list = []
            for test in range(5): 
                train_x_list, train_y_list = [], []
                for i in range(5):
                    if test == i:
                        test_x = torch.tensor(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_x_{i}.npy"), dtype=torch.float32)
                        test_y = torch.tensor(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_y_{i}.npy"), dtype=torch.float32) * 1.38
                    else:
                        train_x_list.append(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_x_{i}.npy"))
                        train_y_list.append(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_y_{i}.npy"))
                train_x = torch.tensor(np.concatenate(train_x_list, axis=0), dtype=torch.float32)
                train_y = torch.tensor(np.concatenate(train_y_list, axis=0), dtype=torch.float32) * 1.38
                test_dataset = Dataset(test_x.to(device), test_y.to(device))
                test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
                train_dataset = Dataset(train_x.to(device), train_y.to(device))
                train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)
                nrmse, r2 = train(args.epoch, train_dataloader, test_dataloader)
                nrmse_list.append(nrmse)
                r2_list.append(r2)
            logger.info(f'testing average nrmse {sum(nrmse_list)/len(nrmse_list):.4f}, r2 score {sum(r2_list)/len(r2_list):.4f}')

main()