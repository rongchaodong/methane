import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Dataset
import math
import logging
from sklearn.metrics import mean_squared_error, r2_score
from config import Config
from model import LSTM, MyEALSTM, MultiTCN, Transformer, iTransformer, Pyraformer, TSMixer, TimeMixer, PatchTST, P_sLSTM, DUET
from train_learnable_hierarchical_fluxnet import FluxnetBaselineDataset


config = Config()
criterion = nn.MSELoss()

parser = argparse.ArgumentParser(description="training base model for methane dataset")
parser.add_argument("--simulated", action="store_true", help="training simulated data or observed data")
parser.add_argument("--model", type=str, help="Model selection")
parser.add_argument("--data_type", type=str, default="seq", help="Model selection: random, or seq")
parser.add_argument("--epoch", type=int, default=100, help="Training epoches")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--context", type=int, default=4, help="Length of context year as prior knowledge used in hierarchy model")
args = parser.parse_args()

config.num_context_years = args.context


class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TemporalDataset(Dataset):
    def __init__(self, years, data_source: str):
        self.years = years
        self.year_to_idx = {year: i for i, year in enumerate(range(1979, 1979 + config.max_num_years))}
        if args.data_type == "seq":
            if data_source == "full":
                self.input_files = [f"../../processed_data/TEM-MDM/full/temporal/input_{year}.npy" for year in years]
                self.output_files = [f"../../processed_data/TEM-MDM/full/temporal/output_{year}.npy" for year in years]
            elif data_source == "seq":
                self.input_files = [f"../../processed_data/TEM-MDM/sequence/temporal/input_{year}.npy" for year in years]
                self.output_files = [f"../../processed_data/TEM-MDM/sequence/temporal/output_{year}.npy" for year in years]
        elif args.data_type == "random":
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

        return torch.tensor(input_data, dtype=torch.float32).to(config.device), torch.tensor(output_data, dtype=torch.float32).to(config.device)

    def __init__(self, years, folds):
        self.years = years
        self.folds = folds 
        self.device = config.device
        if args.data_type == "seq":
            self.input_files = [f"../../processed_data/TEM-MDM/sequence/spatial/input_{year}_{fold}.npy" for year in years for fold in folds]
            self.output_files = [f"../../processed_data/TEM-MDM/sequence/spatial/output_{year}_{fold}.npy" for year in years for fold in folds]
        elif args.data_type == "random":
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

logging.basicConfig(
    filename=f"log/{args.data_type}_{args.model}.log" if args.simulated else f"log/base_obs_{args.model}.log", 
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
elif args.model == 'tsmixer':
    model = TSMixer(config).to(config.device)
elif args.model == 'timemixer':
    model = TimeMixer(config).to(config.device)
elif args.model == 'patchtst':
    model = PatchTST(config).to(config.device)
elif args.model == 'pslstm':
    model = P_sLSTM(config).to(config.device)
elif args.model == 'duet':
    model = DUET(config).to(config.device)

if args.model == 'hierarchy' and args.data_type == "random" and not args.simulated:
    print("cannot train hierarchy model in random dataset!")
    exit(-1)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# total_params = sum(p.numel() for p in model.parameters())
# print(f"Model {args.model} has: {total_params} parameters.")

def obs_train(epoches : int, train_dataloader: DataLoader, test_dataloader: DataLoader):
    #hyperparameter
    if args.simulated:
        patience = 10
        min_delta = 1e-4
    else:
        patience = 100
        min_delta = 1e-5
    wait = 0
    r2_max = -10.0

    for epoch in range(epoches):
        logger.info(f'Epoch {epoch}: Training start.')
        running_loss = 0.0
        model.train()
        for batch_X, batch_y in train_dataloader:
            if torch.isnan(batch_X).any():
                continue
            # print("batch_X shape:", batch_X.shape)
            _, prediction = model(batch_X)
            # print("prediction shape:", prediction.shape)
            # exit()

            loss = criterion(prediction, batch_y)

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
            best_nrmse = nrmse
            best_r2 = r2
            if args.simulated:
                torch.save(record_param, f'../model_save/{args.model}/base.pth')
                logger.info(f"New best model saved at epoch {epoch} with R2 = {r2:.4f}")
            else:
                # import time
                # time.sleep(1)
                # torch.save(record_param, f'../model_save/{args.model}/base_obs.pth')
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
        train_years = range(1979, 1999)
        train_dataset = TemporalDataset(train_years, "seq")
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_years = range(1999, 2019)
        test_dataset = TemporalDataset(test_years, "seq")
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        obs_train(args.epoch, train_dataloader, test_dataloader)
    else: # fluxnet 
        test_dataset = FluxnetBaselineDataset(config, mode='test')
        train_dataset = FluxnetBaselineDataset(config, mode='train')
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        obs_train(args.epoch, train_dataloader, test_dataloader)
if __name__ == "__main__":
    main()