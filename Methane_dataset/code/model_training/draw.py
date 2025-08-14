import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from model import LSTM, Transformer
from hybrid import LSTM_Transformer
from hierarchical import Hierarchical, HierarchicalTemporalDataset
from config import Config
import argparse

config = Config()

parser = argparse.ArgumentParser(description="draw prediction for models")
parser.add_argument("--simulated", action="store_true", help="training simulated data or observed data")
parser.add_argument("--model", type=str, help="Model selection")
parser.add_argument("--data_type", type=str, default="seq", help="Model selection: random, or seq")
parser.add_argument("--context", type=int, default=4, help="Length of context year as prior knowledge used in hierarchy model")
parser.add_argument("--index", type=int, help="print which site")
args = parser.parse_args()

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
        if args.data_type == "seq":
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

def yearly_draw(model_str: str, simulated: bool, year: int = 4, index: int = 0):
    if model_str == "lstm":
        model = LSTM(config).to(config.device)
    elif model_str == "transformer":
        model = Transformer(config).to(config.device)
    elif model_str == "hybrid":
        model = LSTM_Transformer(config).to(config.device)
    elif model_str == "hierarchy":
        model = Hierarchical(10, 8, 5, 8, 1).to(config.device)

    try:
        if model_str == "hierarchy":
            path = f'../model_save/{model_str}/base-{year}.pth'
        elif args.simulated:
            path = f'../model_save/{model_str}/base.pth'
        else:
            path = f'../model_save/{model_str}/base_obs.pth'
        model.load_state_dict(torch.load(path, weights_only=True))
    except FileNotFoundError:
        print(path)
        print("cannot find {path} the model!")
        exit()

    if simulated:
        test_years = range(1999, 2019)
        if model_str == "hierarchy":
            all_years = range(1979, 2019)
            test_dataset = HierarchicalTemporalDataset(test_years, config, all_years, "seq")
            test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        else:
            test_dataset = TemporalDataset(test_years)
            test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    else:
        test_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_x.npy'), dtype=torch.float32)
        test_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_y.npy'), dtype=torch.float32)
        mean = torch.tensor(np.load('../../data/TEM-MDM/mean_vals.npy'))
        std = torch.tensor(np.load('../../data/TEM-MDM/std_vals.npy'))
        test_x = (test_x - mean) / std
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_dataset = Dataset(test_x.to(device), test_y.to(device))
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        if simulated:
            if model_str == "hierarchy":
                for batch_X_daily, batch_x_yearly, batch_y in test_dataloader:
                    if torch.isnan(batch_X_daily).any():
                        continue
                    _, prediction = model(batch_X_daily, batch_x_yearly)
                    break
            else:
                for batch_X, batch_y in test_dataloader:
                    _, prediction = model(batch_X)
                    print("shape", prediction.shape, batch_y.shape)
                    break
            all_targets = batch_y.cpu()
            all_predictions = prediction.cpu()
        else:
            for batch_X, batch_y in test_dataloader:
                _, prediction = model(batch_X)
                all_predictions.append(prediction.cpu().numpy().reshape(-1))
                all_targets.append(batch_y.cpu().numpy().reshape(-1))


        plt.figure(figsize=(12, 4))
        plt.plot(all_targets[index], label='Ground Truth')
        plt.plot(all_predictions[index], label='Prediction')
        plt.xlabel('Day of Year')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if simulated:
            if model_str == "hierarchy":
                plt.savefig(f'figure/prediction_TEM_{model_str}-{year}.png')
            else:
                plt.savefig(f'figure/prediction_TEM_{model_str}.png')
        else:
            plt.savefig(f'figure/prediction_OBS_{model_str}-{args.index}.png')


yearly_draw(args.model, args.simulated, args.context, args.index)