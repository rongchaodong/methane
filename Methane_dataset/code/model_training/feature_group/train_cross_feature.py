import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Dataset
import math
import logging
from sklearn.metrics import mean_squared_error, r2_score
from config import Config
from cross_feature_attention import ImprovedCrossFeatureAttentionModel, NoIntraGroupModel, NoInterGroupModel


config = Config()
criterion = nn.MSELoss()

parser = argparse.ArgumentParser(description="training base model for methane dataset")
parser.add_argument("--type", type=str, default="temporal", help="please choose temporal or spatial")
parser.add_argument("--simulated", action="store_true", help="training simulated data or observed data")
parser.add_argument("--data_type", type=str, default="seq", help="Model selection: random, or seq")
parser.add_argument("--epoch", type=int, default=100, help="Training epoches")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--ablation", type=str, default="full", 
                    help="Ablation study model: 'full' (default), 'no_intra' (without intra-group attention), 'no_inter' (without inter-group attention)")
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
            elif data_source == "sparse":
                self.input_files = [f"../../processed_data/TEM-MDM/sparse/temporal/input_{year}.npy" for year in years]
                self.output_files = [f"../../processed_data/TEM-MDM/sparse/temporal/output_{year}.npy" for year in years]
            elif data_source == "1percent":
                self.input_files = [f"../../processed_data/TEM-MDM/1percent/temporal/input_{year}.npy" for year in years]
                self.output_files = [f"../../processed_data/TEM-MDM/1percent/temporal/output_{year}.npy" for year in years]
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

class SpatialDataset(Dataset):
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
    filename=f"log/{args.data_type}_{args.type}_cross_{args.ablation}.log" if args.simulated else f"log/base_obs_{args.type}_cross_{args.ablation}.log", 
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO 
)

logger = logging.getLogger()

# Model selection based on ablation flag
if args.ablation == "no_intra":
    model = NoIntraGroupModel(config).to(config.device)
    logger.info("Using ablation model: No Intra-Group Attention")
elif args.ablation == "no_inter":
    model = NoInterGroupModel(config).to(config.device)
    logger.info("Using ablation model: No Inter-Group Attention")
else:  # full model
    model = ImprovedCrossFeatureAttentionModel(config).to(config.device)
    logger.info("Using full model: Improved Cross-Feature Attention")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

total_params = sum(p.numel() for p in model.parameters())
print(f"模型的参数量为: {total_params}")
logger.info(f"Model parameters: {total_params}")

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
            _, prediction = model(batch_X)
            # print("prediction shape:", prediction.shape)
            # print("batch_y shape:", batch_y.shape)

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
                torch.save(record_param, f'../model_save/cross/base_{args.ablation}.pth')
                logger.info(f"New best model saved at epoch {epoch} with R2 = {r2:.4f}")
            else:
                # pass
                import time
                time.sleep(1)
                torch.save(record_param, f'../model_save/cross/base_obs_{args.ablation}.pth')
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
            train_dataset = TemporalDataset(train_years, "seq")
            train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            test_years = range(1999, 2019)
            test_dataset = TemporalDataset(test_years, "seq")
            test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            obs_train(args.epoch, train_dataloader, test_dataloader)         
        else: # simulated spatial dataset
            years = range(1979, 2019)
            train_folds = [1, 2, 3, 4]
            test_folds = [0]
            train_dataset = SpatialDataset(years, train_folds)
            test_dataset = SpatialDataset(years, test_folds)
            train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            obs_train(args.epoch, train_dataloader, test_dataloader)
    else: # fluxnet
        if args.type == "temporal":
            test_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_x.npy'), dtype=torch.float32)
            test_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_y.npy'), dtype=torch.float32)
            train_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/train_data_x.npy'), dtype=torch.float32)
            train_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/train_data_y.npy'), dtype=torch.float32)
            mean = torch.tensor(np.load('../../data/TEM-MDM/mean_vals.npy'))
            std = torch.tensor(np.load('../../data/TEM-MDM/std_vals.npy'))
            test_x = (test_x - mean) / std
            train_x = (train_x - mean) / std
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_dataset = Dataset(test_x.to(device), test_y.to(device))
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
            train_dataset = Dataset(train_x.to(device), train_y.to(device))
            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            obs_train(args.epoch, train_dataloader, test_dataloader)
        else:
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_dataset = Dataset(test_x.to(device), test_y.to(device))
            test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
            train_dataset = Dataset(train_x.to(device), train_y.to(device))
            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)
            obs_train(args.epoch, train_dataloader, test_dataloader)

main()