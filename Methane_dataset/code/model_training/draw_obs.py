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
from sklearn.metrics import mean_squared_error, r2_score

config = Config()

parser = argparse.ArgumentParser(description="draw prediction for models")
parser.add_argument("--simulated", action="store_true", help="training simulated data or observed data")
parser.add_argument("--base_model", type=str, help="Model selection")
parser.add_argument("--test_model", type=str, help="Model selection")
parser.add_argument("--data_type", type=str, default="seq", help="Model selection: random, or seq")
parser.add_argument("--context", type=int, default=4, help="Length of context year as prior knowledge used in hierarchy model")
args = parser.parse_args()

class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_data(model1: str, model2: str, year: int = 4):
    if model1 == "lstm":
        base_model = LSTM(config).to(config.device)
    else:
        print("wrong base model, please use LSTM as baseline")
    
    if model2 == "transformer":
        test_model = Transformer(config).to(config.device)
    elif model2 == "hybrid":
        test_model = LSTM_Transformer(config).to(config.device)
    elif model2 == "hierarchy":
        test_model = Hierarchical(10, 8, 5, 8, 1).to(config.device)

    try:
        path = f'../model_save/lstm/base_obs.pth'
        base_model.load_state_dict(torch.load(path, weights_only=True))
        if model2 == "hierarchy":
            path = f'../model_save/{model2}/base_{model2}-{year}.pth'
        elif args.simulated:
            path = f'../model_save/{model2}/base.pth'
        else:
            path = f'../model_save/{model2}/base_obs.pth'
        test_model.load_state_dict(torch.load(path, weights_only=True))
    except FileNotFoundError:
        print(path)
        print("cannot find the model!")
        exit()

    test_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_x.npy'), dtype=torch.float32)
    test_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_y.npy'), dtype=torch.float32)
    mean = torch.tensor(np.load('../../data/TEM-MDM/mean_vals.npy'))
    std = torch.tensor(np.load('../../data/TEM-MDM/std_vals.npy'))
    test_x = (test_x - mean) / std
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = Dataset(test_x.to(device), test_y.to(device))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    base_model.eval()
    base_all_predictions = []
    base_all_targets = []
    test_all_predictions = []
    test_all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            _, base_prediction = base_model(batch_X)
            _, test_prediction = test_model(batch_X)
            # print("shape", prediction.shape, batch_y.shape)
            # print(batch_X[0][0])
            # print(batch_y[0][0])
            base_all_predictions.append(base_prediction.cpu().numpy().reshape(-1))
            base_all_targets.append(batch_y.cpu().numpy().reshape(-1))
            test_all_predictions.append(test_prediction.cpu().numpy().reshape(-1))
            test_all_targets.append(batch_y.cpu().numpy().reshape(-1))

        base_all_predictions = np.array(base_all_predictions)
        base_all_targets = np.array(base_all_targets)
        test_all_predictions = np.array(test_all_predictions)
        test_all_targets = np.array(test_all_targets)
        # print(base_all_predictions.shape, base_all_targets.shape)
        # print(test_all_predictions.shape, test_all_targets.shape)

        base_r2_list = []
        test_r2_list = []
        for i in range(base_all_predictions.shape[0]):
            if np.isnan(base_all_targets[i]).any():
                continue
            r2 = r2_score(base_all_targets[i], base_all_predictions[i])
            base_r2_list.append(r2)

        for i in range(test_all_predictions.shape[0]):
            if np.isnan(test_all_targets[i]).any():
                continue
            r2 = r2_score(test_all_targets[i], test_all_predictions[i])
            test_r2_list.append(r2)
        
        print(base_r2_list)
        print(test_r2_list)
        # print(np.mean(base_r2_list))
        # print(np.mean(test_r2_list))

        base_all_predictions = np.concatenate(base_all_predictions, axis=0)
        base_all_targets = np.concatenate(base_all_targets, axis=0)
        mask = ~np.isnan(base_all_targets)
        r2 = r2_score(base_all_targets[mask], base_all_predictions[mask])
        print(r2)

        test_all_predictions = np.concatenate(test_all_predictions, axis=0)
        test_all_targets = np.concatenate(test_all_targets, axis=0)
        mask = ~np.isnan(test_all_targets)
        r2 = r2_score(test_all_targets[mask], test_all_predictions[mask])
        print(r2)

        return base_r2_list, test_r2_list

def draw_obs(base_r2_list, test_r2_list):
    import numpy as np
    import matplotlib.pyplot as plt

    base_r2_arr = np.array(base_r2_list)
    test_r2_arr = np.array(test_r2_list)


    base_mask = (base_r2_arr > -10) & (base_r2_arr < 1)
    test_mask = (test_r2_arr > -10) & (test_r2_arr < 1)
    inlier_mask = base_mask & test_mask

    # Inliers
    base_inliers = base_r2_arr[inlier_mask]
    test_inliers = test_r2_arr[inlier_mask]
    print(base_inliers.shape)

    # Outliers
    base_outliers = base_r2_arr[~inlier_mask]
    test_outliers = test_r2_arr[~inlier_mask]

    plt.figure(figsize=(8, 8))

    # Scatter inliers
    plt.scatter(test_inliers, base_inliers, color='blue', label='Sites (inliers)')
    # # Scatter outliers in red
    # if len(base_outliers) > 0:
    #     plt.scatter(test_outliers, base_outliers, color='red', label='Outliers')

    # Set axis limits to focus on the main cluster
    plt.xlim(-3, 1)
    plt.ylim(-3, 1)

    # Draw y=x line within the focused region
    min_val = -3
    max_val = 1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    plt.xlabel('LSTM+Transformer R2')
    plt.ylabel('LSTM R2')
    plt.title('LSTM+Transformer vs LSTM R2 for Each Test Site')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'figure/OBS_all_{args.base_model}_{args.test_model}.png')
    plt.close()

def print_index(base_r2_list, test_r2_list):
    sites = ['DE.Hte', 'DE.SfN', 'DE.Zrk', 'FI.Lom', 'FI.Si2', 'FI.Sii', 'RU.Ch2', 'RU.Che', 'US.BZB', 'US.BZF', 'US.ICs', 'US.Ivo', 'US.Los', 'US.MRM', 'US.Myb', 'US.Myb', 'US.ORv']
    base_r2_arr = np.array(base_r2_list)
    test_r2_arr = np.array(test_r2_list)
    # Find indices where test > base
    indices = np.where(test_r2_arr > base_r2_arr)[0]
    print(indices)
    for i, site in enumerate(sites):
        if i not in indices:
            print(f"Site index {i} ({site}) is less than LSTM R2.")
    return indices

base_r2_list, test_r2_list = get_data(args.base_model, args.test_model, args.context)
draw_obs(base_r2_list, test_r2_list)
print_index(base_r2_list, test_r2_list)
# index is  [0  2  3  4  5  6 11 12 14]  [ 0  1  2  3  4  5  7 12 14]