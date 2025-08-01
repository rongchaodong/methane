import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from model import LSTM, Transformer
from hybrid import LSTM_Transformer
from hierarchical import Hierarchical

def yearly_draw(model_str: str, simulated: bool, year: int):
    if model_str == "lstm":
        model = LSTM()

    try:
        model = model.load_state_dict(torch.load('../model_save/{model_str}/base_model.pth'))
    except FileNotFoundError:
        print("cannot find the model!")

    
    test_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_x.npy'), dtype=torch.float32)
    test_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_y.npy'), dtype=torch.float32)
    mean = torch.tensor(np.load('../../data/TEM-MDM/mean_vals.npy'))
    std = torch.tensor(np.load('../../data/TEM-MDM/std_vals.npy'))
    test_x = (test_x - mean) / std
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = Dataset(test_x.to(device), test_y.to(device))
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            _, prediction = model(batch_X)
            # all_predictions.append(prediction.cpu().numpy().reshape(-1))
            # all_targets.append(batch_y.cpu().numpy().reshape(-1))

        # all_predictions = np.concatenate(all_predictions, axis=0)
        # all_targets = np.concatenate(all_targets, axis=0)
        # mask = ~np.isnan(all_targets)


            plt.figure(figsize=(12, 4))
            plt.plot(batch_y.cpu().numpy(), label='Ground Truth')
            plt.plot(prediction.detach().cpu().numpy(), label='Prediction')
            plt.xlabel('Day of Year')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('prediction_vs_ground_truth.png')