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
        print("未找到模型文件 'your_model.pth'。请确保文件路径正确，或先训练并保存一个模型。")

    
    test_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_x.npy'), dtype=torch.float32)
    test_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_y.npy'), dtype=torch.float32)
    train_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/train_data_x.npy'), dtype=torch.float32)
    train_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/train_data_y.npy'), dtype=torch.float32)
    mean = torch.tensor(np.load('../../data/TEM-MDM/mean_vals.npy'))
    # this file is missing, I copy from data/
    std = torch.tensor(np.load('../../data/TEM-MDM/std_vals.npy'))
    test_x = (test_x - mean) / std
    test_y = test_y + 100
    train_x = (train_x - mean) / std
    train_y = train_y + 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = Dataset(test_x.to(device), test_y.to(device))
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    train_dataset = Dataset(train_x.to(device), train_y.to(device))
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)

    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in train_dataloader:
            _, prediction = model(batch_X)
            all_predictions.append(prediction.cpu().numpy().reshape(-1))
            all_targets.append(batch_y.cpu().numpy().reshape(-1))

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mask = ~np.isnan(all_targets)
    with torch.no_grad():
        preds = []
        for i in range(len(X_batch)):
            # 每次预测时，重置hidden_cell，以确保每个预测是独立的
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))
            # 输入单个数据点进行预测
            pred_single = model(X_batch[i:i+1])
            preds.append(pred_single)
        pred = torch.cat(preds)

    # 5. 使用您提供的代码进行绘图
    plt.figure(figsize=(12, 4))
    plt.plot(y_batch.cpu().numpy(), label='Ground Truth')
    plt.plot(pred.detach().cpu().numpy(), label='Prediction')
    plt.xlabel('Day of Year')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prediction_vs_ground_truth.png')