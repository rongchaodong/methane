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
    
parser = argparse.ArgumentParser(description="Pretraining script for methane dataset")
parser.add_argument("--valid_type", type=str, help="Use temporal dataset or spatial dataset")
parser.add_argument("--spatial_fold", type=int, default=0, help="The fold using as cross validation in spatial experiment")
parser.add_argument("--model", type=str, help="Model selection")
parser.add_argument("--load_pretrain", action="store_true", help="If load pretrain model")
parser.add_argument("--id", type=str, help="Experiment identifier")
parser.add_argument("--epoch", type=int, default=200, help="Training epoches")
parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
args = parser.parse_args()
    
config = Config()
device = torch.device("cuda")
if args.valid_type == 'temporal':
    train_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/train_data_x.npy'), dtype=torch.float32)
    train_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/train_data_y.npy'), dtype=torch.float32) * 1.38
    test_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_x.npy'), dtype=torch.float32)
    test_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_y.npy'), dtype=torch.float32) * 1.38
else:
    train_x_list, train_y_list = [], []
    for fold in range(5):
        if fold == args.spatial_fold:
            test_x = torch.tensor(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_x_{fold}.npy"), dtype=torch.float32)
            test_y = torch.tensor(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_y_{fold}.npy"), dtype=torch.float32) * 1.38
        else:
            train_x_list.append(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_x_{fold}.npy"))
            train_y_list.append(np.load(f"../../processed_data/FLUXNET-CH4/spatial/data_y_{fold}.npy"))

    train_x = torch.tensor(np.concatenate(train_x_list, axis=0), dtype=torch.float32)
    train_y = torch.tensor(np.concatenate(train_y_list, axis=0), dtype=torch.float32) * 1.38

indices_to_keep = [i for i in range(test_x.shape[2]) if i not in [4,15]]
train_x = train_x[:, :, indices_to_keep]
test_x = test_x[:, :, indices_to_keep]
mean = torch.tensor(np.load('../../data/TEM-MDM/mean_vals.npy'))
std = torch.tensor(np.load('../../processed_data/TEM-MDM/std_vals.npy'))
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std
train_y = train_y + 100
test_y = test_y + 100

if args.model == 'lstm':
    model = LSTM(config).to(device)
    res_model = LSTM(config).to(device)
elif args.model == 'ealstm':
    model = MyEALSTM(config).to(device)
    res_model = MyEALSTM(config).to(device)
elif args.model == 'tcn':
    model = MultiTCN(config).to(device)
    res_model = MultiTCN(config).to(device)
elif args.model == 'transformer':
    model = Transformer(config).to(device)
    res_model = Transformer(config).to(device)
elif args.model == 'itransformer':
    model = iTransformer(config).to(config.device)
    res_model = iTransformer(config).to(config.device)
elif args.model == 'pyraformer':
    model = Pyraformer(config).to(config.device)
    res_model = Pyraformer(config).to(config.device)
optimizer = torch.optim.Adam(res_model.parameters(), lr=args.lr)
criterion = nn.MSELoss()

model.load_state_dict(torch.load(f'../model_save/{args.model}/base_model.pth', weights_only=True))
if args.valid_type == 'temporal':
    logging.basicConfig(
        filename=f"log/residual_{args.model}_{args.valid_type}_{args.id}.log", 
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO 
    )
else:
    logging.basicConfig(
        filename=f"log/residual_{args.model}_{args.valid_type}_fold{args.spatial_fold}.log", 
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO 
    )

train_x, train_y = train_x.to(device), train_y.to(device)
model.eval()
train_y = (train_y - model(train_x)).detach()

train_dataset = Dataset(train_x, train_y)
test_dataset = Dataset(test_x.to(device), test_y.to(device))
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

logger = logging.getLogger()

logger.info("Command-line Arguments:")
logger.info(json.dumps(vars(args), indent=4)) 

# Log the model configuration
logger.info("Model Configuration:")
logger.info(json.dumps(vars(config), indent=4))

r2_max = -10.0
train_loss_list, test_loss_list, nrmse_list, r2_list = [], [], [], []
for epoch in range(args.epoch):
    running_loss = 0.0
    res_model.train()
    for batch_X, batch_y in train_dataloader:
        _, prediction = res_model(batch_X)
        loss = criterion(batch_y, prediction.view_as(batch_y))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(res_model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
    avg_loss = math.sqrt(running_loss / len(train_dataloader))
    
    res_model.eval()  
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            prediction = res_model(batch_X)[1] + model(batch_X).detach()[1]
            all_predictions.append(prediction.cpu().numpy().reshape(-1))
            all_targets.append(batch_y.cpu().numpy().reshape(-1))

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mask = ~np.isnan(all_targets)

    rmse = np.sqrt(mean_squared_error(all_targets[mask], all_predictions[mask]))
    nrmse = rmse/np.mean(all_targets)
    r2 = r2_score(all_targets, all_predictions)
    if r2 > r2_max:
        r2_max = r2
        record_param = model.state_dict()

    logger.info(f'Epoch {epoch}: Training loss {avg_loss}, testing rmse {rmse}, nrmse {nrmse} r2 score {r2}')
    train_loss_list.append(avg_loss)
    test_loss_list.append(rmse)
    nrmse_list.append(nrmse)
    r2_list.append(r2)
logger.info(f'Training loss {np.min(train_loss_list)}, testing rmse {np.min(test_loss_list)}, nrmse {np.min(nrmse_list)} r2 score {np.max(r2_list)}')
if args.load_pretrain:
    if args.valid_type == 'temporal':
        torch.save(record_param, f'../model_save/{args.model}/residual_{args.valid_type}_{args.id}.pth')
    else:
        torch.save(record_param, f'../model_save/{args.model}/residual_{args.valid_type}_fold{args.spatial_fold}.pth')
else:
    if args.valid_type == 'temporal':
        torch.save(record_param, f'../model_save/{args.model}/scratch_{args.valid_type}_{args.id}.pth')
    else:
        torch.save(record_param, f'../model_save/{args.model}/scratch_{args.valid_type}_fold{args.spatial_fold}.pth')