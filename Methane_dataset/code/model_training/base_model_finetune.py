import os, time
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import math
import logging
from sklearn.metrics import mean_squared_error, r2_score
from config import Config
from model import LSTM, Transformer, iTransformer, Pyraformer, TSMixer, TimeMixer, PatchTST, P_sLSTM, DUET
from dataset import FluxnetBaselineDataset


config = Config()
criterion = nn.MSELoss()

parser = argparse.ArgumentParser(description="training base model for methane dataset")
parser.add_argument("--model", type=str, help="Model selection")
parser.add_argument("--data_type", type=str, default="seq", help="Model selection: random, or seq")
parser.add_argument("--epoch", type=int, default=1000, help="Training epoches")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--context", type=int, default=4, help="Length of context year as prior knowledge used in hierarchy model")
args = parser.parse_args()

config.num_context_years = args.context


logging.basicConfig(
    filename=f"log/finetune/{args.data_type}_{args.model}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO 
)

logger = logging.getLogger()

#model selection
if args.model == 'lstm':
    model = LSTM(config).to(config.device)
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

path = f'../model_save/{args.model}/base_model.pth'
model.load_state_dict(torch.load(path, weights_only=True, map_location=config.device))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# total_params = sum(p.numel() for p in model.parameters())
# print(f"Model {args.model} has: {total_params} parameters.")

def obs_train(epoches : int, train_dataloader: DataLoader, test_dataloader: DataLoader):
    #hyperparameter
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
            output_dir = f'../model_save/{args.model}'
            os.makedirs(output_dir, exist_ok=True)
            time.sleep(1)
            torch.save(record_param, f'../model_save/{args.model}/base_obs.pth')
            logger.info(f"New best model saved at epoch {epoch} with R2 = {r2:.4f}")

        else:
            wait += 1
            logger.info(f"No improvement in R2 for {wait} epoch(s)")

        if wait >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    return best_nrmse, best_r2

def main():
    test_dataset = FluxnetBaselineDataset(config, mode='test')
    train_dataset = FluxnetBaselineDataset(config, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    obs_train(args.epoch, train_dataloader, test_dataloader)
if __name__ == "__main__":
    main()