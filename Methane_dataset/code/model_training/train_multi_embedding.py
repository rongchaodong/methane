from hierarchical_multi_embedding import HierarchicalMultiEmbedding, HierarchicalTemporalDataset
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Dataset
import math
import logging
from sklearn.metrics import mean_squared_error, r2_score
from config import Config


config = Config()
criterion = nn.MSELoss()

parser = argparse.ArgumentParser(description="training base model for methane dataset")
parser.add_argument("--type", type=str, default="temporal", help="please choose temporal or spatial")
parser.add_argument("--simulated", action="store_true", help="training simulated data or observed data")
parser.add_argument("--data_type", type=str, default="seq", help="Model selection: random, or seq")
parser.add_argument("--epoch", type=int, default=100, help="Training epoches")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--context", type=int, default=4, help="Length of context year as prior knowledge used in hierarchy model")
args = parser.parse_args()

config.num_context_years = args.context

logging.basicConfig(
    filename=f"log/{args.data_type}_{args.type}_multiple_emb.log" if args.simulated else f"log/base_obs_{args.type}_multiple_emb.log", 
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO 
)

logger = logging.getLogger()


model = HierarchicalMultiEmbedding(10, 8, 5, 8, 1).to(config.device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def hierarchy_train(epoches : int, train_dataloader: DataLoader, test_dataloader: DataLoader):
    #hyperparameter
    patience = 10
    min_delta = 1e-4
    wait = 0
    r2_max = -10.0

    for epoch in range(epoches):
        logger.info(f'Epoch {epoch}: Training start.')
        running_loss = 0.0
        model.train()
        for batch_X_daily, batch_x_yearly, batch_y, batch_x_current_yearly in train_dataloader:
            if torch.isnan(batch_X_daily).any():
                continue
            _, prediction = model(batch_X_daily, batch_x_yearly, batch_x_current_yearly)
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
            for batch_X_daily, batch_x_yearly, batch_y, batch_x_current_yearly in test_dataloader:
                _, prediction = model(batch_X_daily, batch_x_yearly, batch_x_current_yearly)
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
                torch.save(record_param, f'../model_save/mul-emb/base-{args.context}.pth')
            else:
                torch.save(record_param, f'../model_save/mul-emb/base_obs-{args.context}.pth')
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
            logger.info(f'training hierarchy with context = {args.context}')
            all_years = range(1979, 2019)
            warm_up = 1979 + config.num_context_years
            train_years = range(warm_up, 1999)
            test_years = range(1999, 2019)
            train_dataset = HierarchicalTemporalDataset(train_years, config, all_years, "1percent")
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            test_dataset = HierarchicalTemporalDataset(test_years, config, all_years, "seq")
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            hierarchy_train(args.epoch, train_dataloader, test_dataloader) 
        else: # simulated spatial dataset
            years = range(1979, 2019)
            train_folds = [1, 2, 3, 4]
            test_folds = [0]
            
            logger.info(f'training hierarchy with context = {args.context}')
            warm_up = 1979 + config.num_context_years
            train_years = range(warm_up, 1999)
            train_dataset = HierarchicalSpatialDataset(years, config, years, train_folds)
            test_dataset = HierarchicalSpatialDataset(years, config, years, test_folds)
            train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            hierarchy_train(args.epoch, train_dataloader, test_dataloader) 
            # nrmse_list = []
            # r2_list = []
            # for test in range(5): 
            #     train_folds = [i for i in range(5) if i not in [test]]
            #     test_folds = [test]
            #     train_dataset = SpatialDataset(years, train_folds)
            #     test_dataset = SpatialDataset(years, test_folds)
            #     train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            #     test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            #     nrmse, r2 = obs_train(args.epoch, train_dataloader, test_dataloader)
            #     nrmse_list.append(nrmse)
            #     r2_list.append(r2)
            # logger.info(f'testing average nrmse {sum(nrmse_list)/len(nrmse_list):.4f}, r2 score {sum(r2_list)/len(r2_list):.4f}')

main()