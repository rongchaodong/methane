from hierarchical import FluxnetDataset, Hierarchical
from config import Config
import torch
import torch.optim as optim
import numpy as np
import logging
import math
from sklearn.metrics import mean_squared_error, r2_score

config = Config()
criterion = torch.nn.MSELoss()

logging.basicConfig(
    filename=f"log/old_hie_fluxnet.log", 
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO 
)
logger = logging.getLogger()

# 创建数据集
train_dataset = FluxnetDataset(config, mode='train')
test_dataset = FluxnetDataset(config, mode='test')

# 创建模型 (annual_input_dim=10, daily_input_dim=5)
model = Hierarchical(
    annual_input_dim=10,  # K_slow
    annual_hidden_dim=8,
    daily_input_dim=5,    # K_fast
    daily_hidden_dim=8,
    output_dim=1
).to(config.device)

# 数据加载
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def hierarchy_train(epoches : int, train_dataloader: DataLoader, test_dataloader: DataLoader):
    #hyperparameter
    patience = 100
    min_delta = 1e-5
    wait = 0
    r2_max = -10.0

    for epoch in range(epoches):
        logger.info(f'Epoch {epoch}: Training start.')
        running_loss = 0.0
        model.train()
        for batch_X_daily, batch_x_yearly, batch_y in train_dataloader:
            if torch.isnan(batch_X_daily).any():
                continue
            _, prediction = model(batch_X_daily, batch_x_yearly)
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
            for batch_X_daily, batch_x_yearly, batch_y in test_dataloader:
                _, prediction = model(batch_X_daily, batch_x_yearly)
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

            torch.save(record_param, f'../model_save/hierarchy/base_obs.pth')
            logger.info(f"New best model saved at epoch {epoch} with R2 = {r2:.4f}")

        else:
            wait += 1
            logger.info(f"No improvement in R2 for {wait} epoch(s)")

        if wait >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    return best_nrmse, best_r2

best_nrmse, best_r2 = hierarchy_train(1000, train_loader, test_loader)
logger.info(f"Training completed. Best NRMSE: {best_nrmse:.4f}, Best R2: {best_r2:.4f}")