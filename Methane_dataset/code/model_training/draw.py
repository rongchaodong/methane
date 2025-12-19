import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from model import LSTM, Transformer
from hybrid import LSTM_Transformer
from hierarchical import Hierarchical, HierarchicalTemporalDataset
from hierarchical_x_y import EnhancedHierarchicalMultiEmbedding
from hierarchical_x_y import HierarchicalTemporalDataset as hierarchy_temporal_x_y
from cross_feature_attention import ImprovedCrossFeatureAttentionModel, NoInterGroupModel, NoIntraGroupModel
import argparse
import math 

parser = argparse.ArgumentParser(description="draw prediction for models")
parser.add_argument("--simulated", action="store_true", help="training simulated data or observed data")
parser.add_argument("--model", type=str, help="Model selection")
parser.add_argument("--data_type", type=str, default="seq", help="Model selection: random, or seq")
parser.add_argument("--context", type=int, default=4, help="Length of context year as prior knowledge used in hierarchy model")
parser.add_argument("--index", type=int, help="print which site")
args = parser.parse_args()

from config import Config
config = Config()

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
    elif model_str == "mul-emb":
        model = EnhancedHierarchicalMultiEmbedding(11, 8, 5, 8, 1).to(config.device)
    elif model_str == "cross":
        # model = ImprovedCrossFeatureAttentionModel(config).to(config.device)
        model = NoIntraGroupModel(config).to(config.device)
    elif model_str == "learnable_hidden_lstm":
        from learnable_hidden_state_lstm import LearnableHiddenStateLSTMWrapper
        model = LearnableHiddenStateLSTMWrapper(
            input_size=config.input_size,
            hidden_size=8,
            output_size=config.output_size,
            historical_years=2,
            inner_model_type="lstm",
            num_layers=1,
            inner_loss_weight=1.0,
        ).to(config.device)
    elif model_str == "learnable_hidden_fluxnet":
        from learnable_hierarchical import LearnableHierarchicalWrapper, FluxnetHierarchicalDataset
        train_dataset = FluxnetHierarchicalDataset(config, mode='train')
        test_dataset = FluxnetHierarchicalDataset(config, mode='test')
        model = LearnableHierarchicalWrapper(
            input_dim=config.input_size,  # 15 features
            annual_hidden_dim=16,
            daily_hidden_dim=8,
            output_dim=config.output_size,
            inner_loss_weight=1.0,
            enable_contrastive=True,
            dataset=train_dataset
        ).to(config.device)

    try:
        if simulated:
             if model_str == "hierarchy":
                path = f'../model_save/{model_str}/base-{year}.pth'
             elif model_str == "mul-emb":
                path = f'../model_save/{model_str}/base-{year}.pth'
             elif model_str == "learnable_hidden_lstm":
                path = f'../model_save/{model_str}/base_lstm.pth'
             else:
                path = f'../model_save/{model_str}/base_no_intra.pth'
        else:
            if model_str == "hierarchy":
                path = f'../model_save/{model_str}/base_obs-{year}.pth'
            elif model_str == "mul-emb":
                path = f'../model_save/{model_str}/base_obs-{year}.pth'
            elif model_str == "learnable_hidden_lstm":
                path = f'../model_save/{model_str}/base_obs_lstm.pth' 
            elif model_str == "learnable_hidden_fluxnet":
                path = f'../model_save/hierarchical/base_obs_norm.pth'
            elif model_str == "cross":
                path = f'../model_save/{model_str}/base_obs_no_intra.pth'
            else:
                path = f'../model_save/hierarchical/base_lstm.pth'


        model.load_state_dict(torch.load(path, weights_only=True))
    except FileNotFoundError:
        print(path)
        print(f"cannot find {path} the model!")
        exit()

    if simulated:
        test_years = range(1999, 2019)
        if model_str == "hierarchy":
            all_years = range(1979, 2019)
            test_dataset = HierarchicalTemporalDataset(test_years, config, all_years, "seq")
            test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        elif model_str == "mul-emb":
            all_years = range(1979, 2019)
            test_dataset = hierarchy_temporal_x_y(test_years, config, all_years, "seq")
            test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        elif model_str == "learnable_hidden_lstm":
            all_years = range(1979, 2019)
            from train_learnable_hidden_lstm import LearnableHiddenTemporalDataset
            test_dataset = LearnableHiddenTemporalDataset(test_years, all_years, "seq", historical_years=2)
            test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        else:
            test_dataset = TemporalDataset(test_years)
            test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    else:
        # test_x = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_x.npy'), dtype=torch.float32)
        # test_y = torch.tensor(np.load('../../processed_data/FLUXNET-CH4/temporal/test_data_y.npy'), dtype=torch.float32)
        # mean = torch.tensor(np.load('../../data/TEM-MDM/mean_vals.npy'))
        # std = torch.tensor(np.load('../../data/TEM-MDM/std_vals.npy'))
        # test_x = (test_x - mean) / std
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # test_dataset = Dataset(test_x.to(device), test_y.to(device))
        # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        if model_str == "learnable_hidden_fluxnet":
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        else:
            from train_learnable_hierarchical_fluxnet import FluxnetBaselineDataset
            test_dataset = FluxnetBaselineDataset(config, mode='test')
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        if simulated:
            if model_str == "hierarchy" or model_str == "mul-emb":
                for batch_X_daily, batch_x_yearly, batch_y in test_dataloader:
                    if torch.isnan(batch_X_daily).any():
                        continue
                    _, prediction = model(batch_X_daily, batch_x_yearly)
                    break
            elif model_str == "learnable_hidden_lstm":
                for batch_X, historical_X, batch_y, historical_y in test_dataloader:
                    if torch.isnan(batch_X).any():
                        continue
                    prediction, _, _  = model(batch_X, historical_X, historical_y)
                    break
            else:
                for batch_X, batch_y in test_dataloader:
                    _, prediction = model(batch_X)
                    print("shape", prediction.shape, batch_y.shape)
                    break
            all_targets = batch_y.cpu()
            all_predictions = prediction.cpu()
        else:
            # for batch_X, batch_y in test_dataloader:
            #     _, prediction = model(batch_X)
            #     all_predictions.append(prediction.cpu().numpy().reshape(-1))
            #     all_targets.append(batch_y.cpu().numpy().reshape(-1))
            if model_str == "lstm" or model_str == "cross":
                for batch_x, batch_y in test_dataloader:
                    _, predictions = model(batch_x)
                    all_predictions.append(predictions.cpu().numpy().reshape(-1))
                    all_targets.append(batch_y.cpu().numpy().reshape(-1))
            else:
                for batch_x_current, batch_historical_inputs, batch_y, batch_historical_targets, batch_file_idx in test_dataloader:
                    predictions, _, _, _ = model(
                        batch_x_current, 
                        batch_historical_inputs, 
                        batch_file_idx,
                        batch_historical_targets
                    )
                    # predictions = predictions * train_dataset.y_std + train_dataset.y_mean
                    predictions_log = predictions.cpu().numpy() * train_dataset.y_std + train_dataset.y_mean
                    predictions = np.sign(predictions_log) * (np.exp(np.abs(predictions_log)) - 1)
                    all_predictions.append(predictions.reshape(-1))
                    all_targets.append(batch_y.cpu().numpy().reshape(-1))

    if simulated:
        plt.figure(figsize=(12, 4))
        plt.plot(all_targets[index], label='Ground Truth')
        plt.plot(all_predictions[index], label='Prediction')
        plt.xlabel('Day of Year')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if simulated:
            if model_str == "hierarchy" or model_str == "mul-emb":
                plt.savefig(f'figure/prediction_TEM_{model_str}-{year}.png')
            else:
                plt.savefig(f'figure/no_intra_{index}.png')
    else:
        num_indices = len(all_predictions)
        
        print(f"Total number of observed indices: {num_indices}") 

        if num_indices == 0:
            print("No predictions to plot.")
            return

        cols = 3  
        rows = math.ceil(num_indices / cols) 
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4), squeeze=False)
        axes = axes.flatten() 


        for i in range(num_indices):
            site, year = test_dataset.samples[i]
            ax = axes[i] 
            ax.plot(all_targets[i], label='Ground Truth')
            ax.plot(all_predictions[i], label='Prediction')
            ax.set_title(f'Observed Data - {site} ({year})')
            ax.set_xlabel('Day of Year')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        

        for j in range(num_indices, len(axes)):
            axes[j].axis('off')

        plt.tight_layout() 

        plt.savefig(f'figure/no_intra_obs_{index}.png')


def compare_models_draw(simulated: bool, index: int = 0):
    """
    Draw comparison of 5 models: LSTM, Cross (full), No Inter-Group, No Intra-Group, Transformer
    """
    model_configs = [
        # ("lstm", "LSTM", LSTM(config)),
        ("cross_full", "HCFA", ImprovedCrossFeatureAttentionModel(config)),
        ("cross_no_inter", "No Inter-Group", NoInterGroupModel(config)),
        ("cross_no_intra", "No Intra-Group", NoIntraGroupModel(config)),
        ("transformer", "Transformer", Transformer(config)),
    ]
    
    # Prepare datasets
    if simulated:
        test_years = range(1999, 2019)
        test_dataset = TemporalDataset(test_years)
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    else:
        from train_learnable_hierarchical_fluxnet import FluxnetBaselineDataset
        test_dataset = FluxnetBaselineDataset(config, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Collect predictions for all models
    all_model_predictions = []
    ground_truth = None
    
    for model_key, model_name, model in model_configs:
        model = model.to(config.device)
        
        # Load model weights
        try:
            if simulated:
                if model_key == "cross_full":
                    path = f'../model_save/cross/base-2.pth'
                elif model_key == "cross_no_inter":
                    path = f'../model_save/cross/base_no_inter.pth'
                elif model_key == "cross_no_intra":
                    path = f'../model_save/cross/base_no_intra.pth'
                else:
                    path = f'../model_save/{model_key}/base.pth'
            else:
                if model_key == "cross_full":
                    path = f'../model_save/cross/base_obs-2.pth'
                elif model_key == "cross_no_inter":
                    path = f'../model_save/cross/base_obs_no_inter.pth'
                elif model_key == "cross_no_intra":
                    path = f'../model_save/cross/base_obs_no_intra.pth'
                else:
                    path = f'../model_save/{model_key}/base_obs.pth'
            
            model.load_state_dict(torch.load(path, weights_only=True))
            print(f"Loaded model: {model_name} from {path}")
        except FileNotFoundError:
            print(f"Warning: Could not find {path}, skipping {model_name}")
            all_model_predictions.append((model_name, None))
            continue
        
        # Run inference
        model.eval()
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            if simulated:
                for batch_X, batch_y in test_dataloader:
                    _, prediction = model(batch_X)
                    predictions_list.append(prediction.cpu().numpy())
                    targets_list.append(batch_y.cpu().numpy())
                    break  # Only take first batch for visualization
                
                predictions = predictions_list[0]
                targets = targets_list[0]
            else:
                for batch_x, batch_y in test_dataloader:
                    _, prediction = model(batch_x)
                    predictions_list.append(prediction.cpu().numpy().reshape(-1))
                    targets_list.append(batch_y.cpu().numpy().reshape(-1))
                
                predictions = predictions_list[index] if index < len(predictions_list) else predictions_list[0]
                targets = targets_list[index] if index < len(targets_list) else targets_list[0]
        
        all_model_predictions.append((model_name, predictions))
        
        if ground_truth is None:
            ground_truth = targets
    
    # Create comparison plot
    plt.figure(figsize=(14, 6))
    
    # Plot ground truth
    if ground_truth is not None:
        if simulated:
            plt.plot(ground_truth[index], label='Ground Truth', linewidth=2, color='black', linestyle='--')
        else:
            plt.plot(ground_truth, label='Ground Truth', linewidth=2, color='black', linestyle='--')
    
    # Plot predictions from all models
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for i, (model_name, predictions) in enumerate(all_model_predictions):
        if predictions is not None:
            if simulated:
                plt.plot(predictions[index], label=model_name, linewidth=1.5, color=colors[i], alpha=0.8)
            else:
                plt.plot(predictions, label=model_name, linewidth=1.5, color=colors[i], alpha=0.8)
    
    # Add title based on dataset type
    if simulated:
        plt.title(f'Model Comparison - Simulated Data (Index {index})', fontsize=14, fontweight='bold')
    else:
        site, year = test_dataset.samples[index]
        plt.title(f'Model Comparison - Observed Data: {site} ({year})', fontsize=14, fontweight='bold')
    
    plt.xlabel('Day of Year', fontsize=12)
    plt.ylabel('CH4 Emissions', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    if simulated:
        plt.savefig(f'figure/simulated_idx{index}.png', dpi=300)
        print(f"Saved: figure/model_comparison_simulated_idx{index}.png")
    else:
        plt.savefig(f'figure/model_comparison_obs_idx{index}.png', dpi=300)
        print(f"Saved: figure/obs_idx{index}.png")
    
    plt.close()


# Draw comparisons for both simulated and observed data
print("Drawing simulated data comparison (index=0)...")
compare_models_draw(simulated=True, index=0)

print("\nDrawing observed data comparison (index=3)...")
compare_models_draw(simulated=False, index=0)