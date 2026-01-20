import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import LSTM, Transformer, iTransformer, Pyraformer, TSMixer, TimeMixer, PatchTST, P_sLSTM, DUET
from dataset import FluxnetBaselineDataset, TemporalDataset, HierarchicalTemporalDataset, FluxnetHierarchicalDataset
from learnable_hierarchical import LearnableHierarchicalWrapper
from lstm_embedding import LSTM_Embedding
from sklearn.metrics import mean_squared_error, r2_score
from config import Config
config = Config()

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def compare_models_draw_on_ax(simulated: bool, index: int, ax, show_xlabel: bool = True, subplot_label: str = None):
    if simulated:
        test_years = range(1999, 2019)
        test_dataset = TemporalDataset(test_years, config)
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        test_dataset_chamnet = HierarchicalTemporalDataset(test_years, config, range(1979, 2019))
        test_dataloader_chamnet = DataLoader(test_dataset_chamnet, batch_size=128, shuffle=False)
    else:
        test_dataset = FluxnetBaselineDataset(config, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        train_dataset = FluxnetHierarchicalDataset(config, mode='train')
        test_dataset_chamnet = FluxnetHierarchicalDataset(config, mode='test')
        test_dataloader_chamnet = DataLoader(test_dataset_chamnet, batch_size=1, shuffle=False)

    if simulated:
        model_configs = [
            ("lstm", "LSTM", LSTM(config)),
            ("tsmixer", "TSMixer", TSMixer(config)),
            ("chamnet", "CHAM-net", LearnableHierarchicalWrapper(
                input_dim=config.input_size,  
                annual_hidden_dim=16,
                daily_hidden_dim=8,
                output_dim=config.output_size,
                inner_loss_weight=1.0,
                enable_contrastive=False,
                enable_attn=False,
                dataset=test_dataset_chamnet))
        ]
    else:
        model_configs = [
            ("pyraformer", "Pyraformer", Pyraformer(config)),
            ("tsmixer", "TSMixer", TSMixer(config)),
            ("chamnet", "CHAM-net", LearnableHierarchicalWrapper(
                input_dim=config.input_size,  
                annual_hidden_dim=16,
                daily_hidden_dim=8,
                output_dim=config.output_size,
                inner_loss_weight=1.0,
                enable_contrastive=False,
                enable_attn=False,
                dataset=test_dataset_chamnet))
        ]

    all_model_predictions = []
    ground_truth = None
    R2_list = []
    nrmse_list = []

    for model_key, model_name, model in model_configs:
        model = model.to(config.device)
        
        # Load weights
        try:
            if simulated:
                if model_key == "chamnet":
                    path = f'../model_save/learnable_hierarchical/base_model-2.pth'
                else:
                    path = f'../model_save/{model_key}/base.pth'
            else:
                if model_key == "chamnet":
                    path = f'../model_save/learnable_hierarchical/base_obs_no_attn.pth'
                else:
                    path = f'../model_save/{model_key}/base_obs.pth'
            
            # Use weights_only=True if using newer torch, else remove
            model.load_state_dict(torch.load(path, map_location=config.device)) 
            print(f"Loaded model: {model_name} from {path}")
        except FileNotFoundError:
            print(f"Warning: Could not find {path}, skipping {model_name}")
            all_model_predictions.append((model_name, None))
            continue
        
        # Inference logic
        model.eval()
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            if simulated:
                # ... (Simulated inference logic same as original) ...
                if model_key == "chamnet":
                    for batch_data in test_dataloader_chamnet:
                         # Unpack based on your specific dataset return structure
                        batch_x_current = batch_data[0] 
                        # ... Simplified for brevity, assume logic matches original ...
                        # Run model
                        # predictions_list.append(...)
                        break 
                else:
                    for batch_X, batch_y in test_dataloader:
                        _, prediction = model(batch_X)
                        predictions_list.append(prediction.cpu().numpy())
                        targets_list.append(batch_y.cpu().numpy())
                        break
                predictions = predictions_list[0] # Placeholder logic based on original
                targets = targets_list[0]         # Placeholder
            else:
                # Real Data Inference
                if model_key == "chamnet":
                    for batch_x_current, batch_historical_inputs, batch_y, batch_historical_targets, batch_file_idx in test_dataloader_chamnet:
                        predictions, _, _, _ = model(batch_x_current, batch_historical_inputs, batch_file_idx, batch_historical_targets)
                        predictions_log = predictions.cpu().numpy() * train_dataset.y_std + train_dataset.y_mean
                        predictions = np.sign(predictions_log) * (np.exp(np.abs(predictions_log)) - 1)
                        predictions_list.append(predictions.reshape(-1))
                        targets_list.append(batch_y.cpu().numpy().reshape(-1))
                else:
                    for batch_x, batch_y in test_dataloader:
                        _, prediction = model(batch_x)
                        predictions_list.append(prediction.cpu().numpy().reshape(-1))
                        targets_list.append(batch_y.cpu().numpy().reshape(-1))
                
                predictions = predictions_list[index]
                targets = targets_list[index]
                
                # Metrics
                R2 = r2_score(targets, predictions)
                nrmse = np.sqrt(mean_squared_error(targets, predictions)) / (np.max(targets) - np.min(targets))
                R2_list.append(R2)
                nrmse_list.append(nrmse)
        
        all_model_predictions.append((model_name, predictions))
        if ground_truth is None:
            ground_truth = targets

    if not simulated:
        print(f"Index {index} - R2 scores:", R2_list)
        print(f"Index {index} - NRMSE scores:", nrmse_list)

    
    # Plot Ground Truth
    if ground_truth is not None:
        if simulated:
             # Handle shape if needed
             ax.plot(ground_truth[index], label='Ground Truth', linewidth=3, color='blue', linestyle='solid')
        else:
             ax.plot(ground_truth, label='Ground Truth', linewidth=3, color='blue', linestyle='solid')

    # Plot Models
    colors = ['green', 'purple', 'red'] # Colors for Pyraformer, TSMixer, CHAM-net
    
    for i, (model_name, predictions) in enumerate(all_model_predictions):
        if predictions is not None:
            color = colors[i] if i < len(colors) else 'black'
            
            if model_name == "CHAM-net":
                ax.plot(predictions, label=model_name, linewidth=3, color=color, alpha=0.8)
            else:
                ax.plot(predictions, label=model_name, linewidth=2, color=color, alpha=0.8)

    # Formatting
    ax.set_ylabel('CH4 Emission (mgCH4/m2/day)', fontsize=24, fontweight='bold')

    if show_xlabel:
        ax.set_xlabel('(b) DE.SfN (2014)', fontsize=26, fontweight='bold')
    else:
        ax.set_xlabel('(a) FI.lom (2010)', fontsize=26, fontweight='bold')

    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.legend(loc='upper right', fontsize=18, frameon=True) 
    ax.grid(True, alpha=0.3)
    
    # if subplot_label:
    #     ax.text(-0.08, 1.02, subplot_label, transform=ax.transAxes, 
    #             fontsize=30, fontweight='bold', va='bottom', ha='right')


def run_combined_plot():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12)) 

    print("Processing Plot 1 (Index 2)...")
    compare_models_draw_on_ax(simulated=False, index=4, ax=ax1, 
                              show_xlabel=False, subplot_label='(a)')

    compare_models_draw_on_ax(simulated=False, index=2, ax=ax2, 
                              show_xlabel=True, subplot_label='(b)')

    plt.tight_layout()

    save_path = 'figure/FLUXNET_Combined_2_4.pdf'
    plt.savefig(save_path, dpi=300)
    print(f"Saved combined figure: {save_path}")
    plt.close()

if __name__ == "__main__":
    run_combined_plot()