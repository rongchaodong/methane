import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import LSTM, Transformer, iTransformer, Pyraformer, TSMixer, TimeMixer, PatchTST, P_sLSTM, DUET
from dataset import FluxnetBaselineDataset, TemporalDataset, HierarchicalTemporalDataset, FluxnetHierarchicalDataset
from learnable_hierarchical import LearnableHierarchicalWrapper
from lstm_embedding import LSTM_Embedding
# from hierarchical_x_y import EnhancedHierarchicalMultiEmbedding
# from hierarchical_x_y import HierarchicalTemporalDataset as hierarchy_temporal_x_y
# from cross_feature_attention import ImprovedCrossFeatureAttentionModel, NoInterGroupModel, NoIntraGroupModel
import argparse
import math 
from sklearn.metrics import mean_squared_error, r2_score

parser = argparse.ArgumentParser(description="draw prediction for models")
parser.add_argument("--simulated", action="store_true", help="training simulated data or observed data")
parser.add_argument("--model", type=str, help="Model selection")
parser.add_argument("--index", type=int, help="print which site")
args = parser.parse_args()

from config import Config
config = Config()

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def draw_historical_sites(site_name):
    dataset = FluxnetBaselineDataset(config, mode='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_targets = {}
    for i, (_, batch_y) in enumerate(dataloader):
        site, year = dataset.samples[i]
        if site == site_name:
            all_targets[year] = batch_y.cpu().numpy().reshape(-1)
    all_colors = [
        'green',         # 1. - LSTM
        'orange',        # 2. - Ps-LSTM
        'purple',        # 3. - transformer
        'gray',         # 4. - iTransformer
        'pink',          # 5. - Pyraformer
        'gray',          # 6. - TSMixer
        'olive',         # 7. - TimeMixer
        'cyan',          # 8. - PatchTST
        'gold',          # 9. - DUET
        'red'            # 10. - CHAM-net
        ]
    # print(len(all_targets))
    # exit()
    assert len(all_targets) < len(all_colors), "Not enough colors for the number of historical sites."
    plt.figure(figsize=(12, 6))
    for i, (year, predictions) in enumerate(all_targets.items()):
        if predictions is not None:
            plt.plot(predictions, label=f'{site_name} ({year})', linewidth=2, color=all_colors[i], alpha=0.8)
    
    # plt.title(f'{site_name} Methane Emission from FLUXNET dataset', fontsize=18, fontweight='bold')
    plt.xlabel('Day of Year', fontsize=26, fontweight='bold')
    plt.ylabel('Methane Emission', fontsize=26, fontweight='bold')
    plt.legend(fontsize=22)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'figure/historical_{site_name}.pdf')


def draw_pattern_scale(site_names: list, target_year: int):
    dataset = FluxnetBaselineDataset(config, mode='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_targets = {}
    for i, (_, batch_y) in enumerate(dataloader):
        site, year = dataset.samples[i]
        if year == target_year and site in site_names:
            all_targets[site] = batch_y.cpu().numpy().reshape(-1)
    all_colors = [
        'green',         # 1. - LSTM
        'orange',        # 2. - Ps-LSTM
        'purple',        # 3. - transformer
        'gray',         # 4. - iTransformer
        'pink',          # 5. - Pyraformer
        'gray',          # 6. - TSMixer
        'olive',         # 7. - TimeMixer
        'cyan',          # 8. - PatchTST
        'gold',          # 9. - DUET
        'red'            # 10. - CHAM-net
        ]
    # print(len(all_targets))
    # exit()
    assert len(all_targets) < len(all_colors), "Not enough colors for the number of historical sites."
    plt.figure(figsize=(12, 6))
    for i, (site, predictions) in enumerate(all_targets.items()):
        if predictions is not None:
            plt.plot(predictions, label=f'{site} ({target_year})', linewidth=2, color=all_colors[i], alpha=0.8)
    
    # plt.title(f'{site_name} Methane Emission from FLUXNET dataset', fontsize=18, fontweight='bold')
    plt.xlabel('Day of Year', fontsize=26, fontweight='bold')
    plt.ylabel('Methane Emission', fontsize=26, fontweight='bold')
    plt.legend(fontsize=22)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'figure/pattern_scale.pdf')

def compare_models_draw(simulated: bool, index: int = 0):
    # Prepare datasets
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

    # best 5 models
    if simulated:
        model_configs = [
            ("lstm", "LSTM", LSTM(config)),
            # ("transformer", "Transformer", Transformer(config)),
            # ("pyraformer", "Pyraformer", Pyraformer(config)),
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
            # ("lstm", "LSTM", LSTM(config)),
            # ("pslstm", "P-sLSTM", P_sLSTM(config)),
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
    all_model_configs = [
        ("lstm", "LSTM", LSTM(config)),
        ("pslstm", "P-sLSTM", P_sLSTM(config)),
        ("transformer", "Transformer", Transformer(config)),
        ("itransformer", "iTransformer", iTransformer(config)),
        ("pyraformer", "Pyraformer", Pyraformer(config)),
        ("tsmixer", "TSMixer", TSMixer(config)),
        ("timemixer", "TimeMixer", TimeMixer(config)),
        ("patchtst", "PatchTST", PatchTST(config)),
        ("duet", "DUET", DUET(config)),
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

    
    # Collect predictions for all models
    all_model_predictions = []
    ground_truth = None
    
    R2_list = []
    nrmse_list = []
    for model_key, model_name, model in model_configs:
        model = model.to(config.device)
        
        # Load model weights
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
                if model_key == "chamnet":
                    for batch_x_current, batch_historical_inputs, batch_y, batch_historical_targets, batch_file_idx in test_dataloader_chamnet:
                        if torch.isnan(batch_x_current).any():
                            continue
                        predictions, _, _, _ = model(
                        batch_x_current, 
                        batch_historical_inputs, 
                        batch_file_idx,
                        batch_historical_targets
                        )
                        predictions_list.append(predictions.cpu().numpy())
                        targets_list.append(batch_y.cpu().numpy())
                        break  # Only take first batch for visualization
                else:
                    for batch_X, batch_y in test_dataloader:
                        _, prediction = model(batch_X)
                        predictions_list.append(prediction.cpu().numpy())
                        targets_list.append(batch_y.cpu().numpy())
                        break  # Only take first batch for visualization

                predictions = predictions_list[0]
                targets = targets_list[0]
            else:
                if model_key == "chamnet":
                    for batch_x_current, batch_historical_inputs, batch_y, batch_historical_targets, batch_file_idx in test_dataloader_chamnet:
                        predictions, _, _, _ = model(
                            batch_x_current, 
                            batch_historical_inputs, 
                            batch_file_idx,
                            batch_historical_targets
                        )
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
                R2 = r2_score(targets, predictions)
                nrmse = np.sqrt(mean_squared_error(targets, predictions)) / (np.max(targets) - np.min(targets))
                R2_list.append(R2)
                nrmse_list.append(nrmse)
        
        all_model_predictions.append((model_name, predictions))
        
        if ground_truth is None:
            ground_truth = targets
    
    if not simulated:
        print("R2 scores for models:", R2_list)
        print("NRMSE scores for models:", nrmse_list)
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    
    # Plot ground truth
    if ground_truth is not None:
        if simulated:
            plt.plot(ground_truth[index], label='Ground Truth', linewidth=3, color='blue', linestyle='solid')
        else:
            plt.plot(ground_truth, label='Ground Truth', linewidth=3, color='blue', linestyle='solid')
    
    # Plot predictions from all models
    if simulated:
        colors = ['green', 'purple', 'red']
    else:
        colors = ['green', 'purple', 'red']
    all_colors = [
        'green',         # 1. - LSTM
        'orange',        # 2. - Ps-LSTM
        'purple',        # 3. - transformer
        'brown',         # 4. - iTransformer
        'pink',          # 5. - Pyraformer
        'gray',          # 6. - TSMixer
        'olive',         # 7. - TimeMixer
        'cyan',          # 8. - PatchTST
        'gold',          # 9. - DUET
        'red'            # 10. - CHAM-net
        ]
    for i, (model_name, predictions) in enumerate(all_model_predictions):
        if predictions is not None:
            if simulated:
                if model_name == "CHAM-net":
                    plt.plot(predictions[index], label=model_name, linewidth=3, color=colors[i], alpha=0.8)
                else:
                    plt.plot(predictions[index], label=model_name, linewidth=2, color=colors[i], alpha=0.8)
            else:
                if model_name == "CHAM-net":
                    plt.plot(predictions, label=model_name, linewidth=3, color=colors[i], alpha=0.8)
                else:
                    plt.plot(predictions, label=model_name, linewidth=2, color=colors[i], alpha=0.8)
    
    # Add title based on dataset type
    if simulated:
        plt.title(f'TEM dataset', fontsize=18, fontweight='bold')
    else:
        site, year = test_dataset.samples[index]
        plt.title(f'FLUXNET dataset: {site} ({year})', fontsize=18, fontweight='bold')
    
    plt.xlabel('Day of Year', fontsize=18, fontweight='bold')
    plt.ylabel('Methane Emission', fontsize=18, fontweight='bold')
    plt.legend(loc='best', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    if simulated:
        plt.savefig(f'figure/TEM_{index}.pdf', dpi=300)
        print(f"Saved: figure/TEM_{index}.pdf")
    else:
        plt.savefig(f'figure/FLUXNET_{index}.pdf', dpi=300)
        print(f"Saved: figure/FLUXNET_{index}.pdf")
    
    plt.close()


draw_pattern_scale(["DE.Hte", "US.Myb", "US.ORv"], 2013)
draw_historical_sites("FI.Lom")

# draw_historical_sites("DE.SfN")
# draw_historical_sites("US.Los")
# Draw comparisons for both simulated and observed data
# print("Drawing simulated data comparison (index=0)...")
# for i in range(19):
#     compare_models_draw(simulated=True, index=i)
# compare_models_draw(simulated=True, index=14)

# best 14

# print("\nDrawing observed data comparison (index=3)...")
# for i in range(19):
#     compare_models_draw(simulated=False, index=i)
# compare_models_draw(simulated=False, index=2)
# compare_models_draw(simulated=False, index=4)