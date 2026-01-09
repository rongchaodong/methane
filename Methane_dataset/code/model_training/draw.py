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
                path = f'../model_save/lstm/base.pth'


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
            test_dataset = FluxnetBaselineDataset(config, mode='train')
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

        plt.savefig(f'figure/obs_training_all.png')

def draw_historical_sites(site_name):
    dataset = FluxnetBaselineDataset(config, mode='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_targets = {}
    for i, (_, batch_y) in enumerate(dataloader):
        site, year = dataset.samples[i]
        if site == site_name:
            all_targets[year] = batch_y.cpu().numpy().reshape(-1)
    all_colors = [
        'green',         # 1. 绿色 - LSTM
        'orange',        # 2. 橙色 - Ps-LSTM
        'purple',        # 3. 紫色 - transformer
        'brown',         # 4. 棕色 - iTransformer
        'pink',          # 5. 粉色 - Pyraformer
        'gray',          # 6. 灰色 - TSMixer
        'olive',         # 7. 橄榄色 - TimeMixer
        'cyan',          # 8. 青色 - PatchTST
        'gold',          # 9. 金色 - DUET
        'red'            # 10. 红色 - CHAM-net
        ]
    # print(len(all_targets))
    # exit()
    assert len(all_targets) < len(all_colors), "Not enough colors for the number of historical sites."
    plt.figure(figsize=(12, 6))
    for i, (year, predictions) in enumerate(all_targets.items()):
        if predictions is not None:
            plt.plot(predictions, label=f'{site_name}_{year}', linewidth=2, color=all_colors[i], alpha=0.8)
    
    plt.title(f'Historical Data for Site: {site_name}')
    plt.xlabel('Day of Year')
    plt.ylabel('Methane Flux')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'figure/historical_{site_name}.png')

    
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
    plt.figure(figsize=(14, 6))
    
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
        'green',         # 1. 绿色 - LSTM
        'orange',        # 2. 橙色 - Ps-LSTM
        'purple',        # 3. 紫色 - transformer
        'brown',         # 4. 棕色 - iTransformer
        'pink',          # 5. 粉色 - Pyraformer
        'gray',          # 6. 灰色 - TSMixer
        'olive',         # 7. 橄榄色 - TimeMixer
        'cyan',          # 8. 青色 - PatchTST
        'gold',          # 9. 金色 - DUET
        'red'            # 10. 红色 - CHAM-net
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
        plt.title(f'TEM dataset', fontsize=14, fontweight='bold')
    else:
        site, year = test_dataset.samples[index]
        plt.title(f'FLUXNET dataset: {site} ({year})', fontsize=14, fontweight='bold')
    
    plt.xlabel('Day of Year', fontsize=12)
    plt.ylabel('CH4 Emissions', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    if simulated:
        plt.savefig(f'figure/TEM_{index}.png', dpi=300)
        print(f"Saved: figure/TEM_{index}.png")
    else:
        plt.savefig(f'figure/FLUXNET_{index}.png', dpi=300)
        print(f"Saved: figure/FLUXNET_{index}.png")
    
    plt.close()

draw_historical_sites("DE.SfN")
draw_historical_sites("FI.Lom")
draw_historical_sites("US.Los")

# Draw comparisons for both simulated and observed data
# print("Drawing simulated data comparison (index=0)...")
# for i in range(19):
#     compare_models_draw(simulated=True, index=i)
# compare_models_draw(simulated=True, index=19)

# best 14

# print("\nDrawing observed data comparison (index=3)...")
# for i in range(19):
#     compare_models_draw(simulated=False, index=i)
# compare_models_draw(simulated=False, index=18)

# best index: 0 1 2 3 4 5 6 8 10 12 14 15 
# choose 14 for now, seems good [0.23908430337905884, -3.834561347961426, 0.4589861035346985, 0.44660162925720215, 0.5736039876937866] NRMSE: [np.float64(0.13204366874958218), np.float64(0.3328341164403951), np.float64(0.11134058880052077), np.float64(0.11260774315247449), np.float64(0.09884526203298415)]
# choose 4, and 2, pattern and scale


# yearly_draw("lstm", False, 4, 0)