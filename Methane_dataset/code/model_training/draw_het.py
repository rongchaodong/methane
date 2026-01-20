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

from config import Config
config = Config()

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def draw_combined_figure(pattern_site_names, pattern_target_year, historical_site_name):

    pattern_data = {}    
    historical_data = {}  
    
    dataset = FluxnetBaselineDataset(config, mode='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, (_, batch_y) in enumerate(dataloader):
        site, year = dataset.samples[i]
        data_value = batch_y.cpu().numpy().reshape(-1)
        
        if year == pattern_target_year and site in pattern_site_names:
            pattern_data[site] = data_value
            
        if site == historical_site_name:
            historical_data[year] = data_value

    all_colors = [
        'green',         # 1. - LSTM
        'orange',        # 2. - Ps-LSTM
        'purple',        # 3. - transformer
        'gray',          # 4. - iTransformer
        'pink',          # 5. - Pyraformer
        'gray',          # 6. - TSMixer
        'olive',         # 7. - TimeMixer
        'cyan',          # 8. - PatchTST
        'gold',          # 9. - DUET
        'red'            # 10. - CHAM-net
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 12)) 
    
    ax_a = axes[0]
    
    assert len(pattern_data) < len(all_colors), "Not enough colors for the number of sites in Pattern Scale."
    
    for i, (site, predictions) in enumerate(pattern_data.items()):
        if predictions is not None:
            ax_a.plot(predictions, label=f'{site} ({pattern_target_year})', linewidth=2, color=all_colors[i], alpha=0.8)
    
    ax_a.set_xlabel('(a) Spatial heterogeneity in DE.Hte, US.Myb, and US.ORv.', fontsize=26, fontweight='bold')
    ax_a.set_ylabel('CH4 Emission (mgCH4/m2/day)', fontsize=24, fontweight='bold')
    ax_a.legend(fontsize=22)
    ax_a.tick_params(axis='both', which='major', labelsize=24)
    ax_a.grid(True)

    ax_b = axes[1]
    
    assert len(historical_data) < len(all_colors), "Not enough colors for the number of years in Historical Sites."
    
    for i, (year, predictions) in enumerate(historical_data.items()):
        if predictions is not None:
            ax_b.plot(predictions, label=f'{historical_site_name} ({year})', linewidth=2, color=all_colors[i], alpha=0.8)
    
    ax_b.set_xlabel('(b) Temporal evolutionary trends in FI.Lom.', fontsize=26, fontweight='bold')
    ax_b.set_ylabel('CH4 Emission (mgCH4/m2/day)', fontsize=24, fontweight='bold')
    ax_b.legend(fontsize=22)
    ax_b.tick_params(axis='both', which='major', labelsize=24)
    ax_b.grid(True)

    plt.tight_layout()
    plt.savefig(f'figure/challendge.pdf')
    # plt.show()

draw_combined_figure(
    pattern_site_names=["DE.Hte", "US.Myb", "US.ORv"], 
    pattern_target_year=2013, 
    historical_site_name="FI.Lom"
)