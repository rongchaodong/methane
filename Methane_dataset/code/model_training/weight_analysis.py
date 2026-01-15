import torch
from learnable_hierarchical import LearnableHierarchicalWrapper
from config import Config
from dataset import FluxnetHierarchicalDataset, HierarchicalTemporalDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def load_pretrained_model(pretrained_model_path, fluxnet_model):
    pretrained_state = torch.load(pretrained_model_path, map_location='cpu', weights_only=True)
    current_state = fluxnet_model.state_dict()
    
    for key, value in pretrained_state.items():
        if key in current_state:
            if current_state[key].shape == value.shape:
                current_state[key] = value
            else:
                print(f"Shape mismatch for {key}: {current_state[key].shape} vs {value.shape}")
    
    fluxnet_model.load_state_dict(current_state)
    print("Successfully loaded pretrained model with location weight matching capability")

    return fluxnet_model

def main():
    simulated = False
    feature_names = ['clelev', 'clfaotxt', 'cltveg', 'phh2o', 'topsoil_bulk_density', 'vegetation_type', 'wetlandtype', 'climatetype', 'ch4', 'kco', 'PREC', 'SOLR', 'TAIR', 'VAPR', 'NPP']
    config = Config()
    if simulated:
        path = '../model_save/learnable_hierarchical/base_model-2.pth'
        all_years = range(1979, 2019)
        train_years = range(1979 + config.num_context_years, 1999)
        test_years = range(1999, 2019)
        train_dataset = HierarchicalTemporalDataset(train_years, config, all_years)
        test_dataset = HierarchicalTemporalDataset(test_years, config, all_years)
        data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=False)
    else:
        path = "../model_save/learnable_hierarchical/base_obs_no_attn.pth"
        train_dataset = FluxnetHierarchicalDataset(config, mode='train')
        test_dataset = FluxnetHierarchicalDataset(config, mode='test')
        data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = LearnableHierarchicalWrapper(
            input_dim=config.input_size,  # 15 features
            annual_hidden_dim=16,
            daily_hidden_dim=8,
            output_dim=config.output_size,
            inner_loss_weight=1.0,
            enable_contrastive=False,
            enable_attn=False,
            dataset=train_dataset
        ).to(config.device)
    load_pretrained_model(path, model)

    projections = model.model.encoder.output_projections
    
    weights = []
    
    for layer in projections:
        w = layer.weight.data.cpu().numpy().flatten()
        # b = layer.bias.data.cpu().numpy().flatten()
        # w = np.concatenate([w, b]) 
        weights.append(w)
        
    weights_matrix = np.array(weights)

    test_inputs = []
    for batch_x, _, _, _, _ in data_loader:
        batch_x = batch_x.to(config.device)
        if simulated:
            batch_x = torch.mean(batch_x, dim=[0, 1]).squeeze(0)
        else:
            batch_x = torch.mean(batch_x, dim=1).squeeze(0)
        test_inputs.append(batch_x.cpu().numpy())
    test_inputs = np.array(test_inputs)  # (num_samples, 15)
    if not simulated:
        test_inputs = test_inputs[:-1]


    pca = PCA(n_components=3)
    weights_pca = pca.fit_transform(weights_matrix) # (N, 3)

    correlation_matrix = np.zeros((3, 15))

    for i in range(3): 
        for j in range(15): 
            corr = np.corrcoef(weights_pca[:, i], test_inputs[:, j])[0, 1]
            correlation_matrix[i, j] = corr

    plt.figure(figsize=(16, 6))
    ax = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", 
                xticklabels=feature_names, 
                yticklabels=['Weight_1', 'Weight_2', 'Weight_3'],
                cmap='RdBu_r', center=0, annot_kws={"size": 22},
                cbar_kws={'pad': 0.01})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)
    plt.title("Correlation between Learned Weights and Input Features", fontsize=24, fontweight='bold')
    plt.xticks(fontsize=20, rotation=30)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig('weight_feature_correlation.pdf', dpi=300)
    print("Figure saved as weight_feature_correlation.pdf")

if __name__ == "__main__":
    main()