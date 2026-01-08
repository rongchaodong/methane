import torch
from learnable_hierarchical import LearnableHierarchicalWrapper
from config import Config
from dataset import FluxnetHierarchicalDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def load_pretrained_model(pretrained_model_path, fluxnet_model):
    pretrained_state = torch.load(pretrained_model_path, map_location='cpu', weights_only=True)
    current_state = fluxnet_model.state_dict()
    
    for key, value in pretrained_state.items():
        if key in current_state:
            print(f"Loading weight: {key}")
            if current_state[key].shape == value.shape:
                print(value.shape)
                current_state[key] = value
                print(f"Loaded: {key}")
            else:
                print(f"Shape mismatch for {key}: {current_state[key].shape} vs {value.shape}")
    
    fluxnet_model.load_state_dict(current_state)
    print("Successfully loaded pretrained model with location weight matching capability")

    return fluxnet_model


def main():
    feature_names = ['clelev', 'clfaotxt', 'cltveg', 'phh2o', 'topsoil_bulk_density', 'vegetation_type', 'wetlandtype', 'climatetype', 'ch4', 'kco', 'PREC', 'SOLR', 'TAIR', 'VAPR', 'NPP']
    path = "../model_save/learnable_hierarchical/base_obs_norm_le.pth"
    config = Config()
    train_dataset = FluxnetHierarchicalDataset(config, mode='train')
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

    # self.output_projection = nn.Linear(32, 1)
    print(len(model.model.encoder.output_projections))
    projections = model.model.encoder.output_projections
    
    weights = []
    
    for layer in projections:
        w = layer.weight.data.cpu().numpy().flatten()
        # b = layer.bias.data.cpu().numpy().flatten()
        # w = np.concatenate([w, b]) 
        
        weights.append(w)
        
    weights_matrix = np.array(weights)
    print(f"Weight Matrix Shape: {weights_matrix.shape}")

    test_dataset = FluxnetHierarchicalDataset(config, mode='test')
    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_inputs = []
    for batch_x, _, _, _, _ in data_loader:
        batch_x = batch_x.to(config.device)
        batch_x = torch.mean(batch_x, dim=1).squeeze(0)
        test_inputs.append(batch_x.cpu().numpy())
    test_inputs = np.array(test_inputs)  # (num_samples, 15)
    # print(f"Test Inputs Shape: {test_inputs.shape}")
    # print(weights_matrix.shape)
    # exit()
    tsne = TSNE(n_components=2, perplexity=5, max_iter=1000, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(weights_matrix)

    n_cols = 5
    n_rows = int(np.ceil(15 / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 4 * n_rows))
    axes = axes.flatten()
    test_inputs = test_inputs[:-1]

    for i in range(15):
        ax = axes[i]
        
        feature_vals = test_inputs[:, i]
        
        # 绘制散点图
        # c=feature_vals 会根据数值大小自动映射颜色
        # cmap='Spectral_r' 或 'viridis' 是科研常用的 colormap
        scatter = ax.scatter(
            tsne_results[:, 0], 
            tsne_results[:, 1], 
            c=feature_vals, 
            cmap='Spectral_r', 
            s=10, 
            alpha=0.6,
            edgecolors='none'
        )
        
        ax.set_title(f"Colored by: {feature_names[i]}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    for j in range(15, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('tsne_feature_analysis.png', dpi=300, bbox_inches='tight')
    print("Done! Saved to tsne_feature_analysis.png")

if __name__ == "__main__":
    main()