import torch
import torch.nn as nn
from config import Config
import torch.nn.functional as F

config = Config()

# find the exact location in the TEM model
training_index = [595, 345, 345, 345, 350, 350, 350, 350, 350, 350, 350, 30, 30, 351, 351, 351, 351, 351, 29, 29, 29, 29, 532, 532, 532, 532, -1, -1, -1, -1, -1, 266, 266, -1, -1, 42, 42, 42, 27, 27, -1, -1, 369, 369, 369, 369, 133, 126, 126, 126, 12, 112, 148, 148, 148, 148, 352, 357, 357, 357, 357, 357, 357, 357, 136, 136, 136, 136, -1, -1, -1]
test_index = [350, 30, 351, 29, 532, -1, 266, -1, 27, -1, 133, 126, 148, 352, 357, 357, 136]

def find_similar_pretrained_location(current_input, current_location_idx, pretrained_location_weights, projection_layer):
    """
    Args:
        current_input: [seq_len, input_size]
        current_location_idx: int 
        pretrained_location_weights: dict
        projection_layer: nn.Linear layer for similarity computation
    Returns:
        best_match_idx: int
    """
    
    # don't know how to compare raw input data with weights directly
    current_features = _extract_location_features(current_input)
    
    best_similarity = -float('inf')
    best_match_idx = 0
    
    for pretrained_idx, pretrained_weights in pretrained_location_weights.items():
        # veryt simple similarity
        similarity = _compute_similarity(current_features, pretrained_weights, projection_layer)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_idx = pretrained_idx
    
    return best_match_idx

def _extract_location_features(input_data):
    mean_features = input_data.mean(dim=1)  # [input_size]
    std_features = input_data.std(dim=1)    # [input_size]
    max_features = input_data.max(dim=1)[0] # [input_size]
    min_features = input_data.min(dim=1)[0] # [input_size]

    return mean_features

def _compute_similarity(current_features, pretrained_weights, projection_layer):
    pretrained_weight = pretrained_weights['weight']  # [1, hidden_dim*2]

    device = current_features.device
    pretrained_weight = pretrained_weight.to(device)
    
    current_projected = projection_layer(current_features)  # [1, 32]

    similarity = F.cosine_similarity(current_projected, pretrained_weight, dim=1)

    return similarity.item()

class DailyDecoder(nn.Module):
    def __init__(self, input_dim, daily_hidden_dim, annual_context_dim, output_dim=1, dataset=None, num_heads=4, dropout=0.1):
        super(DailyDecoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, daily_hidden_dim, num_layers=1, batch_first=True)
        
        # Learnable weight parameter for historical information
        self.historical_weight = nn.Parameter(torch.randn(1))
        self.pretrained_location_weights = None
        
        self.similarity_projection = nn.Linear(in_features=15, out_features=32)

        # Use separate output projections for each location
        # Get the actual number of locations from the dataset
        if dataset is not None and hasattr(dataset, '__location_length__'):
            self.max_locations = dataset.__location_length__()
        else:
            print("Warning: No dataset provided")
            exit()
        
        self.output_projections = nn.ModuleList([
            nn.Linear(daily_hidden_dim * 2, output_dim) for _ in range(self.max_locations)
        ])

        # Weight projection layer: project annual weights to LSTM hidden state dimension
        self.weight_projection = nn.Linear(annual_context_dim, daily_hidden_dim)
        

        self.fc = nn.Linear(daily_hidden_dim, output_dim)
        self.head = nn.Sequential(
            nn.Linear(daily_hidden_dim, daily_hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(daily_hidden_dim * 4, output_dim)
        )

    def forward(self, x_current, file_idx, training):
        # x_current: [B, 365, input_size=15] - all features
        # annual_weights: [B, context_len, output_dim, 2H]
        # attention_weights: [B, context_len]

        batch_size, seq_len, _ = x_current.shape
        device = x_current.device
        
        # Pre-compute historical enhancement if available
        historical_enhancement = None # [1, B, daily_hidden_dim]
        
        # create historical enhancement from pretrained location weights if available
        if self.pretrained_location_weights is not None:
            weights = []
            for loc in range(batch_size):
                location_idx = file_idx[loc].item()
                current_loc = x_current[loc:loc+1, :, :]  # [1, 365, input_size]
                if training: 
                    if training_index[location_idx] != -1:
                        best_match_idx = training_index[location_idx]
                        best_match_idx = f'location_{best_match_idx}'
                    else: # fallback to using cosine similarity to select
                        best_match_idx = find_similar_pretrained_location(current_loc, loc, self.pretrained_location_weights, self.similarity_projection)
                elif test_index[location_idx] != -1:
                    best_match_idx = test_index[location_idx]
                    best_match_idx = f'location_{best_match_idx}'
                else:
                    best_match_idx = find_similar_pretrained_location(current_loc, loc, self.pretrained_location_weights, self.similarity_projection)
                pretrained_weights = self.pretrained_location_weights[best_match_idx]
                 # Update the output projection for this location
                self.output_projections[location_idx].weight.data = pretrained_weights['weight']
                if pretrained_weights['bias'] is not None and self.output_projections[location_idx].bias is not None:
                    self.output_projections[location_idx].bias.data = pretrained_weights['bias']
                
                location_weights = self.output_projections[location_idx].weight  # [output_dim, 2H]
                weights.append(location_weights)

            weights = torch.stack(weights, dim=0).to(device=device)  # [B, output_dim, 2H]

            batch_size, output_dim, hidden_dim = weights.shape
            weights = weights.view(-1, hidden_dim)  # [B * output_dim, 2H]
            historical_enhancement = self.weight_projection(weights) # [B * output_dim, H]
            # normalize historical enhancement: scale to [-1, 1]
            historical_enhancement = 2 * (historical_enhancement - historical_enhancement.min()) / (historical_enhancement.max() - historical_enhancement.min()) - 1
            historical_enhancement = historical_enhancement.unsqueeze(0)  # [1, B, daily_hidden_dim]

        # Initialize hidden states
        h_states = [torch.zeros(1, batch_size, self.lstm.hidden_size, device=device) for _ in range(1)]
        c_states = [torch.zeros(1, batch_size, self.lstm.hidden_size, device=device) for _ in range(1)]

        
        # Apply historical enhancement to the final layer's hidden state, and maintain tanh scale
        # initial the last layer into historical enhancement, not sure useful
        # I can also try to use zero as hidden state, but let's try this first
        if historical_enhancement is not None:
            # print("Applying historical enhancement with weight: ", self.historical_weight)
            # print("historical enhancement: ", historical_enhancement)
            h_states[-1] = (h_states[-1] + self.historical_weight * historical_enhancement) / (1 + torch.abs(self.historical_weight))
        # print("enhanced hidden state: ", h_states[-1])
        # exit()

        lstm_out_enhanced, _ = self.lstm(x_current, (h_states[-1], c_states[-1]))  # [B, seq_len, hidden_size]

        prediction = self.head(lstm_out_enhanced).squeeze(-1) # [B, 365]

        # print(prediction.shape)
        # exit()
        return prediction  # prediction:[B, 365]


class Hierarchical(nn.Module):
    def __init__(self, input_dim, annual_hidden_dim, daily_hidden_dim, output_dim=1, dataset=None):
        super(Hierarchical, self).__init__()
        self.decoder = DailyDecoder(input_dim, daily_hidden_dim, annual_hidden_dim * 2, output_dim, dataset=dataset)

    def forward(self, x_current, file_idx, training):
        # x_current: [B, 365, input_size=15] - current year data
        # file_idx: [B] - location indices for each sample in the batch
        
        # Get main prediction
        prediction = self.decoder(x_current, file_idx, training) # [B, 365]
        
        return prediction
    


class LearnableHierarchicalWrapper(nn.Module):
    def __init__(self, input_dim, annual_hidden_dim, daily_hidden_dim, output_dim=1, dataset=None):
        super(LearnableHierarchicalWrapper, self).__init__()

        self.model = Hierarchical(input_dim, annual_hidden_dim, daily_hidden_dim, output_dim, dataset=dataset)


    def forward(self, x_current, file_idx, training):
        """
        Forward pass with second-order gradient support and adaptive contrastive learning
        Args:
            x_current: current year input data [B, 365, input_size]
            file_idx: location indices for each sample in the batch [B]
        """
        outputs = self.model(x_current, file_idx, training)  # [B, 365]
        return outputs