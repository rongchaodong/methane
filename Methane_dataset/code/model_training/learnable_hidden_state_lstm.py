import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HistoricalAttention(nn.Module):
    """Compute attention weights between current timestep and historical information"""
    def __init__(self, hidden_size, historical_dim=None):
        super(HistoricalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.historical_dim = historical_dim or hidden_size
        
        # Attention mechanism: current state as query, historical information as key
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(self.historical_dim, hidden_size)
        self.value_projection = nn.Linear(self.historical_dim, hidden_size)
        
        self.scale = 1.0 / (hidden_size ** 0.5)
        
    def forward(self, current_state, historical_data):
        """
        Args:
            current_state: (batch_size, hidden_size)
            historical_data: (batch_size, num_years, historical_dim)
        Returns:
            attention_weights: (batch_size, num_years)
            weighted_value: (batch_size, hidden_size)
        """
        batch_size, num_years, _ = historical_data.shape
        
        # Compute query, key, value
        query = self.query_projection(current_state)  # (batch_size, hidden_size)
        keys = self.key_projection(historical_data)   # (batch_size, num_years, hidden_size)
        values = self.value_projection(historical_data)  # (batch_size, num_years, hidden_size)
        
        # Compute attention scores
        scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2) * self.scale
        attention_weights = F.softmax(scores, dim=1)  # (batch_size, num_years)
        
        # Weighted sum
        weighted_value = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        
        return attention_weights, weighted_value

class InnerModel(nn.Module):
    """Inner model: learn historical information"""
    def __init__(self, input_size, hidden_size, output_size, model_type='lstm'):
        super(InnerModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model_type = model_type
        
        if model_type == 'lstm':
            self.model = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif model_type == 'linear':
            self.model = nn.Linear(input_size, hidden_size)
        elif model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size * 4),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, hidden_size)
            )
        
        self.output_projection = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_years, input_size)
        Returns:
            output: (batch_size, num_years, hidden_size)
        """
        batch_size, num_years, input_size = x.shape
        
        if self.model_type == 'lstm':
            # For LSTM, we need to process each year separately
            outputs = []
            for year in range(num_years):
                year_data = x[:, year, :].unsqueeze(1)  # (batch_size, 1, input_size)
                print(f"Year {year} data shape: {year_data.shape}")
                lstm_out, _ = self.model(year_data)
                outputs.append(lstm_out[:, -1, :])  # (batch_size, hidden_size)
            output = torch.stack(outputs, dim=1)  # (batch_size, num_years, hidden_size)
        else:
            # For linear and mlp, process each year directly
            x_reshaped = x.view(-1, input_size)  # (batch_size * num_years, input_size)
            output_reshaped = self.model(x_reshaped)  # (batch_size * num_years, hidden_size)
            output = output_reshaped.view(batch_size, num_years, -1)  # (batch_size, num_years, hidden_size)
            
        return output

class LearnableHiddenStateLSTM(nn.Module):
    """LSTM model with learnable hidden states"""
    def __init__(self, input_size, hidden_size, output_size, 
                 historical_input_size=None, historical_years=2,
                 inner_model_type='lstm', num_layers=1):
        super(LearnableHiddenStateLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.historical_input_size = historical_input_size or input_size
        self.historical_years = historical_years
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # Historical information attention mechanism
        self.historical_attention = HistoricalAttention(hidden_size, self.historical_input_size)
        
        # Inner model: learn historical information
        self.inner_model = InnerModel(
            self.historical_input_size, 
            hidden_size, 
            hidden_size,
            model_type=inner_model_type
        )
        
        # Learnable weight parameter
        self.historical_weight = nn.Parameter(torch.randn(1))
        
        # Output layer
        self.output_projection = nn.Linear(hidden_size, output_size)
        
        # Initialize parameters
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model parameters"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif param.dim() == 0:  # Scalar parameter
                nn.init.normal_(param, mean=0.0, std=0.1)
    
    def forward(self, x, historical_data=None, historical_targets=None, training_inner=True):
        """
        Args:
            x: (batch_size, seq_len, input_size) - 当前序列数据
            historical_data: (batch_size, num_years, seq_len, historical_input_size) - 历史数据
            historical_targets: (batch_size, num_years, seq_len, output_size) - 历史目标值
            training_inner: bool - 是否训练inner模型
        Returns:
            outputs: (batch_size, seq_len, output_size) - 预测输出
            inner_loss: float - inner模型损失
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize hidden states and cell states
        h_states = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]
        c_states = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]
        
        outputs = []
        inner_losses = []
        
        for t in range(seq_len):
            # Standard LSTM update
            current_input = x[:, t, :]
            
            # First LSTM layer
            h_new, c_new = self.lstm_layers[0](current_input, (h_states[0], c_states[0]))
            h_states[0], c_states[0] = h_new, c_new
            
            # Multi-layer LSTM
            for layer in range(1, self.num_layers):
                h_new, c_new = self.lstm_layers[layer](
                    h_states[layer-1], (h_states[layer], c_states[layer])
                )
                h_states[layer], c_states[layer] = h_new, c_new
            
            # Apply historical information enhancement
            if historical_data is not None and training_inner:
                # Compute historical information attention weights
                attention_weights, weighted_historical = self.historical_attention(
                    h_states[-1], historical_data[:, :, t, :]  # Get historical data for current timestep
                )
                
                # Process historical information through inner model
                historical_features = self.inner_model(historical_data[:, :, t, :])
                
                # Compute weighted average of historical information
                # attention_weights: (batch_size, num_years)
                # historical_features: (batch_size, num_years, hidden_size)
                weighted_historical_features = torch.sum(
                    attention_weights.unsqueeze(-1) * historical_features, dim=1
                )
                
                # Enhance hidden state
                h_states[-1] = h_states[-1] + self.historical_weight * weighted_historical_features
                
                # Compute inner model loss
                if historical_targets is not None:
                    inner_predictions = self.output_projection(historical_features)
                    inner_loss = self._compute_inner_loss(
                        inner_predictions, 
                        historical_targets[:, :, t, :],
                        attention_weights
                    )
                    inner_losses.append(inner_loss)
            
            # Output prediction
            output = self.output_projection(h_states[-1])
            outputs.append(output.unsqueeze(1))
        
        # Concatenate outputs from all timesteps
        final_outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len, output_size)
        
        # Compute total inner model loss
        inner_loss = torch.mean(torch.stack(inner_losses)) if inner_losses else torch.tensor(0.0, device=device)
        
        return final_outputs, inner_loss
    
    def _compute_inner_loss(self, predictions, targets, attention_weights):
        """
        Compute inner model loss: Σ A_k * ||f(x_k) - y_k||^2
        Args:
            predictions: (batch_size, num_years, output_size)
            targets: (batch_size, num_years, output_size)
            attention_weights: (batch_size, num_years)
        """
        # Compute MSE loss for each year
        mse_loss = F.mse_loss(predictions, targets, reduction='none')  # (batch_size, num_years, output_size)
        mse_loss = torch.mean(mse_loss, dim=-1)  # (batch_size, num_years)
        
        # Weighted sum
        weighted_loss = torch.sum(attention_weights * mse_loss, dim=1)  # (batch_size,)
        
        return torch.mean(weighted_loss)
    
    def predict_with_historical(self, x, historical_data):
        """
        Predict using historical data only (without updating inner model)
        Args:
            x: (batch_size, seq_len, input_size)
            historical_data: (batch_size, num_years, seq_len, historical_input_size)
        Returns:
            outputs: (batch_size, seq_len, output_size)
        """
        return self.forward(x, historical_data, training_inner=True)[0]

class LearnableHiddenStateLSTMWrapper(nn.Module):
    """Wrapper class providing simpler interface"""
    def __init__(self, input_size, hidden_size, output_size, 
                 historical_input_size=None, historical_years=5,
                 inner_model_type='lstm', num_layers=1,
                 inner_loss_weight=1.0):
        super(LearnableHiddenStateLSTMWrapper, self).__init__()
        
        self.model = LearnableHiddenStateLSTM(
            input_size, hidden_size, output_size,
            historical_input_size, historical_years,
            inner_model_type, num_layers
        )
        
        self.inner_loss_weight = inner_loss_weight
        
    def forward(self, x, historical_data=None, historical_targets=None):
        """
        Forward pass, return main prediction and total loss
        """
        outputs, inner_loss = self.model(x, historical_data, historical_targets)
        
        # Compute main task loss (real labels need to be passed here)
        # main_loss = F.mse_loss(outputs, targets) if targets is not None else torch.tensor(0.0, device=outputs.device)
        
        total_loss = inner_loss * self.inner_loss_weight
        # total_loss = main_loss + inner_loss * self.inner_loss_weight
        
        return outputs, total_loss, inner_loss
    
    def predict(self, x, historical_data=None):
        """Only perform prediction"""
        return self.model.predict_with_historical(x, historical_data)

# Example usage and test code
def test_learnable_hidden_state_lstm():
    """Test learnable hidden state LSTM model"""
    # Parameter settings
    batch_size = 4
    seq_len = 365  # One year of data
    input_size = 10
    hidden_size = 64
    output_size = 1
    historical_years = 5
    
    # Create model
    model = LearnableHiddenStateLSTMWrapper(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        historical_years=historical_years,
        inner_model_type='lstm'
    )
    
    # Create test data
    x = torch.randn(batch_size, seq_len, input_size)
    historical_data = torch.randn(batch_size, historical_years, seq_len, input_size)
    historical_targets = torch.randn(batch_size, historical_years, seq_len, output_size)
    
    print(f"Input shape: {x.shape}")
    print(f"Historical data shape: {historical_data.shape}")
    print(f"Historical targets shape: {historical_targets.shape}")
    
    # Forward pass
    outputs, total_loss, inner_loss = model(x, historical_data, historical_targets)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Inner model loss: {inner_loss.item():.4f}")
    
    # Prediction only mode
    predictions = model.predict(x, historical_data)
    print(f"Prediction shape: {predictions.shape}")
    
    return model

if __name__ == "__main__":
    # Run test
    model = test_learnable_hidden_state_lstm()
    print("Model test completed!")
