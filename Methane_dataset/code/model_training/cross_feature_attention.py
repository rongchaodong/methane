from distutils.command.config import config
import torch
import torch.nn as nn
from typing import List, Tuple
from config import Config


class FeatureProjector(nn.Module):
    """
    Projects scalar features into a shared embedding space and adds a learnable
    feature-id embedding. This creates tokens along the feature dimension.
    """
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.scalar_proj = nn.Linear(1, d_model)
        self.feature_id_emb = nn.Embedding(num_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F]
        returns: [B, T, F, D]
        """
        b, t, f = x.shape
        x_expanded = x.unsqueeze(-1)  # [B, T, F, 1]
        proj = self.scalar_proj(x_expanded) # [B, T, F, D]

        # Create feature id embedding and broadcast to [B, T, F, D]
        device = x.device
        feature_ids = torch.arange(f, device=device)  # [F]
        feat_emb = self.feature_id_emb(feature_ids)  # [F, D]
        feat_emb = feat_emb.view(1, 1, f, -1).expand(b, t, f, -1)  # [B, T, F, D]

        return proj + feat_emb


class IntraGroupAttention(nn.Module):
    """
    Self-attention within a group over the feature tokens for each time step.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_group_tokens: torch.Tensor) -> torch.Tensor:
        """
        x_group_tokens: [B, T, Fg, D]
        returns per-time representation pooled across features: [B, T, D]
        """
        b, t, f, d = x_group_tokens.shape
        x_bt = x_group_tokens.view(b * t, f, d)  # [B*T, Fg, D]
        attn_out, _ = self.attn(x_bt, x_bt, x_bt)  # [B*T, Fg, D]
        attn_out = self.norm(attn_out + x_bt)
        # Mean-pool over feature tokens per time step
        pooled = attn_out.mean(dim=1)  # [B*T, D]
        return pooled.view(b, t, d)


class InterGroupAttention(nn.Module):
    """
    Cross-attention among group-level tokens (per time step, 3 tokens: soil, climate, methane).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, group_tokens: torch.Tensor) -> torch.Tensor:
        """
        group_tokens: [B, T, G=3, D]
        returns fused per-time representation: [B, T, D]
        """
        b, t, g, d = group_tokens.shape
        x_bt = group_tokens.view(b * t, g, d)  # [B*T, 3, D]
        attn_out, _ = self.attn(x_bt, x_bt, x_bt)  # [B*T, 3, D]
        attn_out = self.norm(attn_out + x_bt)
        fused = attn_out.mean(dim=1)  # [B*T, D]
        return fused.view(b, t, d)


class TemporalDecoder(nn.Module):
    """
    Temporal decoder over daily sequence producing per-day predictions.
    """
    def __init__(self, d_model: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(d_model, hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_seq: [B, T, D]
        returns: hidden_seq [B, T, H], pred [B, T]
        """
        h, _ = self.lstm(x_seq)
        h = self.norm(h)
        pred = self.fc(h).squeeze(-1)
        return h, pred


class TransformerDecoder(nn.Module):
    """
    Pure Transformer decoder for temporal prediction without LSTM.
    """
    def __init__(self, d_model: int, n_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_seq: [B, T, D]
        returns: hidden_seq [B, T, D], pred [B, T]
        """
        h = x_seq
        for layer in self.transformer_layers:
            h = layer(h)
        h = self.norm(h)
        pred = self.fc(h).squeeze(-1)
        return h, pred


class CrossFeatureAttentionModel(nn.Module):
    """
    Cross-feature attention with two stages:
      1) Intra-group self-attention over features within each group per day
      2) Inter-group self-attention across the three groups per day
    Followed by an LSTM temporal decoder over days to predict daily methane.
    """
    def __init__(self, config: Config, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.config = config
        self.num_features = config.input_size  # 15
        self.d_model = d_model
        self.dropout = config.dropout

        # Group definitions (indices based on user specification)
        self.climate_idx: List[int] = [0, 8, 10, 11, 12, 13]
        self.soil_idx: List[int] = [1, 2, 3, 4, 5, 6]
        self.methane_idx: List[int] = [8, 9, 14]

        self.projector = FeatureProjector(self.num_features, d_model)

        self.intra_climate = IntraGroupAttention(d_model, n_heads, self.dropout)
        self.intra_soil = IntraGroupAttention(d_model, n_heads, self.dropout)
        self.intra_methane = IntraGroupAttention(d_model, n_heads, self.dropout)

        self.inter_group = InterGroupAttention(d_model, n_heads, self.dropout)

        # LSTM decoder (original)
        self.lstm_decoder = TemporalDecoder(d_model, config.hidden_size, config.num_layers, self.dropout)
        
        # Transformer decoder (new)
        self.transformer_decoder = TransformerDecoder(d_model, n_heads, config.num_layers, self.dropout)

        self.fc = nn.Linear(config.hidden_size, config.output_size)

    def _gather_group_tokens(self, x_tokens: torch.Tensor, indices: List[int]) -> torch.Tensor:
        """
        x_tokens: [B, T, F, D]
        returns: [B, T, Fg, D]
        """
        idx_tensor = torch.tensor(indices, device=x_tokens.device)
        return x_tokens.index_select(dim=2, index=idx_tensor)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, T=365, F=15]
        returns: (lstm_hidden, lstm_prediction, transformer_prediction)
        """
        # Project scalars to tokens per feature
        tokens = self.projector(x)  # [B, T, F, D]

        # Intra-group attentions
        climate_tokens = self._gather_group_tokens(tokens, self.climate_idx)  # [B, T, Fc, D]
        soil_tokens = self._gather_group_tokens(tokens, self.soil_idx)        # [B, T, Fs, D]
        methane_tokens = self._gather_group_tokens(tokens, self.methane_idx)  # [B, T, Fm, D]

        climate_repr = self.intra_climate(climate_tokens)   # [B, T, D]
        soil_repr = self.intra_soil(soil_tokens)            # [B, T, D]
        methane_repr = self.intra_methane(methane_tokens)   # [B, T, D]

        # Stack as group tokens per time step and apply inter-group attention
        group_stack = torch.stack([soil_repr, climate_repr, methane_repr], dim=2)  # [B, T, 3, D]
        fused = self.inter_group(group_stack)  # [B, T, D]

        # LSTM temporal decoding and prediction
        lstm_hidden, lstm_pred = self.lstm_decoder(fused)  # lstm_pred: [B, T]
        
        # Transformer temporal decoding and prediction
        transformer_hidden, transformer_pred = self.transformer_decoder(fused)  # transformer_pred: [B, T]

        return lstm_hidden, lstm_pred, transformer_pred


class ImprovedIntraGroupAttention(nn.Module):
    """
    Improved self-attention that maintains batch and temporal structure.
    Processes each batch sample independently while preserving temporal relationships.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_group_tokens: torch.Tensor) -> torch.Tensor:
        """
        x_group_tokens: [B, T, Fg, D]
        returns per-time representation pooled across features: [B, T, D]
        """
        b, t, f, d = x_group_tokens.shape
        
        # Process each batch sample independently
        outputs = []
        for i in range(b):
            # [T, Fg, D] for each batch sample
            batch_tokens = x_group_tokens[i]  # [T, Fg, D]
            
            # Apply attention across features for each time step
            attn_out, _ = self.attn(batch_tokens, batch_tokens, batch_tokens)  # [T, Fg, D]
            attn_out = self.norm(attn_out + batch_tokens)
            
            # Mean-pool over feature tokens per time step
            pooled = attn_out.mean(dim=1)  # [T, D]
            outputs.append(pooled)
        
        # Stack batch outputs
        return torch.stack(outputs, dim=0)  # [B, T, D]


class ImprovedCrossFeatureAttentionModel(nn.Module):
    """
    Improved cross-feature attention model that maintains batch and temporal structure.
    Uses separate attention for each batch sample to preserve temporal relationships.
    """
    def __init__(self, config: Config, d_model: int = 16, n_heads: int = 4):
        super().__init__()
        self.config = config
        self.num_features = config.input_size  # 15
        self.d_model = d_model
        self.dropout = 0.1

        # Group definitions (indices based on user specification)
        self.climate_idx: List[int] = [0, 8, 10, 11, 12, 13]
        self.soil_idx: List[int] = [1, 2, 3, 4, 5, 6]
        self.methane_idx: List[int] = [8, 9, 14]

        self.projector = FeatureProjector(self.num_features, d_model)

        # Use improved attention that maintains batch structure
        self.intra_climate = ImprovedIntraGroupAttention(d_model, n_heads, self.dropout)
        self.intra_soil = ImprovedIntraGroupAttention(d_model, n_heads, self.dropout)
        self.intra_methane = ImprovedIntraGroupAttention(d_model, n_heads, self.dropout)

        self.inter_group = InterGroupAttention(d_model, n_heads, self.dropout)

        # # LSTM decoder (original)
        # self.lstm_decoder = TemporalDecoder(d_model, config.hidden_size, config.num_layers, self.dropout)
        
        # Transformer decoder (new)
        self.transformer_decoder = TransformerDecoder(d_model, n_heads, config.num_layers, self.dropout)

        self.fc = nn.Linear(config.hidden_size, config.output_size)

    def _gather_group_tokens(self, x_tokens: torch.Tensor, indices: List[int]) -> torch.Tensor:
        """
        x_tokens: [B, T, F, D]
        returns: [B, T, Fg, D]
        """
        idx_tensor = torch.tensor(indices, device=x_tokens.device)
        return x_tokens.index_select(dim=2, index=idx_tensor)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, T=365, F=15]
        returns: (lstm_hidden, lstm_prediction, transformer_prediction)
        """
        # Project scalars to tokens per feature
        tokens = self.projector(x)  # [B, T, F, D]

        # Intra-group attentions (improved version maintains batch structure)
        climate_tokens = self._gather_group_tokens(tokens, self.climate_idx)  # [B, T, Fc, D]
        soil_tokens = self._gather_group_tokens(tokens, self.soil_idx)        # [B, T, Fs, D]
        methane_tokens = self._gather_group_tokens(tokens, self.methane_idx)  # [B, T, Fm, D]

        climate_repr = self.intra_climate(climate_tokens)   # [B, T, D]
        soil_repr = self.intra_soil(soil_tokens)            # [B, T, D]
        methane_repr = self.intra_methane(methane_tokens)   # [B, T, D]

        # Stack as group tokens per time step and apply inter-group attention
        group_stack = torch.stack([soil_repr, climate_repr, methane_repr], dim=2)  # [B, T, 3, D]
        fused = self.inter_group(group_stack)  # [B, T, D]

        # # LSTM temporal decoding and prediction
        # lstm_hidden, lstm_pred = self.lstm_decoder(fused)  # lstm_pred: [B, T]
        
        # Transformer temporal decoding and prediction
        transformer_hidden, transformer_pred = self.transformer_decoder(fused)  # transformer_pred: [B, T]

        return transformer_pred, transformer_pred


class NoIntraGroupModel(nn.Module):
    """
    Ablation model: Without Intra-Group Attention
    Features are directly pooled per group, then inter-group attention is applied.
    Pipeline: features → pooling → inter-group → decoder
    """
    def __init__(self, config: Config, d_model: int = 16, n_heads: int = 4):
        super().__init__()
        self.config = config
        self.num_features = config.input_size  # 15
        self.d_model = d_model
        self.dropout = 0.1

        # Group definitions
        self.climate_idx: List[int] = [0, 8, 10, 11, 12, 13]
        self.soil_idx: List[int] = [1, 2, 3, 4, 5, 6]
        self.methane_idx: List[int] = [8, 9, 14]

        self.projector = FeatureProjector(self.num_features, d_model)
        
        # Only inter-group attention (no intra-group)
        self.inter_group = InterGroupAttention(d_model, n_heads, self.dropout)
        
        # Transformer decoder
        self.transformer_decoder = TransformerDecoder(d_model, n_heads, config.num_layers, self.dropout)

    def _gather_group_tokens(self, x_tokens: torch.Tensor, indices: List[int]) -> torch.Tensor:
        """
        x_tokens: [B, T, F, D]
        returns: [B, T, Fg, D]
        """
        idx_tensor = torch.tensor(indices, device=x_tokens.device)
        return x_tokens.index_select(dim=2, index=idx_tensor)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T=365, F=15]
        returns: (prediction, prediction)
        """
        # Project scalars to tokens per feature
        tokens = self.projector(x)  # [B, T, F, D]

        # Get group tokens
        climate_tokens = self._gather_group_tokens(tokens, self.climate_idx)  # [B, T, Fc, D]
        soil_tokens = self._gather_group_tokens(tokens, self.soil_idx)        # [B, T, Fs, D]
        methane_tokens = self._gather_group_tokens(tokens, self.methane_idx)  # [B, T, Fm, D]

        # Simple mean pooling instead of intra-group attention
        climate_repr = climate_tokens.mean(dim=2)  # [B, T, D]
        soil_repr = soil_tokens.mean(dim=2)        # [B, T, D]
        methane_repr = methane_tokens.mean(dim=2)  # [B, T, D]

        # Stack and apply inter-group attention
        group_stack = torch.stack([soil_repr, climate_repr, methane_repr], dim=2)  # [B, T, 3, D]
        fused = self.inter_group(group_stack)  # [B, T, D]

        # Transformer temporal decoding and prediction
        _, transformer_pred = self.transformer_decoder(fused)  # transformer_pred: [B, T]

        return transformer_pred, transformer_pred


class NoInterGroupModel(nn.Module):
    """
    Ablation model: Without Inter-Group Attention
    Features go through intra-group attention, then directly pooled and fed to decoder.
    Pipeline: features → intra-group → pooling → decoder
    """
    def __init__(self, config: Config, d_model: int = 16, n_heads: int = 4):
        super().__init__()
        self.config = config
        self.num_features = config.input_size  # 15
        self.d_model = d_model
        self.dropout = 0.1

        # Group definitions
        self.climate_idx: List[int] = [0, 8, 10, 11, 12, 13]
        self.soil_idx: List[int] = [1, 2, 3, 4, 5, 6]
        self.methane_idx: List[int] = [8, 9, 14]

        self.projector = FeatureProjector(self.num_features, d_model)

        # Only intra-group attention (no inter-group)
        self.intra_climate = ImprovedIntraGroupAttention(d_model, n_heads, self.dropout)
        self.intra_soil = ImprovedIntraGroupAttention(d_model, n_heads, self.dropout)
        self.intra_methane = ImprovedIntraGroupAttention(d_model, n_heads, self.dropout)

        # Transformer decoder
        self.transformer_decoder = TransformerDecoder(d_model, n_heads, config.num_layers, self.dropout)

    def _gather_group_tokens(self, x_tokens: torch.Tensor, indices: List[int]) -> torch.Tensor:
        """
        x_tokens: [B, T, F, D]
        returns: [B, T, Fg, D]
        """
        idx_tensor = torch.tensor(indices, device=x_tokens.device)
        return x_tokens.index_select(dim=2, index=idx_tensor)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T=365, F=15]
        returns: (prediction, prediction)
        """
        # Project scalars to tokens per feature
        tokens = self.projector(x)  # [B, T, F, D]

        # Intra-group attentions
        climate_tokens = self._gather_group_tokens(tokens, self.climate_idx)  # [B, T, Fc, D]
        soil_tokens = self._gather_group_tokens(tokens, self.soil_idx)        # [B, T, Fs, D]
        methane_tokens = self._gather_group_tokens(tokens, self.methane_idx)  # [B, T, Fm, D]

        climate_repr = self.intra_climate(climate_tokens)   # [B, T, D]
        soil_repr = self.intra_soil(soil_tokens)            # [B, T, D]
        methane_repr = self.intra_methane(methane_tokens)   # [B, T, D]

        # Simple mean pooling instead of inter-group attention
        fused = (climate_repr + soil_repr + methane_repr) / 3.0  # [B, T, D]

        # Transformer temporal decoding and prediction
        _, transformer_pred = self.transformer_decoder(fused)  # transformer_pred: [B, T]

        return transformer_pred, transformer_pred

