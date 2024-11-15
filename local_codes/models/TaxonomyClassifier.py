import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

class TaxonomyClassifier(nn.Module):
    def __init__(
        self,
        # Dictionary of taxonomy levels and their possible classes
        taxonomy_levels,
        vocab_size=21,
        d_model=256,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=512,
        dropout=0.1
    ):
        super().__init__()
        
        # Sequence embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            num_encoder_layers
        )
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Create classifier heads for each taxonomy level
        self.classifier_heads = nn.ModuleDict({
            level: nn.Linear(d_model, num_classes, bias=False)
            for level, num_classes in taxonomy_levels.items()
        })
        
        self.d_model = d_model
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Embedding and positional encoding
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.pos_encoder(src)
        
        # Transform sequence
        encoder_output = self.transformer_encoder(
            src,
        )
        
        # Global average pooling
        sequence_features = torch.mean(encoder_output, dim=1)
        
        # Extract shared features
        shared_features = self.feature_extractor(sequence_features)
        
        # Get predictions for each taxonomy level
        predictions = {
            level: head(shared_features)
            for level, head in self.classifier_heads.items()
        }
        
        return predictions

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)