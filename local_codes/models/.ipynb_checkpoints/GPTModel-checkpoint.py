import torch
import torch.nn as nn

class DecoderOnlyModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_length):
        super(DecoderOnlyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(embed_size, num_heads)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        # Embed the input tokens and add positional encodings
        x = self.embedding(x)
        
        # Apply Transformer decoder layers
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, x)
        
        return self.fc_out(x)