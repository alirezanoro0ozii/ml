import torch.nn as nn

class new_Transformer(nn.Module):
    def __init__(
        self,
        output_dim,
        max_seq_len=1000,
        vocab_size=21,
        d_model=256
    ):
        super().__init__()
        
        # Sequence embedding layers
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(output_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=max_seq_len)

        self.d_model = d_model
        self.transformer = nn.Transformer(
            d_model = d_model,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            bias= False
        )
        
    def forward(self, src, tgt, tgt_mask=None):
        # Embedding and positional encoding
        src = self.encoder_embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.pos_encoder(src)

        tgt = self.decoder_embedding(tgt)
        
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
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