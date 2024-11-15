class SimpleAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_taxonomy_ids, num_attention_layers=3, dropout_rate=0.1):
        super(SimpleAttentionClassifier, self).__init__()
        
        # Embedding layer for sequences
        self.sequence_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Stack of attention layers with normalization, dropout, and skip connections
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True) for _ in range(num_attention_layers)
        ])
        
        # Layer normalization for each attention layer
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_attention_layers)])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layer for predicting taxonomy
        self.fc = nn.Linear(embedding_dim, num_taxonomy_ids)

    def forward(self, sequences):
        # Embed the input sequences
        embedded_seq = self.sequence_embedding(sequences)  # (batch_size, seq_len, embed_dim)
        
        # Pass through multiple attention layers with skip connections, layer normalization, and dropout
        for attention_layer, norm_layer in zip(self.attention_layers, self.norm_layers):
            # Attention mechanism (self-attention here)
            attn_output, _ = attention_layer(embedded_seq, embedded_seq, embedded_seq)
            
            # Add skip connection: output + input
            attn_output = attn_output + embedded_seq  # Skip connection (Residual connection)
            
            # Apply normalization
            attn_output = norm_layer(attn_output)
            
            # Apply dropout
            attn_output = self.dropout(attn_output)
            
            # Update input for the next attention layer
            embedded_seq = attn_output
        
        # Mean pooling across the sequence length dimension
        attn_output = attn_output.mean(dim=1)  # (batch_size, embed_dim)
        
        # Pass through a fully connected layer to predict taxonomy IDs
        output = self.fc(attn_output)  # (batch_size, num_taxonomy_ids)
        
        return output
