import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel, BertModel

# 1. LSTM Model
class ProteinLSTM(nn.Module):
    def __init__(self, input_dim=21, embedding_dim=640, hidden_dim=640, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for bidirectional
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # Weighted sum over all residues (simplified version)
        weights = torch.softmax(torch.ones_like(lstm_out[:,:,0]), dim=1).unsqueeze(-1)
        weighted_sum = (lstm_out * weights).sum(dim=1)
        output = self.linear(weighted_sum)
        return self.tanh(output)

# 2. Transformer Model
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class ProteinTransformer(nn.Module):
    def __init__(self, input_dim=24, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                 dim_feedforward=2048, activation=GELU())
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        # Get [CLS] token output (assuming it's the first token)
        cls_output = x[:,0,:]
        return self.tanh(self.output_layer(cls_output))

# 3. CNN Model
class ProteinCNN(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=1024):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, input_dim)  # One-hot embedding
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2)
        
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)  # Convert to channel-first
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return F.max_pool1d(x, x.size(2)).squeeze(2)  # Max pooling over all residues

# 4. ResNet Model
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)

class ProteinResNet(nn.Module):
    def __init__(self, num_tokens=21, embed_dim=512, num_blocks=8):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, embed_dim)
        self.blocks = nn.ModuleList([ResidualBlock(embed_dim) for _ in range(num_blocks)])
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)  # Convert to channel-first
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)  # Convert back for attention
        # Attentive weighted sum
        attn_output, _ = self.attention(x, x, x)
        return attn_output.mean(dim=1)  # Average over sequence length

# 5. ProtBERT using HuggingFace
class ProtBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("Rostlab/prot_bert")
        self.tanh = nn.Tanh()
        
    def forward(self, x, attention_mask=None):
        outputs = self.bert(x, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        return self.tanh(cls_output)

# 6. ESM-1b using HuggingFace
class ESM1b(nn.Module):
    def __init__(self):
        super().__init__()
        self.esm = EsmModel.from_pretrained("facebook/esm1b_t33_650M_UR50S")
        
    def forward(self, x, attention_mask=None):
        outputs = self.esm(x, attention_mask=attention_mask)
        # Mean pooling over all residues
        return outputs.last_hidden_state.mean(dim=1)

# Initialize models and count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize models
lstm_model = ProteinLSTM()
transformer_model = ProteinTransformer()
cnn_model = ProteinCNN()
resnet_model = ProteinResNet()
protbert_model = ProtBERT()
esm1b_model = ESM1b()

# Print parameter counts
print("Parameter counts:")
print(f"LSTM: {count_parameters(lstm_model):,}")
print(f"Transformer: {count_parameters(transformer_model):,}")
print(f"CNN: {count_parameters(cnn_model):,}")
print(f"ResNet: {count_parameters(resnet_model):,}")
print(f"ProtBERT: {count_parameters(protbert_model):,}")
print(f"ESM-1b: {count_parameters(esm1b_model):,}")