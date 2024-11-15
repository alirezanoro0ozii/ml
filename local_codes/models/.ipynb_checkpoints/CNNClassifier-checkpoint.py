class SequenceClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, max_seq_len):
        super(SequenceClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # +1 for padding
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(128 * (max_seq_len // 2), 512)  # Output size depends on conv and pooling layers
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        x = self.conv1(x)
        x = self.pool(torch.relu(x))  # Max pooling
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)  # No softmax here, because we'll use CrossEntropyLoss, which applies it internally
        return output
