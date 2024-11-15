class FNNClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        vocab_size = 21,
        embedding_dim = 128,
        max_seq_len = 1000
    ):
        super(FNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * max_seq_len, 512)  # Output size depends on conv and pooling layers
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)  # No softmax here, because we'll use CrossEntropyLoss, which applies it internally
        return output
