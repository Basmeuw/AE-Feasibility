from torch import nn
import torch.nn.init as init

class Bottleneck(nn.Module):
    def __init__(self, embedding_dim, bottleneck_dim):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(embedding_dim, bottleneck_dim)
        self.ln1 = nn.LayerNorm(bottleneck_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(bottleneck_dim, embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

        self.relu = nn.ReLU()

        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        init.xavier_uniform_(self.fc2.weight)
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)

    def encoder(self, x):
        return self.dropout(self.relu(self.ln1(self.fc1(x))))

    def decoder(self, x):
        return self.ln2(self.fc2(x))

    def forward(self, x):
        return self.decoder(self.encoder(x))