import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, num_layers, num_classes, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, batch_first=True, dropout=dropout, activation="gelu"),
            num_layers
        )
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

