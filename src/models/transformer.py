import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x


class SpectralTransformer(nn.Module):
    def __init__(
            self,
            input_length=1000,
            patch_size=10,
            d_model=64,
            nhead=4,
            num_layers=2,
            num_classes=5):
        """
        Transformer for Exoplanet Spectrum Classification
        The spectrum is divided into non-overlapping patches.
        """
        super(SpectralTransformer, self).__init__()

        self.patch_size = patch_size
        self.seq_length = input_length // patch_size

        self.patch_embedding = nn.Linear(patch_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x is (batch, channels=1, length)
        # Reshape to patches: (batch, seq_length, patch_size)
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_length, self.patch_size)

        # Embed patches: (batch, seq_length, d_model)
        x = self.patch_embedding(x)

        # Transformer expects (seq_length, batch, d_model)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)

        # Pass through transformer
        x = self.transformer_encoder(x)

        # Pool across sequence dimension (mean pooling)
        # x is (seq_length, batch, d_model) -> (batch, d_model)
        x = x.mean(dim=0)

        # Classify
        logits = self.fc(x)
        probas = torch.sigmoid(logits)

        return probas
