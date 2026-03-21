import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DModel(nn.Module):
    def __init__(self, input_length=1000, num_classes=5):
        """
        1D CNN for Exoplanet Spectrum Classification
        """
        super(CNN1DModel, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=11,
            padding=5)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=7,
            padding=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.conv3 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=4)

        # Calculate flattened size
        # length: 1000 -> 250 -> 62 -> 15
        flat_length = input_length // 4 // 4 // 4

        self.fc1 = nn.Linear(64 * flat_length, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x is (batch, channels=1, length)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        # Sigmoid for multi-label classification
        probas = torch.sigmoid(logits)
        return probas
