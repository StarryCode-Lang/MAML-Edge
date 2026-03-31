import torch.nn as nn


def _conv2d_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )


class CNN1DEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(64)
        self.embedding_size = 64 * 64

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


class CNN1D(nn.Module):
    def __init__(self, output_size):
        super(CNN1D, self).__init__()
        self.encoder = CNN1DEncoder()
        self.fc = nn.Linear(self.encoder.embedding_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x


class CNN2DEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_size=64):
        super().__init__()
        self.features = nn.Sequential(
            _conv2d_block(in_channels, hidden_size),
            _conv2d_block(hidden_size, hidden_size),
            _conv2d_block(hidden_size, hidden_size),
            _conv2d_block(hidden_size, hidden_size),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_size = hidden_size

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class CNN2D(nn.Module):
    def __init__(self, output_size, in_channels=3, hidden_size=64):
        super().__init__()
        self.encoder = CNN2DEncoder(in_channels=in_channels, hidden_size=hidden_size)
        self.fc = nn.Linear(self.encoder.embedding_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x
