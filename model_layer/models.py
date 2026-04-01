import torch.nn as nn


def _conv2d_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )


def _normalize_channels(channels, expected_length, default_channels):
    if channels is None:
        channels = default_channels
    channels = tuple(int(channel) for channel in channels)
    if len(channels) != expected_length:
        raise ValueError('Expected {} channels, got {}.'.format(expected_length, len(channels)))
    if min(channels) <= 0:
        raise ValueError('All channels must be positive integers.')
    return channels


class CNN1DEncoder(nn.Module):
    def __init__(self, channels=None, pooled_length=64):
        super().__init__()
        self.channels = _normalize_channels(channels, 3, (32, 64, 64))
        self.pooled_length = int(pooled_length)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, self.channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(self.channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(self.channels[0], self.channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(self.channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(self.channels[1], self.channels[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(self.channels[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(self.pooled_length)
        self.embedding_size = self.channels[-1] * self.pooled_length

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


class CNN1D(nn.Module):
    def __init__(self, output_size, channels=None, pooled_length=64):
        super().__init__()
        self.encoder = CNN1DEncoder(channels=channels, pooled_length=pooled_length)
        self.fc = nn.Linear(self.encoder.embedding_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x


class CNN2DEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_size=64, channels=None):
        super().__init__()
        default_channels = (hidden_size, hidden_size, hidden_size, hidden_size)
        self.channels = _normalize_channels(channels, 4, default_channels)
        self.in_channels = in_channels
        self.features = nn.Sequential(
            _conv2d_block(in_channels, self.channels[0]),
            _conv2d_block(self.channels[0], self.channels[1]),
            _conv2d_block(self.channels[1], self.channels[2]),
            _conv2d_block(self.channels[2], self.channels[3]),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_size = self.channels[-1]

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class CNN2D(nn.Module):
    def __init__(self, output_size, in_channels=3, hidden_size=64, channels=None):
        super().__init__()
        self.encoder = CNN2DEncoder(in_channels=in_channels, hidden_size=hidden_size, channels=channels)
        self.fc = nn.Linear(self.encoder.embedding_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x
