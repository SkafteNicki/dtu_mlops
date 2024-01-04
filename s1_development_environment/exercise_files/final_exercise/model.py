from torch import nn


myawesomemodel = nn.Sequential(
    nn.Conv2d(1, 32, 3),  # [B, 1, 28, 28] -> [B, 32, 26, 26]
    nn.LeakyReLU(),
    nn.Conv2d(32, 64, 3), # [B, 32, 26, 26] -> [B, 64, 24, 24]
    nn.LeakyReLU(),
    nn.MaxPool2d(2),      # [B, 64, 24, 24] -> [B, 64, 12, 12]
    nn.Flatten(),        # [B, 64, 12, 12] -> [B, 64 * 12 * 12]
    nn.Linear(64 * 12 * 12, 10),
)