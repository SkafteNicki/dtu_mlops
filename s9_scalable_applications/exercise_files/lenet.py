from torch import nn


class LeNet(nn.Module):
    """LeNet implementation."""

    def __init__(self) -> None:
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Forward pass of the network."""
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv1(x)), (2, 2))
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = LeNet()

    # print all the parameters of the model
    for m in model.named_modules():
        print(m)

    # print parameters of the first module
    module_1 = model.conv1
    print(list(module_1.named_parameters()))
