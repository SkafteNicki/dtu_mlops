from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
