"""A small CNN for the classifier, and how its checkpoint is saved and loaded."""

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from dataset import CHANNELS, CLASSES
from torch import nn


class Net(nn.Module):
    """Three conv layers and a linear head. Inputs are float images in [0, 1], N x C x H x W."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.width = width
        self.c1 = nn.Conv2d(CHANNELS, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, 2 * width, 3, padding=1)
        self.c3 = nn.Conv2d(2 * width, 4 * width, 3, padding=1)
        self.fc = nn.Linear(4 * width, len(CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one score (logit) per class."""
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def save_checkpoint(model: Net, path: Path) -> None:
    """Save the weights and the width needed to rebuild the model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"width": model.width, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Path) -> tuple[Net, str]:
    """Load a checkpoint. Returns the model in eval mode and its version.

    The version is the start of the file's SHA-256 hash: two different models never report the same version.
    """
    path = Path(path)
    if not path.exists():
        msg = f"no model checkpoint at {path}: train one first with `uv run python train.py`"
        raise SystemExit(msg)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Net(width=checkpoint["width"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    version = "sha-" + hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    return model, version
