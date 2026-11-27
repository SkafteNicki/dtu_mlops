"""Train the small CNN on the training set and save a checkpoint for the API to serve.

    uv run python train.py              # about 1 minute on a laptop CPU; reaches about 80% test accuracy
    uv run python train.py --epochs 1   # quicker, a bit less accurate

This is a stand-in until the Closing the loop module, where training runs as a managed job instead.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from dataset import DATA_DIR, load_folder
from model import Net, save_checkpoint


def evaluate(model: Net, images: torch.Tensor, labels: torch.Tensor) -> float:
    """Accuracy of ``model`` on a set of images."""
    model.eval()
    with torch.no_grad():
        return float((model(images).argmax(1) == labels).float().mean())


def main() -> None:
    """Train, evaluate on the test set and save the checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("models/model.pt"))
    args = parser.parse_args()

    torch.manual_seed(0)
    x_train, y_train = load_folder(DATA_DIR / "rps")
    x_test, y_test = load_folder(DATA_DIR / "rps-test-set")
    print(f"training on {len(x_train)} images, testing on {len(x_test)}")

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for epoch in range(args.epochs):
        model.train()
        for batch in torch.randperm(len(x_train)).split(64):
            optimizer.zero_grad()
            F.cross_entropy(model(x_train[batch]), y_train[batch]).backward()
            optimizer.step()
        print(f"epoch {epoch + 1}: test accuracy {evaluate(model, x_test, y_test):.3f}")

    save_checkpoint(model, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
