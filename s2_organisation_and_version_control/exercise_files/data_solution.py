from pathlib import Path

import torch
import typer
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """Custom dataset for corrupt MNIST."""

    def __init__(self, data_dir: str | Path, train: bool = True) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        if train:
            imgs, targets = [], []
            for i in range(6):
                imgs.append(torch.load(self.data_dir / f"train_images_{i}.pt"))
                targets.append(torch.load(self.data_dir / f"train_target_{i}.pt"))
            self.images = torch.cat(imgs).unsqueeze(1).float()
            self.targets = torch.cat(targets).long()
        else:
            self.images = torch.load(self.data_dir / "test_images.pt").unsqueeze(1).float()
            self.targets = torch.load(self.data_dir / "test_target.pt").long()
        self.images = (self.images - self.images.mean()) / self.images.std()

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int):
        """Return a given sample from the dataset."""
        return self.images[idx], self.targets[idx]


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data and save it to processed directory."""
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess_data)
