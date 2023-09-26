# Adjusted version of
# https://github.com/Lightning-AI/lightning/blob/master/examples/pl_basics/backbone_image_classifier.py
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.demos.mnist_datamodule import MNIST
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


class LitClassifier(LightningModule):
    """Basic MNIST classifier."""

    def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = nn.Linear(28 * 28, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        """Forward pass of the network."""
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("valid_loss", loss, on_step=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        """Configure the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class MyDataModule(LightningDataModule):
    """Data module for MNIST."""

    def __init__(self, batch_size: int = 32):
        super().__init__()
        dataset = MNIST("Datasets", train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = MNIST("Datasets", train=False, download=True, transform=transforms.ToTensor())
        self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])
        self.batch_size = batch_size

    def train_dataloader(self):
        """Train dataloader."""
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        """Validation dataloader."""
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        """Test dataloader."""
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        """Predict dataloader."""
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


def cli_main():
    """Main cli function."""
    cli = LightningCLI(
        LitClassifier,
        MyDataModule,
        seed_everything_default=1234,
        save_config_overwrite=True,
        run=False,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_main()
