
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path


def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    # exchange with the corrupted mnist dataset
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784)

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data" / "corrupt_mnist"

    train_images = []
    train_labels = []

    for i in range(6):
        train_images.append(torch.load(DATA_DIR / f"train_images_{i}.pt"))
        train_labels.append(torch.load(DATA_DIR / f"train_target_{i}.pt"))

    train_images = torch.cat(train_images).unsqueeze(1).float()
    train_labels = torch.cat(train_labels).long()

    test_images = torch.load(DATA_DIR / "test_images.pt").unsqueeze(1).float()
    test_labels = torch.load(DATA_DIR / "test_target.pt").long()

    train = torch.utils.data.TensorDataset(train_images, train_labels)
    test  = torch.utils.data.TensorDataset(test_images, test_labels)

    return train, test

# Added from solution:
def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.show()

if __name__ == "__main__":
    train, test = corrupt_mnist()
    print(f"Size of training set: {len(train)}")
    print(f"Size of test set: {len(test)}")
    print(f"Shape of a training point {(train[0][0].shape, train[0][1].shape)}")
    print(f"Shape of a test point {(test[0][0].shape, test[0][1].shape)}")
    show_image_and_target(train.tensors[0][:25], train.tensors[1][:25])
