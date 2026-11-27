"""Everything that is specific to the dataset: Rock-Paper-Scissors (RPS).

The API, the model, the training script and the traffic script import their dataset-specific values from here and
nowhere else. Swapping the dataset (for example to corrupted MNIST and drawn digits) means changing this file only.

    uv run python dataset.py    # download the RPS training and test sets to data/ (about 230 MB)
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

# Class names, in the order of the model's outputs. Alphabetical, like torchvision's ImageFolder.
CLASSES = ("paper", "rock", "scissors")
# The model sees RGB images of IMAGE_SIZE x IMAGE_SIZE pixels.
IMAGE_SIZE = 64
CHANNELS = 3

# RPS by Laurence Moroney (CC BY 2.0): CGI hands on a white background, one folder per class.
DATA_DIR = Path("data")
DOWNLOADS = {
    "rps": "https://storage.googleapis.com/learning-datasets/rps.zip",
    "rps-test-set": "https://storage.googleapis.com/learning-datasets/rps-test-set.zip",
}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


class BadImageError(ValueError):
    """The request did not contain an image we can classify. ``reason`` is a short label for metrics."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


def decode(data: bytes) -> Image.Image:
    """Turn the bytes of an uploaded file into an RGB image, or raise ``BadImageError``."""
    if not data:
        reason, detail = "empty", "the file is empty"
        raise BadImageError(reason, detail)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # decode now, so a truncated file fails here and not later
    except (UnidentifiedImageError, OSError) as err:
        reason, detail = "not_an_image", f"not a readable image: {err}"
        raise BadImageError(reason, detail) from None
    return img.convert("RGB")


def preprocess(img: Image.Image) -> torch.Tensor:
    """RGB image of any size -> float tensor 3 x IMAGE_SIZE x IMAGE_SIZE in [0, 1].

    The image is centre-cropped to a square before resizing, so a phone photo is not squashed.
    Training and serving both use this function: the same preprocessing on both sides.
    """
    img = ImageOps.fit(img.convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE))
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float().div(255)


def image_files(folder: Path) -> list[Path]:
    """All image files below ``folder``, sorted, so a replay is reproducible."""
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def load_folder(folder: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a ``<folder>/<class>/<image>`` tree into (images N x 3 x H x W, labels N)."""
    images, labels = [], []
    for k, name in enumerate(CLASSES):
        for path in image_files(Path(folder) / name):
            images.append(preprocess(Image.open(path)))
            labels.append(k)
    if not images:
        msg = f"no images in {folder}: download the data first with `uv run python dataset.py`"
        raise SystemExit(msg)
    return torch.stack(images), torch.tensor(labels)


def download(data_dir: Path = DATA_DIR) -> None:
    """Download and unpack the training and test sets into ``data_dir`` (skips what is already there)."""
    import requests  # only needed here, so the API image does not need it

    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in DOWNLOADS.items():
        if (data_dir / name).exists():
            print(f"{data_dir / name} exists, skipping")
            continue
        print(f"downloading {url} ...")
        response = requests.get(url, timeout=600)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            zf.extractall(data_dir)
        print(f"  {len(image_files(data_dir / name))} images in {data_dir / name}")


if __name__ == "__main__":
    download()
