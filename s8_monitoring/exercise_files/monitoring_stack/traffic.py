"""Replay a folder of images through the API, as production traffic: images only, no labels.

    uv run python traffic.py data/rps-test-set                       # the local API
    uv run python traffic.py data/rps-test-set --repeat 5 --delay 0.05
    uv run python traffic.py data/rps-test-set --url https://<your-service>.run.app

The images are sent in shuffled order, with a few malformed requests mixed in, as real clients send.
"""

import argparse
import io
import random
import time
from pathlib import Path

import requests
from dataset import image_files
from PIL import Image


def malformed(rng: random.Random) -> bytes:
    """A request a real client sends by mistake: not an image, a cut-off image or an empty file."""
    kind = rng.choice(["text", "truncated", "empty"])
    if kind == "text":
        return b"image not available, try again later\n"
    if kind == "empty":
        return b""
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), (200, 180, 160)).save(buf, format="JPEG")
    return buf.getvalue()[: len(buf.getvalue()) // 2]


def main() -> None:
    """Send every image in the folder to /predict, ``--repeat`` times."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="a folder of images; subfolders are included")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--repeat", type=int, default=1, help="replay the folder this many times")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to wait between requests")
    parser.add_argument("--malformed-every", type=int, default=50, help="one malformed request per N images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = image_files(args.folder)
    if not paths:
        msg = f"no images in {args.folder}"
        raise SystemExit(msg)
    rng = random.Random(args.seed)
    session = requests.Session()
    try:
        version = session.get(f"{args.url}/health", timeout=30).json()["model_version"]
    except requests.ConnectionError:
        msg = f"no API at {args.url}: is it running?"
        raise SystemExit(msg) from None
    print(f"sending {args.repeat} x {len(paths)} images to {args.url} (model version {version})")

    answered, rejected, sent = 0, 0, 0
    started = time.time()
    for _ in range(args.repeat):
        rng.shuffle(paths)
        for path in paths:
            sent += 1
            if sent % args.malformed_every == 0:
                r = session.post(f"{args.url}/predict", files={"file": ("image.jpg", malformed(rng))}, timeout=30)
                rejected += r.status_code != 200
            # A neutral file name: the API must not be able to read the class from it.
            files = {"file": ("image" + path.suffix, path.read_bytes())}
            r = session.post(f"{args.url}/predict", files=files, timeout=30)
            if r.status_code == 200:
                answered += 1
            else:
                rejected += 1
            if sent % 200 == 0:
                print(f"  {sent} images sent, {answered} answered, {rejected} rejected", flush=True)
            time.sleep(args.delay)
    print(f"done in {time.time() - started:.0f} s: {answered} answered, {rejected} rejected")


if __name__ == "__main__":
    main()
