"""Tests for the solution API: fast, offline, with a tiny untrained model and synthetic images.

    uv run pytest s8_monitoring/exercise_files/monitoring_stack/test_api.py

No data download, network, Docker or credentials are needed.
"""

import io

import numpy as np
import pytest
from dataset import CLASSES
from fastapi.testclient import TestClient
from model import Net, save_checkpoint
from PIL import Image
from prometheus_client.parser import text_string_to_metric_families


def synthetic_image(seed: int, size: tuple[int, int] = (120, 90), fmt: str = "PNG") -> bytes:
    """A random RGB image of any size, encoded as a file."""
    pixels = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format=fmt)
    return buf.getvalue()


def metric_value(client: TestClient, sample_name: str, **labels: str) -> float:
    """The current value of one sample on /metrics (0 if it does not exist yet)."""
    for family in text_string_to_metric_families(client.get("/metrics").text):
        for sample in family.samples:
            if sample.name == sample_name and all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return 0.0


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory):
    """The solution API, serving a tiny untrained model."""
    checkpoint = tmp_path_factory.mktemp("model") / "model.pt"
    save_checkpoint(Net(width=4), checkpoint)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODEL_CHECKPOINT", str(checkpoint))
        from api_solution import app

        with TestClient(app) as test_client:
            yield test_client


@pytest.mark.parametrize(("size", "fmt"), [((64, 64), "PNG"), ((300, 200), "JPEG"), ((90, 160), "PNG")])
def test_predict_returns_class_confidence_and_model_version(client: TestClient, size: tuple, fmt: str) -> None:
    """Any image size is answered with a class, a confidence and the served model version."""
    response = client.post("/predict", files={"file": ("image", synthetic_image(0, size, fmt))})
    assert response.status_code == 200
    answer = response.json()
    assert answer["class"] in CLASSES
    assert 0.0 <= answer["confidence"] <= 1.0
    assert answer["model_version"] == client.get("/health").json()["model_version"]
    assert answer["model_version"].startswith("sha-")


def test_metrics_expose_the_three_service_metrics(client: TestClient) -> None:
    """/metrics has the request counter, the latency histogram and the error counter."""
    client.post("/predict", files={"file": ("image", b"")})  # so the error counter has a labelled sample
    families = {f.name: f.type for f in text_string_to_metric_families(client.get("/metrics").text)}
    assert families["predict_requests"] == "counter"
    assert families["predict_latency_seconds"] == "histogram"
    assert families["predict_errors"] == "counter"


def test_answered_request_is_counted_by_class_and_timed(client: TestClient) -> None:
    """An answered request adds one to its class's counter and one observation to the histogram."""
    latency_before = metric_value(client, "predict_latency_seconds_count")
    predicted = client.post("/predict", files={"file": ("image", synthetic_image(1))}).json()["class"]
    count_before = metric_value(client, "predict_requests_total", predicted_class=predicted)

    assert client.post("/predict", files={"file": ("image", synthetic_image(1))}).json()["class"] == predicted

    assert metric_value(client, "predict_requests_total", predicted_class=predicted) == count_before + 1
    assert metric_value(client, "predict_latency_seconds_count") == latency_before + 2


@pytest.mark.parametrize(
    ("data", "reason"),
    [(b"this is not an image", "not_an_image"), (synthetic_image(2, fmt="JPEG")[:300], "not_an_image"), (b"", "empty")],
    ids=["text", "truncated", "empty"],
)
def test_malformed_request_is_counted_as_an_error(client: TestClient, data: bytes, reason: str) -> None:
    """A malformed request gets a 400, counts as an error with its reason, and is not timed."""
    errors_before = metric_value(client, "predict_errors_total", reason=reason)
    answered_before = metric_value(client, "predict_latency_seconds_count")

    response = client.post("/predict", files={"file": ("image.jpg", data)})

    assert response.status_code == 400
    assert metric_value(client, "predict_errors_total", reason=reason) == errors_before + 1
    assert metric_value(client, "predict_latency_seconds_count") == answered_before
