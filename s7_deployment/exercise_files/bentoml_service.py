from __future__ import annotations

from typing import TYPE_CHECKING

import bentoml
import numpy as np
from onnxruntime import InferenceSession

if TYPE_CHECKING:
    from PIL import Image


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class ImageClassifierService:
    """Image classifier service using ONNX model."""

    def __init__(self) -> None:
        self.model = InferenceSession("model.onnx")

    @bentoml.api
    def predict(self, image: Image.Image) -> list[float]:
        """Predict the class of the input image."""
        input_data = np.array(image).astype(np.float32)
        output = self.model.run(None, {"input": input_data})
        return output[0].tolist()
