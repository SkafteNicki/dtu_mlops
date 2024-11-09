from __future__ import annotations

import bentoml
import numpy as np
from onnxruntime import InferenceSession


@bentoml.service
class ImageClassifierService:
    """Image classifier service using ONNX model."""

    def __init__(self) -> None:
        self.model = InferenceSession("model.onnx")

    @bentoml.api
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict the class of the input image."""
        output = self.model.run(None, {"input": image.astype(np.float32)})
        return output[0]
