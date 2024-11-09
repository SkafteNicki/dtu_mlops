from __future__ import annotations

import asyncio

import bentoml
import numpy as np
from onnxruntime import InferenceSession


@bentoml.service
class ImageClassifierServiceModelA:
    """Image classifier service using ONNX model."""

    def __init__(self) -> None:
        self.model = InferenceSession("model_a.onnx")

    @bentoml.api
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict the class of the input image."""
        output = self.model.run(None, {"input": image.astype(np.float32)})
        return output[0]


@bentoml.service
class ImageClassifierServiceModelB:
    """Image classifier service using ONNX model."""

    def __init__(self) -> None:
        self.model = InferenceSession("model_b.onnx")

    @bentoml.api
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict the class of the input image."""
        output = self.model.run(None, {"input": image.astype(np.float32)})
        return output[0]


@bentoml.service
class ImageClassifierService:
    """Image classifier service using ONNX model."""

    model_a = bentoml.depends(ImageClassifierServiceModelA)
    model_b = bentoml.depends(ImageClassifierServiceModelB)

    @bentoml.api
    async def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict the class of the input image."""
        result_a, result_b = await asyncio.gather(
            self.model_a.to_async.predict(image), self.model_b.to_async.predict(image)
        )
        return (result_a + result_b) / 2
