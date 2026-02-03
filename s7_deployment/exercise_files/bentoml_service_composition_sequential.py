from __future__ import annotations

from pathlib import Path

import bentoml
import numpy as np
from onnxruntime import InferenceSession
from PIL import Image


@bentoml.service
class ImagePreprocessorService:
    """Image preprocessor service."""

    @bentoml.api
    def preprocess(self, image_file: Path) -> np.ndarray:
        """Preprocess the input image."""
        image = Image.open(image_file)
        image = image.resize((224, 224))
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, axis=0)


@bentoml.service
class ImageClassifierService:
    """Image classifier service using ONNX model."""

    preprocessing_service = bentoml.depends(ImagePreprocessorService)

    def __init__(self) -> None:
        self.model = InferenceSession("model.onnx")

    @bentoml.api
    async def predict(self, image_file: Path) -> np.ndarray:
        """Predict the class of the input image."""
        image = await self.preprocessing_service.to_async.preprocess(image_file)
        output = self.model.run(None, {"input": image.astype(np.float32)})
        return output[0]
