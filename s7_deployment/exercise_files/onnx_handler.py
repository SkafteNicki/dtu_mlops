import json

import numpy as np
import onnxruntime as ort
from ts.torch_handler.base_handler import BaseHandler


class ONNXHandler(BaseHandler):
    """ONNX handler class for handling requests to the model server."""

    def initialize(self, ctx):
        """Initialize the model and any other required components."""
        # Initialize the model and any other required components
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        # Load the ONNX model
        model_path = f"{model_dir}/model.onnx"
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, data):
        """Preprocess the input data to a numpy array and return it."""
        # Preprocess the input data
        input_data = data[0].get("body")
        input_data = json.loads(input_data)
        input_array = np.array(input_data, dtype=np.float32)
        return input_array

    def inference(self, input_array):
        """Perform inference and return the output."""
        # Perform inference
        ort_inputs = {self.input_name: input_array}
        ort_outs = self.session.run([self.output_name], ort_inputs)
        return ort_outs

    def postprocess(self, inference_output):
        """Postprocess the inference output and return the result as json."""
        # Postprocess the inference output
        output_data = inference_output[0].tolist()
        return [json.dumps(output_data)]
