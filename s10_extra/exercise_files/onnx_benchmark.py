import torch
import onnxruntime as ort
import torchvision
import time
import sys
import statistics

def timing_decorator(func, repeat=5):
    def wrapper(*args, **kwargs):
        timing_results = []
        for _ in range(repeat):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            timing_results.append(elapsed_time)
        print(f"Avg +- Stddev: {statistics.mean(timing_results):0.3f} +- {statistics.stdev(timing_results):0.3f} seconds")
        return result
    return wrapper

model = torchvision.models.resnet18()
model.eval()

if sys.platform == "win32":
    # Windows doesn't support the new TorchDynamo-based ONNX Exporter
    torch.onnx.export(
        model, torch.randn(1, 3, 224, 224), "resnet18.onnx",
        dynamic_axes={"input.1": {0: "batch_size", 2: "height", 3: "width"}})
else:
    torch.onnx.dynamo_export(model, dummy_input).save("resnet18.onnx")

ort_session = ort.InferenceSession("resnet18.onnx")

@timing_decorator
def torch_predict(image, repeat=10):
    for _ in range(repeat):
        model(image)

@timing_decorator
def onnx_predict(image, repeat=10):
    for _ in range(repeat):
        ort_session.run(None, {"input.1": image.numpy()})

for size in [224, 448, 896]:
    dummy_input = torch.randn(1, 3, size, size)
    print(f"Image size: {size}")
    torch_predict(dummy_input)
    onnx_predict(dummy_input)
    