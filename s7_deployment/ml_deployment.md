![Logo](../figures/icons/onnx.png){ align=right width="130"}
![Logo](../figures/icons/bentoml.png){ align=right width="130"}

# Deployment of Machine Learning Models

---

In one of the [previous modules](../s7_deployment/apis.md) you learned about how to use
[FastAPI](https://fastapi.tiangolo.com/) to create an API to interact with your machine learning models. FastAPI is a
great framework, but it is a general framework meaning that it was not developed with machine learning applications in
mind. This means that there are features which you may consider to be missing when considering running large scale
machine learning models:

* Dynamic-batching: if you have a large number of requests coming in, you may want to process them in batches to
    reduce the overhead of loading the model and running the inference. This is especially true if you are running your
    model on a GPU, where the overhead of loading the model is significant.

* Async inference: FastAPi does support async requests but not no way to call the model asynchronously. This means that
    if you have a large number of requests coming in, you will have to wait for the model to finish processing (because
    the model is not async) before you can start processing the next request.

* Native GPU support: you can definitely run part of your application in FastAPI if you want to. But again it was not
    build with machine learning in mind, so you will have to do some extra work to get it to work.

It should come as no surprise that multiple frameworks have therefore sprung up that better supports deployment of
machine learning algorithms (just listing a few here):

```python exec="1"
# this code is being executed at build time to get the latest number of stars
import requests

def get_github_stars(owner_repo):
    url = f"https://api.github.com/repos/{owner_repo}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("stargazers_count", 0)
    else:
        return None

table =  "| 🌟 Framework | 🧩 Backend Agnostic | 🧠 Model Agnostic | 📂 Repository | ⭐ GitHub Stars |\n"
table += "|--------------|---------------------|-------------------|---------------|----------------|\n"

data = [
    ("Cortex", "Yes", "Yes", "cortexlabs/cortex"),
    ("BentoML", "Yes", "Yes", "bentoml/bentoml"),
    ("Ray Serve", "Yes", "Yes", "ray-project/ray"),
    ("Triton Inference Server", "Yes", "Yes", "NVIDIA/triton-inference-server"),
    ("OpenVINO", "Yes", "Yes", "openvinotoolkit/openvino"),
    ("Seldon-core", "Yes", "Yes", "seldonio/seldon-core"),
    ("Litserve", "Yes", "Yes", "Lightning-AI/LitServe"),
    ("Torchserve", "No", "Yes", "pytorch/serve"),
    ("TensorFlow serve", "No", "Yes", "tensorflow/serving"),
    ("vLLM", "No", "No", "vllm-project/vllm")
]

for framework, backend_agnostic, model_agnostic, repo in data:
    stars_count = get_github_stars(repo)
    stars = f"{stars_count / 1000:.1f}k" if stars_count is not None else "⭐ N/A"
    backend_emoji = "✅" if backend_agnostic == "Yes" else "❌"
    model_emoji = "✅" if model_agnostic == "Yes" else "❌"
    table += f"| {framework} | {backend_emoji} | {model_emoji} | [🔗 Link](https://github.com/{repo}) | {stars} |\n"

print(table)
```

The first 7 frameworks are backend agnostic, meaning that they are intended to work with whatever computational backend
you model is implemented in (TensorFlow, PyTorch, Jax, Sklearn etc.), whereas the last 3 are backend specific (PyTorch,
TensorFlow and a custom framework). The first 9 frameworks are model agnostic, meaning that they are intended to work
with whatever model you have implemented, whereas the last one is model specific in this case to LLM's. When choosing a
framework to deploy your model, you should consider the following:

* **Ease of use**. Some frameworks are easier to use and get started with than others, but may have fewer features. As
    an example from the list above, `Litserve` is very easy to get started with but is a relatively new framework and
    may not have all the features you need.

* **Performance**. Some frameworks are optimized for performance, but may be harder to use. As an example from the list
    above, `vLLM` is a very high performance framework for serving large language models but it cannot be used for other
    types of models.

* **Community**. Some frameworks have a large community, which can be helpful if you run into problems. As an example
    from the list above, `Triton Inference Server` is developed by Nvidia and has a large community of users. As a good
    rule of thumb, the more stars a repository has on GitHub, the larger the community.

In this module we are going to be looking at the `BentoML` framework because it strikes a good balance between ease of
use and having a lot of features that can improve the performance of serving your models. However, before we dive into
this serving framework, we are going to look at a general way to package our machine learning models that should work
with most of the above frameworks.

## Model Packaging

Whenever we want to serve a machine learning model, we in general need 3 things:

* The computational graph of the model, e.g. how to pass data through the model to get a prediction.
* The weights of the model, e.g. the parameters that the model has learned during training.
* A computational backend that can run the model

In the previous module on [Docker](../s3_reproducibility/docker.md) we learned how to package all of these things into
a container. This is a great way to package a model, but it is not the only way. The core assumption we currently have
made is that the computational backend is the same as the one we trained the model on. However, this does not need to
be the case. As long as we can export our model and weights to a common format, we can run the model on any backend
that supports this format.

This is exactly what the [Open Neural Network Exchange (ONNX)](https://onnx.ai/) is designed to do. ONNX is a
standardized format for creating and sharing machine learning models. It defines an extensible computation graph model,
as well as definitions of built-in operators and standard data types. The idea behind ONNX is that a model trained with
a specific framework on a specific device, let's say PyTorch on your local computer, can be exported and run with an
entirely different framework and hardware easily. Learning how to export your models to ONNX is therefore a great way
to increase the longevity of your models and not being locked into a specific framework for serving your models.

<figure markdown>
![Image](../figures/onnx.png){ width="1000" }
<figcaption>
The ONNX format is designed to bridge the gap between development and deployment of machine learning models, by making
it easy to export models between different frameworks and hardware. For example PyTorch is in general considered
an developer friendly framework, however it has historically been slow to run inference with compared to a framework.
<a href="https://medium.com/trueface-ai/two-benefits-of-the-onnx-library-for-ml-models-4b3e417df52e"> Image credit </a>
</figcaption>
</figure>

## ❔ Exercises

1. Start by installing ONNX, ONNX runtime and ONNX script. This can be done by running the following command

    ```bash
    pip install onnx onnxruntime onnxscript
    ```

    the first package contains the core ONNX framework, the second package contains the runtime for running ONNX models
    and the third package contains a new experimental package that is designed to make it easier to export models to
    ONNX.

2. Let's start out with converting a model to ONNX. The following code snippets shows how to export a PyTorch model to
    ONNX.

    === "PyTorch => 2.0"

        ```python
        import torch
        import torchvision

        model = torchvision.models.resnet18(weights=None)
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        onnx_model = torch.onnx.dynamo_export(
            model=model,
            model_args=(dummy_input,),
            export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
        )
        onnx_model.save("resnet18.onnx")
        ```

    === "PyTorch < 2.0 or Windows"

        ```python
        import torch
        import torchvision

        model = torchvision.models.resnet18(weights=None)
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model=model,
            args=(dummy_input,),
            f="resnet18.onnx",
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        ```

    === "PyTorch-lightning"

        ```python
        import torch
        import torchvision
        import pytorch_lightning as pl
        import onnx
        import onnxruntime

        class LitModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = torchvision.models.resnet18(pretrained=True)
                self.model.eval()

            def forward(self, x):
                return self.model(x)

        model = LitModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        model.to_onnx(
            file_path="resnet18.onnx",
            input_sample=dummy_input,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        ```

    Export a model of your own choice to ONNX or just try to export the `resnet18` model as shown in the examples above,
    and confirm that the model was exported by checking that the file exists. Can you figure out what is meant by
    `dynamic_axes`?

    ??? success "Solution"

        The `dynamic_axes` argument is used to specify which axes of the input tensor that should be considered dynamic.
        This is useful when the model can accept inputs of different sizes, e.g. when the model is used in a dynamic
        batching scenario. In the example above we have specified that the first axis of the input tensor should be
        considered dynamic, meaning that the model can accept inputs of different batch sizes. While it may be tempting
        to specify all axes as dynamic, however this can lead to slower inference times, because the ONNX runtime will
        not be able to optimize the computational graph as well.

3. Check that the model was correctly exported by loading it using the `onnx` package and afterwards check the graph
    of model using the following code:

    ```python
    import onnx
    model = onnx.load("resnet18.onnx")
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))
    ```

4. To get a better understanding of what is actually exported, lets try to visualize the computational graph of the
    model. This can be done using the open-source tool [netron](https://github.com/lutzroeder/netron). You can either
    try it out directly in [webbrowser](https://netron.app/) or you can install it locally using `pip install netron`
    and then run it using `netron resnet18.onnx`. Can you figure out what method of the model is exported to ONNX?

    ??? success "Solution"

        When a PyTorch model is exported to ONNX, it is only the `forward` method of the model that is exported. This
        means that it is the only method we have access to when we load the model later. Therefore, make sure that the
        `forward` method of your model is implemented in a way that it can be used for inference.

5. After converting a model to ONNX format we can use the [ONNX Runtime](https://onnxruntime.ai/docs/) to run it.
    The benefit of this is that ONNX Runtime is able to optimize the computational graph of the model, which can lead
    to faster inference times. Lets try to look into that.

    1. Figure out how to run a model using the ONNX Runtime. Relevant
        [documentation](https://onnxruntime.ai/docs/get-started/with-python.html).

        ??? success "Solution"

            To use the ONNX runtime to run a model, we first need to start a inference session, then extract input
            output names of our model and finally run the model. The following code snippet shows how to do this.

            ```python
            import onnxruntime as rt
            ort_session = rt.InferenceSession("<path-to-model>")
            input_names = [i.name for i in ort_session.get_inputs()]
            output_names = [i.name for i in ort_session.get_outputs()]
            batch = {input_names[0]: np.random.randn(1, 3, 224, 224).astype(np.float32)}
            out = ort_session.run(output_names, batch)
            ```

    2. Let's experiment with performance of ONNX vs. PyTorch. Implement a benchmark that measures the time it takes to
        run a model using PyTorch and ONNX. Bonus points if you test for multiple input sizes. To get you started we
        have implemented a timing decorator that you can use to measure the time it takes to run a function.

        ```python
        from statistics import mean, stdev
        import time
        def timing_decorator(func, function_repeat: int = 10, timing_repeat: int = 5):
            """ Decorator that times the execution of a function. """
            def wrapper(*args, **kwargs):
                timing_results = []
                for _ in range(timing_repeat):
                    start_time = time.time()
                    for _ in range(function_repeat):
                        result = func(*args, **kwargs)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    timing_results.append(elapsed_time)
                print(f"Avg +- Stddev: {mean(timing_results):0.3f} +- {stdev(timing_results):0.3f} seconds")
                return result
            return wrapper
        ```

        ??? success "Solution"

            ```python linenums="1" title="onnx_benchmark.py"
            --8<-- "s7_deployment/exercise_files/onnx_benchmark.py"
            ```

    3. To get a better understanding of why running the model using the ONNX runtime is usually faster lets try to see
        what happens to the computational graph. By default the ONNX Runtime will apply these optimization in *online*
        mode, meaning that the optimizations are applied when the model is loaded. However, it is also possible to apply
        the optimizations in *offline* mode, such that the optimized model is saved to disk. Below is an example of how
        to do this.

        ```python
        import onnxruntime as rt
        sess_options = rt.SessionOptions()

        # Set graph optimization level
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        # To enable model serialization after graph optimization set this
        sess_options.optimized_model_filepath = "optimized_model.onnx>"

        session = rt.InferenceSession("<model_path>", sess_options)
        ```

        Try to apply the optimizations in offline mode and use `netron` to visualize both the original and optimized
        model side by side. Can you see any differences?

        ??? success "Solution"

            You should hopefully see that the optimized model consist of fewer nodes and edges than the original model.
            These nodes are often called fused nodes, because they are the result of multiple nodes being fused
            together. In the image below we have visualized the first part of the computational graph of a resnet18
            model, before and after optimization.

            <figure markdown>
            ![Image](../figures/onnx_optimization.png){ width="600" }
            </figure>

6. As mentioned in the introduction, ONNX is able to run on many different types of hardware and execution engine.
    You can check all providers and all the available providers by running the following code

    ```python
    import onnxruntime
    print(onnxruntime.get_all_providers())
    print(onnxruntime.get_available_providers())
    ```

    Can you figure out how to set which provide the ONNX runtime should use?

    ??? success "Solution"

        The provider that the ONNX runtime should use can be set by passing the `providers` argument to the
        `InferenceSession` class. A list should be provided, which prioritizes the providers in the order they are
        listed.

        ```python
        import onnxruntime as rt
        provider_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ort_session = rt.InferenceSession("<path-to-model>", providers=provider_list)
        ```

        In this case we will prefer CUDA Execution Provider over CPU Execution Provider if both are available.

7. As you have probably realised in the exercises [on docker](../s3_reproducibility/docker.md), it can take a long time
    to build the kind of containers we are working with and they can be quite large. There is a reason for this and that
    is that PyTorch is a very large framework with a lot of dependencies. ONNX on the other hand is a much smaller
    framework. This kind of makes sense, because PyTorch is a framework that primarily was designed for developing e.g.
    training models, while ONNX is a framework that is designed for serving models. Let's try to quantify this.

    1. Construct a dockerfile that builds a docker image with PyTorch as a dependency. The dockerfile does actually
        not need to run anything. Repeat the same process for the ONNX runtime. Bonus point for developing a docker
        image that takes a [build arg](https://docs.docker.com/build/guide/build-args/) at build time that specifies
        if the image should be built with CUDA support or not.

        ??? success "Solution"

            The dockerfile for the PyTorch image could look something like this

            ```dockerfile linenums="1" title="inference_pytorch.dockerfile"
            --8<-- "s7_deployment/exercise_files/inference_pytorch.dockerfile"
            ```

            and the dockerfile for the ONNX image could look something like this

            ```dockerfile linenums="1" title="inference_onnx.dockerfile"
            --8<-- "s7_deployment/exercise_files/inference_onnx.dockerfile"
            ```

    2. Build both containers and measure the time it takes to build them. How much faster is it to build the ONNX
        container compared to the PyTorch container?

        ??? success "Solution"

            On unix/linux you can use the [time](https://linuxize.com/post/linux-time-command/) command to measure
            the time it takes to build the containers. Building both images, with and without CUDA support, can be done
            with the following commands

            ```bash
            time docker build . -t pytorch_inference_cuda:latest -f inference_pytorch.dockerfile \
                --no-cache --build-arg CUDA=true
            time docker build . -t pytorch_inference:latest -f inference_pytorch.dockerfile \
                --no-cache --build-arg CUDA=
            time docker build . -t onnx_inference_cuda:latest -f inference_onnx.dockerfile \
                --no-cache --build-arg CUDA=true
            time docker build . -t onnx_inference:latest -f inference_onnx.dockerfile \
                --no-cache --build-arg CUDA=
            ```

            the `--no-cache` flag is used to ensure that the build process is not cached and ensure a fair comparison.
            On my laptop this respectively took `5m1s`, `1m4s`, `0m4s`, `0m50s` meaning that the ONNX
            container was respectively 7x (with CUDA) and 1.28x (no CUDA) faster to build than the PyTorch container.

    3. Find out the size of the two docker images. It can be done in the terminal by running the `docker images`
        command. How much smaller is the ONNX model compared to the PyTorch model?

        ??? success "Solution"

            As of writing the docker image containing the PyTorch framework was 5.54GB (with CUDA) and 1.25GB (no CUDA).
            In comparison the ONNX image was 647MB (with CUDA) and 647MB (no CUDA). This means that the ONNX image is
            respectively 8.5x (with CUDA) and 1.94x (no CUDA) smaller than the PyTorch image.

8. (Optional) Assuming you have completed the module on [FastAPI](../s7_deployment/apis.md) try creating a small
    FastAPI application that serves a model using the ONNX runtime.

    ??? success "Solution"

        Here is a simple example of how to create a FastAPI application that serves a model using the ONNX runtime.

        ```python linenums="1" title="onnx_fastapi.py"
        --8<-- "s7_deployment/exercise_files/onnx_fastapi.py"
        ```

This completes the exercises on the ONNX format. Do note that one limitation of the ONNX format is that is is based on
[ProtoBuf](https://protobuf.dev/), which is a binary format. A protobuf file can have a maximum size of 2GB, which means
that the `.onnx` format is not enough for very large models. However, through the use of
[external data](https://onnxruntime.ai/docs/tutorials/web/large-models.html) it is possible to circumvent this
limitation.

## BentoML

!!! note "BentoML cloud vs BentoML OSS"

    We are only going to be looking at the open-source version of BentoML in this module. However, BentoML also has a
    cloud version that makes it very easy to deploy models that are coded in BentoML to the cloud. If you are interested
    in this, you can check out the
    [BentoML cloud documentation](https://docs.bentoml.com/en/latest/guides/cloud/index.html). This business strategy
    of having an open-source product and a cloud product is very common in the machine learning space (HuggingFace,
    LightningAI, Weights and Biases etc.), because it allows companies to make money from the cloud product while still
    providing a free product to the community.

BentoML is a framework that is designed to make it easy to serve machine learning models. It is designed to be backend
agnostic, meaning that it can be used with any computational backend. It is also model agnostic, meaning that it can be
used with any machine learning model.

Let's consider a simple example of how to serve a model using BentoML. The following
[code snippet](https://docs.bentoml.com/en/latest/get-started/quickstart.html) shows how to serve a model that uses
the `transformers` library to summarize text.

```python
import bentoml
from transformers import pipeline

EXAMPLE_INPUT = (
    "Breaking News: In an astonishing turn of events, the small town of Willow Creek has been taken by storm as "
    "local resident Jerry Thompson's cat, Whiskers, performed what witnesses are calling a 'miraculous and gravity-"
    "defying leap.' Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped a record-breaking "
    "20 feet into the air to catch a fly. The event, which took place in Thompson's backyard, is now being investigated "
    "by scientists for potential breaches in the laws of physics. Local authorities are considering a town festival to "
    "celebrate what is being hailed as 'The Leap of the Century.'"
)

@bentoml.service(resources={"cpu": "2"}, traffic={"timeout": 10})
class Summarization:
    def __init__(self) -> None:
        self.pipeline = pipeline('summarization')

    @bentoml.api
    def summarize(self, text: str = EXAMPLE_INPUT) -> str:
        result = self.pipeline(text)
        return result[0]['summary_text']

```

In `BentoML` we organize our services in classes, where each class is a service that we want to serve. The two important
parts of the code snippet are the `@bentoml.service` and `@bentoml.api` decorators.

* The `@bentoml.service` decorator is used to specify the resources that the service should use and in general how the
    service should be run. In this case we are specifying that the service should use 2 CPU cores and that the timeout
    for the service should be 10 seconds.

* The `@bentoml.api` decorator is used to specify the API that the service should expose. In this case we are specifying
    that the service should have an API called `summarize` that takes a string as input and returns a string as output.

To serve the model using `BentoML` we can execute the following command, which is very similar to the command we used
to serve the model using FastAPI.

```bash
bentoml serve service:Summarization
```

### ❔ Exercises

In general, we advise looking through the [docs](https://docs.bentoml.com/en/latest/index.html) for Bento ML if you
need help with any of the exercises. We are going to assume that you have done the exercises on ONNX and we are
therefore going to be using `BentoML` to serve ONNX models. If you have not done this part, you can still follow along
but you will need to use a PyTorch model instead of an ONNX model.

1. Install BentoML

    ```bash
    pip install bentoml
    ```

    Remember to add the dependency to your `requirements.txt` file.

2. You are in principal free to serve any model you like, but we recommend to just use a
    [torchvision](https://pytorch.org/vision/stable/index.html) model as in the ONNX exercises. Write your first service
    in `BentoML` that serves a model of your choice. We recommend experimenting with providing
    [input/output as tensors](https://docs.bentoml.com/en/latest/guides/iotypes.html) because bentoml supports this
    nativly. Secondly, write a client that can send a request to the service and print the result. Here we recommend
    using the build in [bentoml.SyncHTTPClient](https://docs.bentoml.com/en/latest/reference/client.html).

    ??? success "Solution"

        The following implements a simple BentoML service that serves a ONNX resnet18 model. The service expects the
        both input and output to be numpy arrays.

        ```python linenums="1" title="bentoml_service.py"
        --8<-- "s7_deployment/exercise_files/bentoml_service.py"
        ```

        The service can be served using the following command

        ```bash
        bentoml serve bentoml_service:ImageClassifierService
        ```

        To test that the service works the following client can be used

        ```python linenums="1" title="bentoml_client.py"
        --8<-- "s7_deployment/exercise_files/bentoml_client.py"
        ```

3. We are now going to look at features very `BentoML` really sets itself apart from `FastAPI`. The first is
    *adaptive batching*. As you are hopefully aware, modern machine learning models can process multiple samples at the
    same time and in doing so increases the throughput of the model. When we train a model we often set a fixed
    batch size, however we cannot do that when serving the model because that would mean that we would have to wait for
    the batch to be full before we can process it. *Adaptive batching* simply refers to the process where we specify a
    *maximum batch size* and also a *timeout*. When either the batch is full or the timeout is reached, however many
    samples we have collected are sent to the model for processing. This can be a very powerful feature because it
    allows us to process samples as soon as they arrive, while still taking advantage of the increased throughput of
    batching.

    <figure markdown>
    ![Image](../figures/bentoml_adaptive_batching.png){ width="700" }
    <figcaption>
    The overall architecture of the adaptive batching feature in BentoML. The feature is implemented on the server side
    and mainly consist of dispatcher that is in charge of collecting requests and sending them to the model server when
    either the batch is full or a timeout is reached.
    <a href="https://docs.bentoml.com/en/latest/guides/adaptive-batching.html"> Image credit </a>
    </figcaption>
    </figure>

    1. Look through the
        [documentation on adaptive batching](https://docs.bentoml.com/en/latest/guides/adaptive-batching.html) and
        add adaptive batching to your service from the previous exercise. Make sure your service works as expected by
        testing it with the client from the previous exercise.

        ??? success "Solution"

            ```python linenums="1" title="bentoml_service_adaptive_batching.py"
            --8<-- "s7_deployment/exercise_files/bentoml_service_adaptive_batching.py"
            ```

    2. Try to measure the throughput of your model with and without adaptive batching. Assuming that you have completed
        the [module on testing APIs](testing_apis.md) and therefore are familiar with the `locust` framework, we
        recommend that you write a simple locustfile and use the `locust` command to measure the throughput of your
        model.

        ??? success "Solution"

            The following locust file can be used to measure the throughput of the model with and without adaptive

            ```python linenums="1" title="locustfile.py"
            --8<-- "s7_deployment/exercise_files/locustfile_bentoml.py"
            ```

            and then the following command can be used to measure the throughput of the model

            ```bash
            locust -f locustfile_bentoml.py --host http://localhost:4040 --headless -u 50 -t 60s
            ```

            You should hopefully see that the throughput of the model is higher when adaptive batching is enabled, but
            the speedup is largely dependent on the model you are running, the configuration of the adaptive batching
            and the hardware you are running on.

            On my laptop I saw about a 1.5 - 2x speedup when adaptive batching was enabled.

4. (Optional, requires GPU) Look through the
    [documentation for inference on GPU](https://docs.bentoml.org/en/latest/guides/gpu-inference.html) and add this to
    your service. Check that your service works as expected by testing it with the client from the previous exercise and
    make sure you are seeing a speedup when running on the GPU.

    ??? success "Solution"

        A simple change to the `bento.service` decorator is all that is needed to run the model on the GPU.

        ```python
        @bentoml.service(resources={"gpu": 1})
        class MyService:
            def __init__(self):
                self.model = torch.load('model.pth').to('cuda:0')

5. Another way to speed up the inference is to just use multiple workers. This duplicates the server over multiple
    processes taking advantage of modern multi-core CPUs. This is similar to running `uvicorn` command with the
    `--workers` flag for fastapi applications. Implement multiple workers in your service and test that it works as
    expected by testing it with the client from the previous exercise. Also test that you are seeing a speedup when
    running with multiple workers.

    ??? success "Solution"

        Multiple workers can be added to the `bento.service` decorator as shown below.

        ```python
        @bentoml.service(workers=4)
        class MyService:
            # Service implementation
        ```

        Alternatively, you can set `workers="cpu_count"` to use all available CPU cores. The speedup depends on the
        model you are serving, the hardware you are running on and the number of workers you are using, but it should be
        higher than using a single worker.

6. In addition to increasing the throughput of your deployments `BentoML` can also help with ML applications that
    requires some kind of composition of multiple models. It is very normal in production setups to have multiple models
    that either

    * Runs in a sequence, e.g. the output of one model is the input of another model. You may have a preprocessing
        service that preprocesses the data before it is sent to a model that makes a prediction.
    * Runs concurrently, e.g. you have multiple models that are run at the same time and the output of all the models
        are combined to make a prediction. Ensemble models are a good example of this.

    `BentoML` makes it easy to
    [compose multiple models together](https://docs.bentoml.org/en/latest/guides/model-composition.html).

    1. Implement two services that runs in a sequence e.g. the output of one service is used as the input of another
        service. As an example you can implement either some pre- or post-processing service that is used in conjunction
        with the model you have implemented in the previous exercises.

        ??? success "Solution"

            The following code snippet shows how to implement two services that runs in a sequence.

            ```python linenums="1" title="bentoml_service_composition.py"
            --8<-- "s7_deployment/exercise_files/bentoml_service_composition_sequential.py"
            ```

    2. Implement three services, where two of them runs concurrently and the output of both services are combined in the
        third service to make a prediction. As an example you can expand your previous service to serve two different
        models and then implement a third service that combines the output of both models to make a prediction.

        ??? success "Solution"

            The following code snippet shows how to implement a service that consist of two concurrent services. The
            example assumes that two models called `model_a.onnx` and `model_b.onnx` are available.

            ```python linenums="1" title="bentoml_service_composition.py"
            --8<-- "s7_deployment/exercise_files/bentoml_service_composition_concurrent.py"
            ```

    3. (Optional) Implement a server that consist of both sequential and concurrent services.

7. Similar to deploying a FastAPI application to the cloud, deploying a `BentoML` framework to the cloud
    often requires you to first containerize the application. Because `BentoML` is designed to be easy to use for even
    users not that familiar with Docker, it introduces the concept of a `bentofile`. A `bentofile` is a file that
    specifies how the container should be build. Below is an example of how a `bentofile` could look like.

    ```yaml
    service: 'service:Summarization'
    labels:
      owner: bentoml-team
      project: gallery
    include:
      - '*.py'
    python:
      packages:
        - torch
        - transformers
    ```

    which can then be used to build a `bento` using the following command

    ```bash
    bentoml build
    ```

    A `bento` is not a docker image, but it can be used to build a docker image with the following command

    ```bash
    bentoml containerize summarization:latest
    ```

    1. Can you figure out how the different parts of the `bentofile` are used to build the docker image? Additionally,
        can you figure out from the [source repository](https://github.com/bentoml/BentoML) how the `bentofile` is
        used to build the docker image?

        ??? success "Solution"

            The `service` part specifies both what the container should be called and also what service it should
            serve e.g. the last statement in the corresponding dockerfile is
            `CMD ["bentoml", "serve", "service:Summarization"]`. The `labels` part is used to specify labels about the
            container, see this [link](https://docs.docker.com/reference/dockerfile/#label) for more info. The `include`
            part corresponds to `COPY` statements in the dockerfile and finally the `python` part is used to specify
            what python packages should be installed in the container which corresponds to `RUN pip install ...` in the
            dockerfile.

            Regarding how the `bentofile` is used to build the docker image, the `bentoml` package contains a number
            of templates (written using the [jinja2](https://jinja.palletsprojects.com/en/stable/) templating language)
            that are used to generate the dockerfiles. The templates can be found
            [here](https://github.com/bentoml/BentoML/tree/main/src/bentoml/_internal/container/frontend/dockerfile).

    2. Take whatever service from the previous exercises and try to containerize it. You are free to either write a
        `bentofile` or a `dockerfile` to do this.

        ??? success "Solution"

            The following `bentofile` can be used to containerize the very first service we implemented in this set of
            exercises.

            ```yaml
            service: 'bentoml_service:ImageClassifierService'
            labels:
              owner: bentoml-team
              project: gallery
            include:
            - 'bentoml_service.py'
            - 'model.onnx'
            python:
              packages:
                - onnxruntime
                - numpy
            ```

            The corresponding dockerfile would look something like this

            ```dockerfile
            FROM python:3.11-slim
            WORKDIR /bento
            COPY bentoml_service.py .
            COPY model.onnx .
            RUN pip install onnxruntime numpy bentoml
            CMD ["bentoml", "serve", "bentoml_service:ImageClassifierService"]
            ```

    3. Deploy the container to GCP Run and test that it works.

        ??? success "Solution"

            The following command can be used to deploy the container to GCP Run. We assume that you have already build
            the container and called it `bentoml_service:latest`.

            ```bash
            docker tag bentoml_service:latest \
                <region>-docker.pkg.dev/<project-id>/<repository-name>/bentoml_service:latest
            docker push <region>-docker.pkg.dev/<project-id>/<repository-name>/bentoml_service:latest
            gcloud run deploy bentoml-service \
                --image=<region>-docker.pkg.dev/<project-id>/<repository-name>/bentoml_service:latest \
                --platform managed \
                --port 3000  # default used by BentoML
            ```

            where `<project-id>` should be replaced with the id of the project you are deploying to. The service should
            now be available at the URL that is printed in the terminal.

This completes the exercises on the `BentoML` framework. If you want to deep dive more into this we can recommend
looking into their [tasks feature](https://docs.bentoml.org/en/latest/guides/tasks.html) for use cases that have a
very long running time and build in
[model management feature](https://docs.bentoml.org/en/latest/guides/model-loading-and-management.html) to unify the
way models are loaded, managed and served.

## 🧠 Knowledge check

1. How would you export a `scikit-learn` model to ONNX? What method is exported when you export `scikit-learn` model to
    ONNX?

    ??? success "Solution"

        It is possible to export a `scikit-learn` model to ONNX using the `sklearn-onnx` package. The following code
        snippet shows how to export a `scikit-learn` model to ONNX.

        ```python
        from sklearn.ensemble import RandomForestClassifier
        from skl2onnx import to_onnx
        model = RandomForestClassifier(n_estimators=2)
        dummy_input = np.random.randn(1, 4)
        onx = to_onnx(model, dummy_input)
        with open("model.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        ```

        The method that is exported when you export a `scikit-learn` model to ONNX is the `predict` method.

2. In your own words, describe what the concept of *computational graph* means?

    ??? success "Solution"

        A computational graph is a way to represent the mathematical operations that are performed in a model. It is
        essentially a graph where the nodes are the operations and the edges are the data that is passed between them.
        The computational graph normally represents the forward pass of the model and is the reason that we can easily
        backpropagate through the model to train it, because the graph contains all the necessary information to
        calculate the gradients of the model.

3. In your own words, explain why fusing operations together in the computational graph often leads to better
    performance?

    ??? success "Solution"

        Each time we want to do a computation, the data needs to be loaded from memory into the CPU/GPU. This is a
        slow process and the more operations we have, the more times we need to load the data. By fusing operations
        together, we can reduce the number of times we need to load the data, because we can do multiple operations
        on the same data before we need to load new data.

This ends the module on tools specifically designed for serving machine learning models. As stated in the beginning of
the module, there are a lot of different tools that can be used to serve machine learning models and the choice of tool
often depends on the specific use case. In general, we recommend that whenever you want to serve a machine learning
model, you should try out a few different frameworks and see which one fits your use case the best.
