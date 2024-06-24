![Logo](../figures/icons/onnx.png){ align=right width="130"}

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
machine learning algorithms:

* [Cortex](https://github.com/cortexlabs/cortex)

* [Bento ML](https://github.com/bentoml/bentoml)

* [Ray Serve](https://docs.ray.io/en/master/serve/)

* [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)

* [OpenVINO](https://docs.openvino.ai/2024/index.html)

* [Seldon-core](https://docs.seldon.io/projects/seldon-core/en/latest/)

* [Torchserve](https://pytorch.org/serve/)

* [Tensorflow serve](https://github.com/tensorflow/serving)

The first 6 frameworks are backend agnostic, meaning that they are intended to work with whatever computational backend
you model is implemented in (Tensorflow vs PyTorch vs Jax), whereas the last two are backend specific to respective
Pytorch and Tensorflow. In this module we are going to look at two of the frameworks, namely `Torchserve` because we
have developed Pytorch applications i nthis course and `Triton Inference Server` because it is a very popular framework
for deploying models on Nvidia GPUs (but we can still use it on CPU).

But before we dive into these frameworks, we are going to look at a general way to package our machine learning models
that should work with any of the above frameworks.

## Model Packaging

Whenever we want to serve an machine learning model, we in general need 3 things:

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
a specific framework on a specific device, lets say Pytorch on your local computer, can be exported and run with an
entirely different framework and hardware easily. Learning how to export your models to ONNX is therefore a great way
to increase the longivity of your models and not being locked into a specific framework for serving your models.

<figure markdown>
![Image](../figures/onnx.png){ width="1000" }
<figcaption>
The ONNX format is designed to bridge the gap between development and deployment of machine learning models, by making
it easy to export models between different frameworks and hardware. For example Pytorch is in general considered
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

2. Let's start out with converting a model to ONNX. The following code snippets shows how to export a Pytorch model to
    ONNX.

    === "Pytorch => 2.0"

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

    === "Pytorch < 2.0 or Windows"

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

    === "Pytorch-lightning"

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

        When a Pytorch model is exported to ONNX, it is only the `forward` method of the model that is exported. This
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

    2. Let's experiment with performance of ONNX vs. Pytorch. Implement a benchmark that measures the time it takes to
        run a model using Pytorch and ONNX. Bonus points if you test for multiple input sizes. To get you started we
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
            --8<-- "s10_extra/exercise_files/onnx_benchmark.py"
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
    is that Pytorch is a very large framework with a lot of dependencies. ONNX on the other hand is a much smaller
    framework. This kind of makes sense, because Pytorch is a framework that primarily was designed for developing e.g.
    training models, while ONNX is a framework that is designed for serving models. Let's try to quantify this.

    1. Construct a dockerfile that builds a docker image with Pytorch as a dependency. The dockerfile does actually
        not need to run anything. Repeat the same process for the ONNX runtime. Bonus point for developing a docker
        image that takes a [build arg](https://docs.docker.com/build/guide/build-args/) at build time that specifies
        if the image should be built with CUDA support or not.

        ??? success "Solution"

            The dockerfile for the Pytorch image could look something like this

            ```dockerfile linenums="1" title="inference_pytorch.dockerfile"
            --8<-- "s10_extra/exercise_files/inference_pytorch.dockerfile"
            ```

            and the dockerfile for the ONNX image could look something like this

            ```dockerfile linenums="1" title="inference_onnx.dockerfile"
            --8<-- "s10_extra/exercise_files/inference_onnx.dockerfile"
            ```

    2. Build both containers and measure the time it takes to build them. How much faster is it to build the ONNX
        container compared to the Pytorch container?

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
            container was respectively 7x (with CUDA) and 1.28x (no CUDA) faster to build than the Pytorch container.

    3. Find out the size of the two docker images. It can be done in the terminal by running the `docker images`
        command. How much smaller is the ONNX model compared to the Pytorch model?

        ??? success "Solution"

            As of writing the docker image containing the Pytorch framework was 5.54GB (with CUDA) and 1.25GB (no CUDA).
            In comparison the ONNX image was 647MB (with CUDA) and 647MB (no CUDA). This means that the ONNX image is
            respectively 8.5x (with CUDA) and 1.94x (no CUDA) smaller than the Pytorch image.

8. (Optional) Assuming you have completed the module on [FastAPI](../s7_deployment/apis.md) try creating a small
    FastAPI application that serves a model using the ONNX runtime.

This completes the exercises on the ONNX format. Do note that one limitation of the ONNX format is that is is based on
[ProtoBuf](https://protobuf.dev/), which is a binary format. A protobuf file can have a maximum size of 2GB, which means
that the `.onnx` format is not enough for very large models. However, through the use of
[external data](https://onnxruntime.ai/docs/tutorials/web/large-models.html) it is possible to circumvent this
limitation.

## Torchserve

Text to come...

## Triton Inference Server

Text to come...

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

This ends the module on tools specifically designed for serving machine learning models.
