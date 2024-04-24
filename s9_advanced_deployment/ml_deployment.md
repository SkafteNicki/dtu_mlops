# Machine learning specific serving

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

* [Torchserve](https://pytorch.org/serve/)

* [Tensorflow serve](https://github.com/tensorflow/serving)

The first 4 frameworks are backend agnostic, meaning that they are intended to work with whatever computational backend
you model is implemented in (Tensorflow vs PyTorch vs Jax), whereas the last two are backend specific to respective
Pytorch and Tensorflow. In this module we are going to look at the Triton Inference Server.

<figure markdown>
![](../figures/triton_architechture.jpg){ width="800" }
<figcaption>
The triton high-level architecture. The triton server communicates with a persistent model registry of models that can be served for inference. The server is in charge of receiving the incoming requests and the propergate it to the correct model, send the request to available hardware and then return the result to the client.
<a
href="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html">
Image Credit.
</a>
</figcaption>
</figure>



## Torch script

If you have already completed the privious module on [ONNX](onnx.md) then you are already familiar with the concept of
converting a model to a format that is more suitable for deployment. 


## Triton inference server

> Triton Inference Server is an open source inference serving software that streamlines AI inferencing.

At the core of the triton inference server is the concept of a model repository. This is a directory where you place
your models and a configuration file that tells the server how to load the model. The server supports many different
model formats:




=== "Torchserve"

    ```txt
    <model-repository-path>/
        <model-name>/
            config.pbtxt
            <version-number>/
                model.pt
            <version-number>/
                model.pt
        <model-name>/
            ...
    ```

=== "ONNX"

    If you have completed the privious [module on ONNX](onnx.md) the you can just place the `.onnx` file in the model
    repository as triton-inference server also supports ONNX models.

    ```txt
    <model-repository-path>/
        <model-name>/
            config.pbtxt
            <version-number>/
                model.onnx
            <version-number>/
                model.onnx
        <model-name>/
            ...
    ```

Regardless of format, the overall structure is the same: the inference server can serve multiple models at the same time
and therefore each model has its own directory. Inside each model directory there is a `config.pbtxt` file that tells the
server how to load the model. The `config.pbtxt` file is a protobuf file that contains the following information:

* `name`: the name of the model
* `platform`: the platform that the model is running on (e.g. `onnxruntime_onnx`)
* `max_batch_size`: the maximum batch size that the model can handle
* `input`: the input tensor names and shapes
* `output`: the output tensor names and shapes
* `version_policy`: the version policy for the model (e.g. `latest`)
* `dynamic_batching`: whether dynamic batching is enabled
* `optimization`: whether the model is optimized for inference


## ❔ Exercises

1. Installing triton inference server on your local machine can be a real hassel. The good new is that triton inference
    server is available as a docker container. 

    1.1. Install docker on your local machine.

    1.2. Pull the triton inference server docker container from the [Nvidia docker hub](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver).

    1.3. Run the triton inference server container on your local machine.

2. Assuming that you have already done the module on [using the cloud](../s)


2. Triton server can serve directly from a Google Cloud Storage bucket. Lets try to setup that:

    1. Create a Google Cloud Storage bucket.

    2. Upload your model to the bucket.

    3. When starting the triton server you can specify the model repository as a Google Cloud Storage bucket:
    
        ```bash
        tritonserver --model-repository=gs://bucket/path/to/model/repository ...
        ```

        You will need to give triton credentials to access the bucket. You can do this by setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of a service account key file.

        Note that triton will make a local copy of the model on your machine, so you will need to have enough space on your machine to store the model.

3. A great feature of triton is that it comes with a build in [performance analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/README.html) for measuring and optimizing the performance of your models. 

    1. We are going to assume that you have a triton server instance running on you local machine. If you do:

        ```curl -v localhost:8000/v2/health/ready```

        this should return a `200 OK` response.

    2. Lets pull the triton sdk container and run it on your local machine:

        ```bash
        docker pull nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
        ```

        

        ```bash
        docker run --gpus all --rm -it --net host nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
        ```

        This should give you a shell inside the container. Then you should be able to run the performance analyzer:

        ```bash
        perf_analyzer -m simple
        ```

## 🧠 Knowledge check
