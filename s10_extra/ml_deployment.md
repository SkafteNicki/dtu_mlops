![Logo](../figures/icons/onnx.png){ align=right width="130"}

# Onnx

---

Whenever we want to serve an machine learning model, what we are actually interested in is doing *predictions* e.g.
given a new datapoint we pass it through our model (forward pass) and the returned value is the predicted value of
that datapoint. At a high-level, model predictions depends on three things:

* The codebase that implements the models prediction method
* The model weights which contains an actual instance of the model
* Code dependencies necessary for running the codebase.

We have already in module [M9 on Docker](../s3_reproducibility/docker.md) touch on how to take care of all
these things. Containers makes it easy to link a codebase, model weights and code dependencies into a single object.
We in general can refer to this as *model packaging*, because as the name suggest, we are packaging our model into
a format that is *independent* of the actual environment that we are trying to run the model in.

However, containers is not the only way to do model packaging. If we put some light restrictions on the device we want
run our model predictions on, we can achieve the same result using ONNX. The
[Open Neural Network Exchange (ONNX)](https://onnx.ai/) is a standardized format for creating and sharing machine
learning models. ONNX provides an [open source format](https://github.com/onnx/onnx) for machine learning models,
both deep learning and traditional ML. It defines an extensible computation graph model, as well as definitions of
built-in operators and standard data types.

<figure markdown>
![Image](../figures/onnx.png){ width="600" }
<figcaption> <a href="https://www.xenonstack.com/blog/onnx"> Image credit </a> </figcaption>
</figure>

As the above image indicates, the idea behind ONNX is that a model trained with a specific framework on a specific
device, lets say Pytorch on your local computer, can be exported and run with an entirely different framework and
hardware easily. For example, not all frameworks are created equally. For example Pytorch is in general considered
an developer friendly framework, however it has historically been slow to run inference with compared to a framework
such as [Caffe2](https://caffe2.ai/). ONNX allow you to mix-and-match frameworks based on different usecases, and
essentially increases the longivity of your model.

## ❔ Exercises

1. Start by installing ONNX:

    ```bash
    pip install onnx onnxruntime onnxscript
    ```

    the first package includes the basic building blocks for implementing generalized ONNX models and the second
    package is for running ONNX optimal on different hardware.

2. export


    === "Pytorch"
        ```python
        import torch
        import torchvision
        import onnx
        import onnxruntime

        model = torchvision.models.resnet18(pretrained=True)
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(model, dummy_input, "resnet18.onnx")
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
        torch.onnx.export(model, dummy_input, "resnet18.onnx")
        ```

    !!! note "Export"

        There is a new package called `onnxscript` that is currently in beta. This package is designed to make it

2. As an test that your installation is working, try executing the following Python code

    ```python
    import onnxruntime
    print(onnxruntime.get_all_providers())
    ```

    these providers are *translation layers* that are implemented ONNX, such that the same ONNX model can run on
    completely different hardware. Can you identify at least two of the providers that are necessary for running
    standard Pytorch code on CPU and GPU? Can you identify others

3. One big advantage of having a standardized format, is that we can easily visualize the computational graph of our
   model because it consist only of core ONNX operations. We are here going to use the open-source tool
   [netron](https://github.com/lutzroeder/netron) for visualization. You can either choose to download the program
   or just run it in your [webbrowser](https://netron.app/).

4. fafa

    ```python
    import onnxruntime as ort
    ort_session = ort.InferenceSession(<path-to-model>)
    input_names = [i.name for i in ort_session.get_inputs()]
    output_names = [i.name for i in ort_session.get_outputs()]
    batch = {input_names[0]: np.random.randn(1, 3, 224, 224).astype(np.float32)}
    ort_session.run(output_names, batch)

    ```

6. As you have probably relised in the exercises [on docker](../s3_reproducibility/docker.md), it can take a long time
    to build the kind of containers we are working with and they can be quite large. There is a reason for this and that
    is that Pytorch is a very large framework with a lot of dependencies. ONNX on the other hand is a much smaller
    framework. This kind of makes sense, because Pytorch is a framework that primarily was designed for developing e.g.
    training models, while ONNX is a framework that is designed for serving models. Let's try to quantify this.

    1. Construct a dockerfile that builds a docker image with Pytorch as a depdendency. The dockerfile does actually
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

## 🧠 Knowledge check

1. 


This ends the module on tools specifically designed for serving machine learning models.