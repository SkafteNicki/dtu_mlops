![Logo](../figures/icons/onnx.png){ align=right width="130"}

# Quantization

---

!!! danger
    Module is still under development

## ❔ Exercises

We are in these exercises going to be looking at two different kinds of quantization strategies: quantization-aware
training and post-training quantization. As the names suggest, the quantization is either applied while training or
after training. There are good reasons for doing both:

* If the model you are going to deploy in the end needs to be quantized, either due to hard requirements for how
    big the model can be or in an effort to optimize inference time, quantization-aware training is the better
    approach. The reason here being that the model is specifically optimized to always be quantized and will
    therefore in general end up with a better model.

* If the most important metric for deployment is the overall performance of the model with no regard to model size
    and inference speed, post-training quantization is the better option. This most likely allows you to train a
    better model to begin with and then try out converting the model afterwards. In the best case this can be done
    without any hits to performance.

1. Start by installing [intel neural compressor](https://github.com/intel/neural-compressor)

    ```bash
    uv add neural-compressor
    ```

2. Let's start a new script called `model_converter.py`. Start by filling it with some simple code for loading a given
    `float32` model checkpoint. You should already have such code from earlier exercises. Preferably, add a small CLI
    interface to load a model by passing the filename in the command line:

    ```bash
    python model_converter.py model_checkpoint.ckpt
    ```

    ??? success "Solution"

        We are here going to assume that you are either loading from a `onnx` model or alternatively loading a PyTorch
        Lightning checkpoint, which we can distinguish based on the file extension:

        ```python
        import typer
        import onnx
        from my_model import MyModel

        app = typer.Typer()

        @app.command()
        def convert(model_checkpoint: str) -> None:
            if model_checkpoint.endswith(".ckpt"):
                model = MyModel.load_from_checkpoint(model_checkpoint)
            else:
                model = onnx.load(model_checkpoint)
            # the quantization of `model` is added in the next exercises

        if __name__ == "__main__":
            app()
        ```

3. Next you also need to add

4. Finally, calculate the size (in MB) of the original model and the quantized model. How much smaller is the quantized
    model?

    ??? success "Solution"

        Assuming the models are saved as `checkpoint.onnx` and `checkpoint_quantized.onnx` we can calculate the size
        using `os.path.getsize` in Python:

        ```python
        original_size = os.path.getsize("models/checkpoint.onnx") / (1024 * 1024)
        quantized_size = os.path.getsize("models/checkpoint_quantized.onnx") / (1024 * 1024)
        ```

        The quantized model should be very close to 8 times smaller as `int4` only uses 1/8 the bits to store weights
        compared to the `float32` format.

## 🧠 Knowledge check
