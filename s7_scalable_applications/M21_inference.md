---
layout: default
title: M21 - Scalable Inference
parent: S7 - Scalable applications
nav_order: 3
mathjax: true
---

# Scalable Inference
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

Inference is task of applying our trained model to some new and unseen data, often called *prediction*. Thus, scaling inference is different from scaling data loading and training, mainly due to inference normally only using a single data point (or a few). As we can neither parallelize the data loading or parallelize using multiple GPUs (at least not in any efficient way), this is of no use to us when we are doing inference. Secondly, inference is often not something we do on machines that can perform large computations, as most inference today is actually either done on *edge* devices e.g. mobile phones or in low-cost-low-compute cloud environments. Thus, we need to be smarter about how we scale inference than just throwing more compute at it.

## Choosing the right architecture

When coming up with a model architectures we often look at prior work a make a copy or mix of this. It is a great way to get started, but not all model architectures
are created equal. Take `Distillbert` for example. `Distillbert` is a smaller version of the large natural-language procession model `Bert` that has been trained using
*model distillation*. Model distillation assumes that we already have a big model that performs well. By running our training set through our large model, we get input-output
pairs ${x_i,y_i}_{i=1}^N$ that we can train a small model to mimic. This is exactly what `Distillbert` successfully did, and as you can see in the figure below it is by far
the smallest model in newer time (not that I would call 66 million parameters for "small").

<p align="center">
   <img src="../figures/distill.png" width="600" title="All credit to https://arxiv.org/abs/1910.01108v4">
</p>

As discussed in this [blogpost](https://devblog.pytorchlightning.ai/training-an-edge-optimized-speech-recognition-model-with-pytorch-lightning-a0a6a0c2a413) the
probably largest increase in inference speed you will see (given some specific hardware) is choosing an efficient model architecture. Model distillation is just one way of coming up with more efficient architechtures. In the exercises below we are going to investigate the third generation of mobile nets `MobileNet v3`, which was constructed using a combination of efficient convolutional layers called inverted residual blocks, special swich non-linarity and neural architecture search. You can read more about it [here](https://towardsdatascience.com/everything-you-need-to-know-about-mobilenetv3-and-its-comparison-with-previous-versions-a5d5e5a6eeaa)

### Exercises

1. Write a small script that does inference with both `MobileNet V3 Large` and `ResNet-152` (pre-trained versions of both can be downloaded using `torchvision`) and try to time it. Do you see a difference in inference time? Can you figure out the performance difference between the two model architectures, and in your opinion is this high enough to justify an increase in inference time.

2. To figure out why one net is more efficient than another we can try to count the operations each network need to do for inference. A operation here we can define as a [FLOP (floating point operation)](https://en.wikipedia.org/wiki/FLOPS) which is any mathematical operation (such as +, -, *, /) or assignment that involves floating-point numbers. Luckily for us someone has already created a python package for calculating this in pytorch: [ptflops](https://github.com/sovrasov/flops-counter.pytorch)

   1. Install the package
      ```bash
      pip install ptflops
      ```

   2. Try calling the `get_model_complexity_info` function from the `ptflops` package on the two networks from the previous exercise. What are the results? How many times less operations does the mobile net need to perform compared to the resnet?

3. (Optional) Try out model distilation yourself. Assuming that you already have a trained conv net (on the corrupted mnist dataset), try to run all your training
   data through and record the log-probabilities that the model is predicting. Then design a small conv net, that you train to map from images to the log-probabilities
   that you recorded earlier. Finally, try out your small distilled model by measuring overall performance on the test set and compare to your original model.

## Quantization

Quantization is a technique where all computations are performed with integers instead of floats. We are essentially taking all continues signals and converting them into discretized signals.

<p align="center">
   <img src="../figures/quantization.png" width="300" title="All credit to https://arxiv.org/abs/1910.01108v4">
</p>

As discussed in this [blogpost series](https://devblog.pytorchlightning.ai/benchmarking-quantized-mobile-speech-recognition-models-with-pytorch-lightning-and-grid-9a69f7503d07), while `float` (32-bit) is the primarily used precision in machine learning because is strikes a good balance between memory consumption, precision and computational requirement it does not mean that during inference we can take advantage of quantization to improve the speed of our model. For instance:

* Floating-point computations are slower than integer operations

* Recent hardware have specialized hardware for doing integer operations

* Many neural networks are actually not bottlenecked by how many computations they need to do but by how fast we can transfer data e.g. the memory bandwidth and cache of your system is the limiting factor. Therefore working with 8-bit integers vs 32-bit floats means that we can approximately move data around 4 times as fast.

* Storing models in integers instead of floats save us approximately 75% of the ram/harddisk space whenever we save a checkpoint. This is especially useful in relation to deploying models using docker (as you hopefully remember) as it will lower the size of our docker images.

But how do we convert between floats and integers in quantization? In most cases we often use a *linear affine quantization*:

$$
x_{int} = \text{round}\left( \frac{x_{float}}{s} + z \right) 
$$

where $s$ is a scale and $z$ is the so called zero point. But how does to doing inference in a neural network. The figure below shows all the conversations that we need to make to our standard inference pipeline to actually do computations in quantized format.

<p align="center">
   <img src="../figures/quantization_overview.png" width="800" title="All credit to https://devblog.pytorchlightning.ai/how-to-train-edge-optimized-speech-recognition-models-with-pytorch-lightning-part-2-quantization-2eaa676b1512">
</p>

### Exercises

1. Lets look at how quantized tensors look in pytorch

   1. Start by creating a tensor that contains both random numbers

   2. Next call the `torch.quantize_per_tensor` function on the tensor. What does the quantized tensor
      look like? How does the values relate to the `scale` and `zero_point` arguments.

   3. Finally, try to call the `.dequantize()` method on the tensor. Do you get a tensor back that is
      close to what you initially started out with.

2. As you hopefully saw in the first exercise we are going to perform a number of rounding errors when doing quantization and naively we would expect that this would accumulate and lead to a much worse model. However, in practice we observe that quantization still works, and we actually have a mathematically sound reason for this. Can you figure out why quantization still works with all the small rounding errors? HINT: it has to do with the [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem) 

2. Lets move on to quantization of our model. 

   1. First we need to make sure that our model can actually be quantized by making it *statefull*. By this we mean that your model should fulfill the following:
      * No reusing of activation functions
      * Avoid using in-place operations
   
      Make sure that your model fulfills this

   2. Next, 

3. Lets try to benchmark our quantized model and see if all the trouble that we went through actually paid of. Below is shown some code that you need to adjust to yourself. Also try to perform the benchmark on the non-quantized model and see if you get a difference. 

```python
import torch
import torch.utils.benchmark

torch.backends.quantized.engine = "qnnpack"
q_model = my_model.quantized()
rand_inp = torch.randn(1, *input_shape)
tq = torch.utils.benchmark.Timer(
   setup='from __main__ import q_model, rand_inp',
   stmt='q_model(q_model.quant(inp)'
)
print(f"quantized {tq.timeit(200).median * 1000:.1f} ms")
```


3. (Optional) The quantization we have look on until now is a post-processing step, taking a trained model and converting it. However, quantization can be further implemented into our pipeline by doing `quantization aware training`, where we also apply quantization during training to hopefully get model that quantize better in the end. This can easily be done in lightning using the [QuantizationAwareTraining](https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.QuantizationAwareTraining.html#pytorch_lightning.callbacks.QuantizationAwareTraining) callback. Try it out!


## Compilation

If you ever coded in any low-level language such as c, fortran or c++ you should be familiar with the term *compiling*. Compiling is the task of taken a computer program written in one language and translating it into another. In most cases this means taken whatever you have written in your preferred programming language and translating it into machine code that the computer can execute. But what does compilation have to do with coding pytorch models?

We can take advantage of compilation because not all compiled programs are created equal. 


It happens to be that `Pytorch` comes with its own compiler that 

Jit stands for *just-in-time*, meaning that Jit compile refers to just-in-time compilation.


torch.jit.save

use the `torch.utils.benchmark.Timer` to time your application

### Exercises

1. To see the difference in the this exercises, we start out with a large model. Download one of large imagenet classification models from `torchvision` such as `ResNet-152` (remember to get the pretrained version).

2. Next try to script the model using `torch.jit.script`.

3. Finally, use `torch.utils.benchmark.Timer` to time both the standard model and jit-compiled version of the model. Do you see a decrease in time of the jit compiled model compared to the standard one? If so, what is the precentage increse in efficiency?


Thats all for this topic on doing scalable inference. If you want to further deep dive into this topic, I highly recommend that you also checkout methods such as [pruning](https://towardsdatascience.com/pruning-neural-networks-1bb3ab5791f9). Pytorch has an [build-in module](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) for doing this, which is another way to make models more efficient and thereby scalable.
