---
layout: default
title: M22 - Local deployment
parent: S8 - Deployment
nav_order: 1
---

# Local Deployment
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



## Deploying the model with torchserve

Torchserve is Pytorch own framework for deploying/serving models. It can be a bit rough around the edges but
is fairly easy to work with. We are largely going to follow the instructions listed in the 
[readme file](https://github.com/pytorch/serve/blob/master/README.md#serve-a-model) for torchserve. The intention
is to serve a Resnet type neural network that is trained for classification on [ImageNet](https://www.image-net.org/).

1. Install `torchserve` and its dependencies. I recommend installing version 0.2.0 as the newest version (0.3.0)
   seems to have some incompatilities with the rest of the torch ecosystem. There are separate instructions 
   [here](https://github.com/pytorch/serve#install-torchserve-and-torch-model-archiver) if you are on linux/mac
   vs. windows.
   
2. Create a folder called `model_store`. This is where we will store the model that we are going to deploy

3. Try to run the `torchserve --model-store model_store` command. If the service starts with no errors, you
   have installed it correctly and can continue the exercise. Else it is googling time!

2. We need a model to serve. The problem however is that we cannot give a `torchserve` a raw file of trained
   model weights as these essentially is just a list of floats. We need a file that both contains the model
   definition and the trained weights. For this we are going to use `TorchScript`, Pytorchs build-in way
   to create serializable models. The great part about scriptet models are:
   
   * TorchScript code can be invoked in its own interpreter, which is basically a restricted Python interpreter. 
     This interpreter does not acquire the Global Interpreter Lock, and so many requests can be processed on the 
     same instance simultaneously.
   * This format allows us to save the whole model to disk and load it into another environment, such as in a 
     server written in a language other than Python
   * TorchScript gives us a representation in which we can do compiler optimizations on the code to provide 
     more efficient execution
   * TorchScript allows us to interface with many backend/device runtimes that require a broader view of the 
     program than individual operators.

   Luckily `TorchScript` is very easy to use. Choose a resnet model from `torchvision` package and script it
   
   ```python
   model = ResnetFromTorchVision(pretrained=True)
   script_model = torch.jit.script(model)
   script_model.save('deployable_model.pt')
   ```

3. Check that output of the scripted model corresponds to output of the non-scripted model. You can do this on
   a single random input, and you should check that the top-5 predicted indices are the same e.g.
   ```python
   assert torch.allclose(unscripted_top5_indices, scripted_top5_indices)
   ```
   (HINT: [torch.topk](https://pytorch.org/docs/stable/generated/torch.topk.html))

4. Call the model archiver. We have provided a file called `index_to_name.json` that maps from predicted class
   indices to interpretable class name e.g. `1->"goldfish"`. This file should be provided as the `extra-files` 
   argument such that the deployed model automatically outputs the class name. Note that this files of course
   only works for models trained on imagenet.
   ```
   torch-model-archiver \
       --model-name my_fancy_model 
       --version 1.0 \
       --serialized-file path/to/serialized_model.pt \
       --export-path model_store 
       --extra-files index_to_name.json 
       --handler image_classifier
   ```

5. Checkout the `model_store` folder. Has the model archiver correctly created a model (with `.mar` extension)
   inside the folder?

6. Finally, we are going to deploy our model and use it:

   6.1. Start serving your model in one terminal:
        ```
        torchserve --start --ncs --model-store model_store --models my_fancy_model=my_fancy_model.mar
        ```
       
   6.2. Next, pick a image that you want to do inference on. It can be any image that you want but try to pick
        one that actually contains an object from the set of imagenet classes. I have also provided a image of
        my own cat in the `my_cat.jpg` file.
       
   6.3. Open another terminal, which we are going to use for inference. The easiest way to do inference is using
        `curl` directly in the terminal but you are also free to experiment with the `requests` API directly in
        python. Using `curl` should look something like this
        ```
        curl http://127.0.0.1:8080/predictions/my_fancy_model -T my_image.jpg
        ```

7. (Optional) One strategy that researchers often resort to when trying to push out a bit of extra performance
   is creating [ensembles](https://en.wikipedia.org/wiki/Ensemble_learning) of models. Before Alexnet, this was often the
   way that teams won the imagenet competition, by pooling together their individual models. Try creating and serving
   a ensemble model. HINT: We have already started creating a ensemble model in the `ensemblemodel.py` file.

