# 08. Model deployment

Lets say that you have spend 1000 GPU hours and trained the most awesome model that you want to share with the
world. One way to do this is of course to just place all your code in a github repository, upload a file with
the trained model weights to your favorite online storage (assuming it is too big for github to handle) and
ask people to just download your code and the weights to run the code themself. This is a fine approach in small
research setting, but in production you need to be able to **deploy** the model to a environment that is fully 
contained such that people can just execute without looking (too hard) at the code. 

<p align="center">
  <img src="../figures/deployment.jpg" width="600" title="hover text">
</p>

The main focus today is to solve the exercises that has to do with Azure. The remaining two are not strictly
required but will introduce you to the concept of data drifing and how to natively serve pytorch models.

## Deployment on Azure

We are going to continue with Azure, where we this time focus on deploying a trained model

1. Complete this [exercise](https://docs.microsoft.com/en-us/learn/modules/register-and-deploy-model-with-amls/5-deploying-a-model)

2. Afterwards answer [this](https://docs.microsoft.com/en-us/learn/modules/register-and-deploy-model-with-amls/5a-knowledge-check)
   knowledge check.
   
3. Finally, try to deploy either your MNIST model or the model in your own project

Here is the general [learning module](https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-ml-service/)
from Microsoft about deploying models to Azure, and below is listed a couple of reading resourses:

* https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=azcli
* https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-azure-container-instance
* https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-azure-kubernetes-service?tabs=python

## Drifting data

A big part of deployment is also monitoring the performance of your deployed model. Especially, the concept of
detecting **data drifting** comes into play. Data drift is one of the top reasons model accuracy degrades over time. 
For machine learning models, data drift is the change in model input data that leads to model performance degradation. 
In practical terms this means that the model is receiving input that is outside of the scope that it was trained on. 

For this exercise we will rely on **TorchDrift** that contains state-of-the-art methods for detecting data
drifting in classification models. It is a model agnostic tool that should work with any black-box classification
model (in Pytorch). You can learn more about TorchDrift in this [video](https://www.youtube.com/watch?v=rV5BhoKILoE&t=1s).

### Exercises

1. Install TorchDrift `pip install torchdrift`

2. Look over the following [example](https://torchdrift.org/notebooks/drift_detection_on_images.html). It goes
   over the API of TorchDrift and better explains the concept of drifting data distributions
   
3. Implement drift detection on your MNIST example, this includes

    3.1 Start by creating artificially drifted MNIST data similar to the above example (HINT: apply Gaussian
        blur to the training data)
        
    3.2 Implement drift detection by going over the same steps as in the example. You can skip over a large
        part of it because you should already have a trained classifier ready to rock.
    
    3.4 While the `p-value` calculated by the drift detectors are great for having a single number for detecting
        if input is within spec, while developing it is beneficially to visualize the training data distribution
        and the drifting distribution. However, as the features we are visualizing are higher dimension than
        2, we need to resort to dimensionality reduction. In the example they use IsoMap. Try out other
        dimensionality reduction methods (HINT: [sklearn](https://scikit-learn.org/stable/modules/manifold.html))
    
    3.4 TorchDrift comes with multiple drift detectors and one may work better on your dataset than others. 
        Investigate the different choices of `kernel` parameter for the `KernelMMDDriftDetector`. You are
        supposed to produce one table that compares the `score` and `p-value` for the different detectors and a single
        plot with a subplot of embeddings of each detector similar to the figure below
    
    3.5 Repeat the exercise with some out-of-distribution data for example 
        [FashionMnist](https://github.com/zalandoresearch/fashion-mnist) dataset. Is it easier for the
        drift detectors to figure out true out-of-distribution data compared to the slightly blurred data?

![exercise](../figures/drifting_ex.PNG)

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
