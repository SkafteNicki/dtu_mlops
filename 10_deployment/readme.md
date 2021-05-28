# 11. Model deployment

Lets say that you have spend 1000 GPU hours and trained the most awesome model that you want to share with the
world. One way to do this is ofcause to just place all your code in a github reposatory, upload a file with
the trained model weights to your favorit online storage (assuming it is too big for github to handle) and
ask people to run your code. This is a fine approach in a research setting, but in production you need to be
able to **deploy** the model to a enviroment that is fully contained that people can just execute without
looking (too hard) at the code. 

Today is all about deploying your model. The hope is that by the end of this exercise you will be able to
access your model online and to inference in real-time.

## Drifing data

One of the first question that often comes up is how to . This is where detection of **data drifing** comes 
into play. Data drift is one of the top reasons model accuracy degrades over time. For machine learning models,
data drift is the change in model input data that leads to model performance degradation. In practical terms
this means that the model is receiving input that is outside of the scope that it was trained on. 

For this exercise we will rely on **TorchDrift** that contains state-of-the-art methods for detecting data
drifting in classification models. It is a model agnostic tool that should work with any black-box classification
model (in Pytorch). You can learn more about TorchDrift in this [video](https://www.youtube.com/watch?v=rV5BhoKILoE&t=1s).

### Exercises

1. Install TorchDrift `pip install torchdrift`

2. Look over the following [example](https://torchdrift.org/notebooks/drift_detection_on_images.html). It goes
   over the API of TorchDrift and better explains the concept of drifting data distributions
   
3. Implement drift detection on your mnist example, this includes

    3.1 Start by creating artifically drifted mnist data similar to the above example (HINT: apply gaussian
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
        supposed to produce one table that compares the `p-value` for the different detectors and a single
        plot with a subplot of embeddings of each detector.
    
    3.5 Repeat the exercise with some out-of-distribution data for example 
        [FashionMnist](https://github.com/zalandoresearch/fashion-mnist) dataset. Is it easier for the
        drift detectors to figure out true out-of-distribution data compared to the slightly blurred data.

## Deploying the model with torchserve

Torchserve is Pytorch own framework for deploying/serving models. It can be a bit rough around the edges but
is fairly easy to work with. We are largely going to follow the instructions listed in the 
[readme file](https://github.com/pytorch/serve/blob/master/README.md#serve-a-model) for torchserve. The intention
is to serve a Resnet type neural network that is trained for classification on [ImageNet](https://www.image-net.org/).

1. Install `torchserve` either using pip or conda:
   ```
   pip install torchserve torch-model-archiver
   conda install torchserve torch-model-archiver -c pytorch
   ```
   
2. create a folder called `model_store`

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

   Lukely `TorchScript` is very easy to use. Choose a resnet model from `torchvision` and script it
   
   ```
   model = ResnetFromTorchVision(pretrained=True)
   script_model = torch.jit.script(model)
   script_model.save('deployable_model.pt')
   ```

3. We are going to serve a Resnet type neural network that is trained for classification on [ImageNet](https://www.image-net.org/).
   To create a model that we can serve we need both the model definition and a file containing the weights
   that we want do use in our served model. We have already provided a file called `resnet.py` (directly
   taken from `torchvision`) that contains the model definition. Inside the file you can see various paths
   to different versions of resnet. Download the model weights that you want to serve.

4. TorchScript is a way to create serializable and optimizable models from PyTorch code. Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency.


4. Call the model archiver
   ```
   torch-model-archiver \
       --model-name my_fancy_model 
       --version 1.0 \
       --model-file path/to/model_definition.py \ 
       --serialized-file path/to/model_weights.pth \
       --export-path model_store 
       --extra-files index_to_name.json 
       --handler image_classifier
   ```


1. Find someone else that can test your model for you. 


## Creating fully deployable packages

Torchserve all not by default create a fully contained enviroment that can be shipped of to production. This
would require to also include all dependencies nessesary ect. We therefore turn our attention now to
[bentoML](https://github.com/bentoml/BentoML) that was created with making model serving easy to do. 

Follow these instructions

https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python





