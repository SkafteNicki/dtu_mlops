# 1. Introduction to PyTorch

The intention behind the first set of exercises is to bring everyones 
Pytorch skills up-to-date. If you already are Pytorch-Jedi feel free to
pass the first set of exercises, but I recommend still that you still complete it.

The exercises are in large part taken directly from the 
[deep learning course at udacity](https://github.com/udacity/deep-learning-v2-pytorch).
Note that these exercises are given as notebooks, in a large part of the course you
are expected to write your code in scripts.

The notebooks contains a lot of explaining text. The exercises that you are
supposed to fill out are inlined in the text in small "exercise" blocks:

![exercise](../figures/exercise.PNG)

If you need a fresh up on any deep learning topic in general throughout the course, we recommend to find the relevant 
chapter in the [deep learning](https://www.deeplearningbook.org/) book by Ian Goodfellow, 
Yoshua Bengio and Aaron Courville (can also be found in the literature folder).

## Exercises

1. Complete the [Tensors in Pytorch](1_Tensors_in_PyTorch.ipynb) notebook. It focuses on basic
   manipulation of Pytorch tensors. You can pass this notebook if you are comfortable doing this.
   
   1.1. (Bonus exercise): Efficiently write a function that calculates the pairwise squared distance
        between an `[N,d]` tensor and `[M,d]` tensor. You should use the following identity:
        ``` ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b> ```. Hint: you need to use broadcasting.
   
2. Complete the [Neural Networks in Pytorch](2_Neural_Networks_in_PyTorch.ipynb) notebook. It focuses on 
   building a very simple neural network using the Pytorch `nn.Module` interface.
   
   2.1 (Bonus exercise): One layer that arguably is missing in Pytorch is for doing reshapes.
       It is of course possible to do this directly to tensors, but sometimes it is great to
       have it directly in a `nn.Sequential` module. Write a `Reshape` layer which `__init__`
       takes a variable number arguments e.g. `Reshape(2)` or `Reshape(2,3)` and the forward
       takes a single input `x` where the reshape is applied to all other dimensions than the
       batch dimension.

3. Complete the [Training Neural Networks](3_Training_Neural_Networks.ipynb) notebook. It focuses on
   how to write a simple training loop for training a neural network.
   
   3.1 (Bonus exercise): A working training loop in Pytorch should have these three function calls:
       ``optimizer.zero_grad()``, ``loss.backward()``, ``optimizer.step()``. Explain what would happen
       in the training loop (or implement it) if you forgot each of the function calls.

   3.2 (Bonus exercise): Many state-of-the-art results depend on the concept of learning rate schedulers.
       In short a learning rate scheduler go in and either statically or dynamically changes the learning
       rate of your optimizer, such that training speed is either increased or decreased. Implement a 
       [learning rate scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
       in the notebook.
   
4. Complete the [Fashion MNIST](4_Fashion_MNIST.ipynb) notebook, that summaries concepts learned in the
   notebook 2 and 3 on building a neural network for classifying the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) 
   dataset.
   
   4.1 (Bonus exercise): The exercise focuses on the Fashion MNIST dataset but should without much
       work be able to train on multiple datasets. Implement a variable `dataset` that can take the
       values `mnist`, `fashionmnist` and `cifar` and train a model on the respective dataset.

5. Complete the [Inference and Validation](5_Inference_and_Validation.ipynb) notebook. This notebook adds
   important concepts on how to do inference and validation on our neural network.
   
   5.1 (Bonus exercise): The exercise shows how dropout can be used to prevent overfitting. However, today it
       is often used to get uncertainty estimates of the network predictions using [Monte Carlo Dropout](http://proceedings.mlr.press/v48/gal16.pdf).
       Implement monte carlo dropout such that we at inference time gets different predictions for the same
       input (HINT: do not set the network in evaluation mode). Construct a histogram of class prediction for a
       single image using 100 monte carlo dropout samples.

6. Complete the [Saving_and_Loading_Models](6_Saving_and_Loading_Models.ipynb) notebook. This notebook addresses
   how to save and load model weights. This is important if you want to share a model with someone else.

   6.1 (Bonus exercise): Being able to save and load weights are important for the concept of early stopping. In
       short, early stopping monitors some metric (often on the validation set) and then will stop the training
       and save a checkpoint when this metric have not improve for `N` steps. Implement early stopping in one of
       the previous notebooks.

### Final exercise

As the final exercise we will develop an simple baseline model which we will continue to develop on during the course.

#### Goal: 
Implement a mnist convolutional neural network that achives atlest 95 % accuracy on the test set.

We will already in this exercise start to think about how to organise our code and
you shall therefore complete the following subgoals
1. Implement your model in a script called `model.py`
2. Implement your data setup in a script called `data.py`
3. Implement training and evaluation of your model in `main.py` script
   
   3.1 Your script should be able to take an additional argument indicating if the model
   should train or evaluate. Something like:
   ```
   python main.py train
   python main.py evaluate
   ```
   which can be implemented in various ways.
   
To start you off, a very barebone version of each script is provided in the `final_exercise` folder. As documentation 
that your model is actually working, when running in the `train` command the script needs to
produce a single plot with the training curve (training step vs training loss). When the `eval` command is run,
it should write the test set accuracy to the terminal.

It is part of the exercise to not implement in notebooks as code development in the real life 
happens in script. As the model is simple to run (for now) you should be able to complete
the exercise on your laptop, even if you are only training on cpu. That said you are allowed to upload your scripts
to your own "Google Drive" and then you can call your scripts from a google colab notebook, which is shown in the image
below where all code is place in the `fashion_trainer.py` script and the colab notebook is just used to execute it.

![colab](../figures/colab.PNG)

Be sure to have completed the final exercise before the next session, as we will be building on top of the model
you have created.
