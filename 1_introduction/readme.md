# 1. Introduction to PyTorch

The intention behind the first set of exercises is to bring everyones 
Pytorch skills up-to-date. If you already are Pytorch-jedi feel free to
pass the first exercise, but I recommend still that you still complete it.

The exercises are in large part taken directly from the 
[deep learning course at udacity](https://github.com/udacity/deep-learning-v2-pytorch).
Note that these exercises are given as notebooks, in a large part of the course you
are expected to write your code in scripts.

The notebooks contains a lot of explaining text. The exercises that you are
supposed to fill out are inlined in the text in small "exercise" blocks:

![exercise](../figures/exercise.PNG)

## Exercises

1. Complete the [Tensors in Pytorch](Tensors in PyTorch.ipynb) notebook. It focuses on basic
   manipulation of pytorch tensors. You can pass this notebook if you are conftable doing this.
   
   1.1. (Bonus exercise): Efficiently write a function that calculates the pairwise squared distance
        between an `[N,d]` tensor and `[M,d]` tensor. You should use the following identity:
        ``` ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b> ```. Hint: you need to use broadcasting.
   
2. Complete the [Neural Networks in PyTorch] notebook. It focuses on building a very simple
   neural network using the pytorch `nn.Module` interface.
   
   2.1 (Bonus exercise): One layer that argubly is missing in Pytorch is for doing reshapes.
       It is ofcause possible to do this directly to tensors, but sometimes it is great to
       have it directly in a `nn.Sequential` module. Write a `Reshape` layer which `__init__`
       takes a variable number arguments e.g. `Reshape(2)` or `Reshape(2,3)` and the forward
       takes a single input that it reshaped and returned.

3. Complete the Part 3 - Training Neural Networks (Exercises).ipynb notebook. It focuses on
   how to write a simple neural network for training. 

### Final exercise

As the final exercise we will develop an simple baseline model which we will
continue to develop on during the course.

#### Goal: 
Implement a mnist convolutional neural network that achives atlest 90 % accuracy
on the test set

We will already in this exercise start to think about how to organise our code and
you shall therefore complete the following subgoals
1. Implement your model in a script called `model.py`
2. Implement your data setup in a script called `data.py`
3. Implement training and evaluation of your model in `main.py` script
   
   3.1 Your script should be able to take an additional argument indicating if the model
   should train or evaluate. Something like:
   ```
   python main.py train
   python main.py eval
   ```
   which can be implemented in various ways.
   
   

As documentation that your model is actually working, when running in the `train` command the script needs to
produce a single plot with the training curve (training step vs training loss). When the `eval` command is run,
it should write the test set accuracy to the terminal.

It is part of the excersise to not implement in notebooks as code development in the real
world happens in script. As the model is simple to run (for now) you should be able to complete
the exercise on your laptop, even only using your cpu. That said you are allowed to upload your scripts
to your own "Google Drive" and then you can call your scripts from a google colab notebook.


In the next set of exercises we 






