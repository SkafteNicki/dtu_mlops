# 1. Introduction to PyTorch

The intention behind the first set of exercises is to bring everyones 
Pytorch skills up-to-date. If you already are Pytorch-jedi feel free to
pass the first exercise, but I recommend still that you still complete it.

The exercises are in large part taken directly from the 
[deep learning course at udacity](https://github.com/udacity/deep-learning-v2-pytorch)

The part of the notebooks that you are intended to fill out are marked as exercises

![exercise](../figures/exercise.PNG)

## Exercises

1. Complete the [Tensors in Pytorch](Part 1 - Tensors in PyTorch (Exercises).ipynb) notebook. If you already 


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






