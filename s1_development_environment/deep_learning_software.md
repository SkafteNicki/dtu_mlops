![Logo](../figures/icons/pytorch.png){ align=right width="130"}

# Deep Learning Software

---

!!! info "Core Module"

Deep learning has, since its
[revolution back in 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html),
transformed our lives. From Google Translate to driverless cars to personal assistants to protein engineering, deep
learning is transforming nearly every sector of our economy and our lives. However, it did not take long before people
realized that deep learning is not a simple beast to tame and it comes with its own kinds of problems, especially if you
want to use it in a production setting. In particular, the concept of
[technical debt](https://research.google/pubs/pub43146/) was invented to indicate the significant maintenance costs at a
system level that it takes to run machine learning in production. MLOps should very much be seen as the response to the
concept of technical debt, namely that we should develop methods, processes, and tools (with inspiration from classical
DevOps) to counter the problems we run into when working with deep learning models.

It is important to note that all the concepts and tools that have been developed for MLOps can be used together with
more classic machine learning models (think K-nearest neighbor, Random forest, etc.), however, deep learning comes
with its own set of problems which mostly have to do with the sheer size of the data and models we are
working with. For these reasons, we are focusing on working with deep learning models in this course.

## Software Landscape for Deep Learning

Regarding software for Deep Learning, the landscape is currently dominated by three software frameworks (listed in order
of when they were published):

![Logo](../figures/tensorflow.png){ align=right width="120"}
![Logo](../figures/pytorch.png){ align=right width="130"}
![Logo](../figures/jax.png){ align=right width="200"}

* [TensorFlow](https://github.com/tensorflow/tensorflow)

* [PyTorch](https://github.com/pytorch/pytorch)

* [JAX](https://github.com/google/jax)

We won't go into a longer discussion on which framework is best, as it is pointless. PyTorch and TensorFlow have been
around for the longest and therefore have bigger communities and feature sets at this point in time. They are both very
similar in the sense that they both have features directed against research and production. JAX is kind of the new kid
on the block, which in many ways improves on PyTorch and TensorFlow, but it is still not as mature as the other frameworks.
As the frameworks use different kinds of programming principles (object-oriented vs. functional programming), comparing
them is essentially meaningless.

In this course, we have chosen to work with PyTorch because we find it a bit more intuitive and it is the framework that
we use for our day-to-day research life. Additionally, as of right now, it is absolutely the
[dominating framework](https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2022/) for published models, research
papers, and [competition winners](https://blog.mlcontests.com/p/winning-at-competitive-ml-in-2022?s=w).

The intention behind this set of exercises is to bring everyone's PyTorch skills up to date. If you already are a
PyTorch-Jedi, feel free to pass the first set of exercises, but I recommend that you still complete it. The exercises
are, in large part, taken directly from the
[deep learning course at Udacity](https://github.com/udacity/deep-learning-v2-pytorch). Note that these exercises are
given as notebooks, which is the last time we are going to use them actively in the course. Instead, after this set of
exercises, we are going to focus on writing code in Python scripts.

The notebooks contain a lot of explanatory text. The exercises that you are supposed to fill out are inlined in the text
in small "exercise" blocks:

<figure markdown>
![Image](../figures/exercise.PNG){width="1000"}
</figure>

If you need a refresher on any deep learning topic in general throughout the course, we recommend finding the relevant
chapter in the [deep learning](https://www.deeplearningbook.org/) book by Ian Goodfellow, Yoshua Bengio, and Aaron
Courville (which can also be found in the literature folder). It is not necessary to be good at deep learning to pass
this course as the focus is on all the software needed to get deep learning models into production. However, it's
important to have a basic understanding of the concepts.

### ❔ Exercises

<!-- markdownlint-disable -->
[Exercise files](https://github.com/SkafteNicki/dtu_mlops/tree/main/s1_development_environment/exercise_files){ .md-button }
<!-- markdownlint-restore -->

1. Start a Jupyter Notebook session in your terminal (assuming you are standing at the root of the course material).
    Alternatively, you should be able to open the notebooks directly in your code editor. For VS code users you can read
    more about how to work with Jupyter Notebooks in VS code
    [here](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

2. Complete the
    [Tensors in PyTorch](https://github.com/SkafteNicki/dtu_mlops/tree/main/s1_development_environment/exercise_files/1_Tensors_in_PyTorch.ipynb)
    notebook. It focuses on the basic manipulation of PyTorch tensors. You can pass this notebook if you are comfortable
    doing this.

3. Complete the
    [Neural Networks in PyTorch](https://github.com/SkafteNicki/dtu_mlops/tree/main/s1_development_environment/exercise_files/2_Neural_Networks_in_PyTorch.ipynb)
    notebook. It focuses on building a very simple neural network using the PyTorch `nn.Module` interface.

4. Complete the
    [Training Neural Networks](https://github.com/SkafteNicki/dtu_mlops/tree/main/s1_development_environment/exercise_files/3_Training_Neural_Networks.ipynb)
    notebook. It focuses on how to write a simple training loop for training a neural network.

5. Complete the
    [Fashion MNIST](https://github.com/SkafteNicki/dtu_mlops/tree/main/s1_development_environment/exercise_files/4_Fashion_MNIST.ipynb)
    notebook, which summarizes concepts learned in notebooks 2 and 3 on building a neural network for classifying the
    [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

6. Complete the
    [Inference and Validation](https://github.com/SkafteNicki/dtu_mlops/tree/main/s1_development_environment/exercise_files/5_Inference_and_Validation.ipynb)
    notebook. This notebook adds important concepts on how to do inference and validation on our neural network.

7. Complete the
    [Saving_and_Loading_Models](https://github.com/SkafteNicki/dtu_mlops/tree/main/s1_development_environment/exercise_files/6_Saving_and_Loading_Models.ipynb)
    notebook. This notebook addresses how to save and load model weights. This is important if you want to share a
    model with someone else.

## 🧠 Knowledge check

1. If tensor `a` has shape `[N, d]` and tensor `b` has shape `[M, d]` how can we calculate the pairwise distance
    between rows in `a` and `b` without using a for loop?

    ??? success "Solution"

        We can take advantage of [broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html) to do this

        ```python
        a = torch.randn(N, d)
        b = torch.randn(M, d)
        dist = torch.sum((a.unsqueeze(1) - b.unsqueeze(0))**2, dim=2)  # shape [N, M]
        ```

2. What should be the size of `S` for an input image of size 1x28x28, and how many parameters does the neural network
    then have?

    ```python
    from torch import nn
    neural_net = nn.Sequential(
        nn.Conv2d(1, 32, 3), nn.ReLU(), nn.Conv2d(32, 64, 3), nn.ReLU(), nn.Flatten(), nn.Linear(S, 10)
    )
    ```

    ??? success "Solution"

        Since both convolutions have a kernel size of 3, stride 1 (default value) and no padding that means that we lose
        2 pixels in each dimension, because the kernel can not be centered on the edge pixels. Therefore, the output
        of the first convolution would be 32x26x26. The output of the second convolution would be 64x24x24. The size of
        `S` must therefore be `64 * 24 * 24 = 36864`. The number of parameters in a convolutional layer is
        `kernel_size * kernel_size * in_channels * out_channels + out_channels` (last term is the bias) and the number
        of parameters in a linear layer is `in_features * out_features + out_features` (last term is the bias).
        Therefore, the total number of parameters in the network is
        `3*3*1*32 + 32 + 3*3*32*64 + 64 + 36864*10 + 10 = 387,466`, which could be calculated by running:

        ```python
        sum([prod(p.shape) for p in neural_net.parameters()])
        ```

3. A working training loop in PyTorch should have these three function calls: `optimizer.zero_grad()`,
    `loss.backward()`, `optimizer.step()`. Explain what would happen in the training loop (or implement it) if you
    forgot each of the function calls.

    ??? success "Solution"

        `optimizer.zero_grad()` is in charge of zeroring the gradient. If this is not done, then gradients would
        accumulate over the steps leading to exploding gradients. `loss.backward()` is in charge of calculating the
        gradients. If this is not done, then the gradients will not be calculated and the optimizer will not be able
        to update the weights. `optimizer.step()` is in charge of updating the weights. If this is not done, then the
        weights will not be updated and the model will not learn anything.

### Final exercise

As the final exercise, we will develop a simple baseline model that we will continue to develop during the course.
For this exercise, we provide the data in the `data/corruptmnist` folder. Do **NOT** use the data in the
`corruptmnist_v2` folder as that is intended for another exercise. As the name suggests, this is a (subsampled)
corrupted version of the regular [MNIST](https://en.wikipedia.org/wiki/MNIST_database). Your overall task is the
following:

> **Implement an MNIST neural network that achieves at least 85% accuracy on the test set.**

Before any training can start, you should identify the corruption that we have applied to the MNIST dataset to
create the corrupted version. This can help you identify what kind of neural network to use to get good performance, but
any network should be able to achieve this.

One key point of this course is trying to stay organized. Spending time now organizing your code will save time in the
future as you start to add more and more features. As subgoals, please fulfill the following exercises:

1. Implement your model in a script called `model.py`.

    ??? example "Starting point for `model.py`"

        ```python linenums="1" title="model.py"
        --8<-- "s1_development_environment/exercise_files/final_exercise/model.py"
        ```

    ??? success "Solution"

        The provided solution implements a convolutional neural network with 3 convolutional layers and a single
        fully connected layer. Because the MNIST dataset consists of images, we want an architecture that can take
        advantage of the spatial information in the images.

        ```python linenums="1" title="model.py"
        --8<-- "s1_development_environment/exercise_files/final_exercise/model_solution.py"
        ```

2. Implement your data setup in a script called `data.py`. The data was saved using `torch.save`, so to load it you
    should use `torch.load`.

    !!! warning "Saving the model"

        When saving the model, you should use `#!python torch.save(model.state_dict(), "model.pt")`, and when loading
        the model, you should use `#!python model.load_state_dict(torch.load("model.pt"))`. If you do
        `#!python torch.save(model, "model.pt")`, this can lead to problems when loading the model later on, as it will
        try to not only save the model weights but also the model definition. This can lead to problems if you change
        the model definition later on (which you most likely are going to do).

    ??? example "Starting point for `data.py`"

        ```python linenums="1" title="model.py"
        --8<-- "s1_development_environment/exercise_files/final_exercise/data.py"
        ```

    ??? success "Solution"

        Data is stored in `.pt` files which can be loaded using `torch.load` (1). We iterate over the files, load them
        and concatenate them into a single tensor. In particular, we have highlighted the use of `.unsqueeze` function.
        Convolutional neural networks (which we propose as a solution) need the data to be in the shape `[N, C, H, W]`
        where `N` is the number of samples, `C` is the number of channels, `H` is the height of the image and `W` is the
        width of the image. The dataset is stored in the shape `[N, H, W]` and therefore we need to add a channel.
        { .annotate }

        1. :man_raising_hand: The `.pt` files are nothing else than a `.pickle` file in disguise. The
            `torch.save/torch.load` function is essentially a wrapper around the `pickle` module in Python, which
            produces serialized files. However, it is convention to use `.pt` to indicate that the file contains PyTorch
            tensors.

        We have additionally in the solution added functionality for plotting the images together with the labels for
        inspection. Remember: all good machine learning starts with a good understanding of the data.

        ```python linenums="1" hl_lines="17 18" title="model.py"
        --8<-- "s1_development_environment/exercise_files/final_exercise/data_solution.py"
        ```

3. Implement training and evaluation of your model in `main.py` script. The `main.py` script should be able to take
    additional subcommands indicating if the model should be trained or evaluated. It will look something like this:

    ```bash
    python main.py train --lr 1e-4
    python main.py evaluate trained_model.pt
    ```

    which can be implemented in various ways. We provide you with a starting script that uses the `typer` library to
    define a command line interface (CLI), which you can learn more about in
    [this module](../s2_organisation_and_version_control/cli.md) later in the course.

    ??? note "VS code and command line arguments"

        If you try to execute the above code in VS code using the debugger (F5) or the build run functionality in the
        upper right corner:

        <figure markdown>
        ![Image](../figures/vs_code_run.png){width="300"}
        </figure>

        you will get an error message saying that you need to select a command to run e.g. `main.py` either needs the
        `train` or `evaluate` command. This can be fixed by adding a `launch.json` to a specialized `.vscode` folder in
        the root of the project. The `launch.json` file should look something like this:

        ```json
        {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Python: Current File",
                    "type": "python",
                    "request": "launch",
                    "program": "${file}",
                    "args": [
                        "train",
                        "--lr",
                        "1e-4"
                    ],
                    "console": "integratedTerminal",
                    "justMyCode": true
                }
            ]
        }
        ```

        This will inform VS code that then we execute the current file (in this case `main.py`) we want to run it with
        the `train` command and additionally pass the `--lr` argument with the value `1e-4`. You can read more about
        creating a `launch.json` file [here](https://code.visualstudio.com/docs/python/debugging). If you want to have
        multiple configurations you can add them to the `configurations` list as additional dictionaries.

    ??? example "Starting point for `main.py`"

        ```python linenums="1" title="main.py"
        --8<-- "s1_development_environment/exercise_files/final_exercise/main.py"
        ```

    ??? success "Solution"

        The solution implements a simple training loop and evaluation loop. Furthermore, we have added additional
        hyperparameters that can be passed to the training loop. Highlighted in the solution are the different lines
        where we take care that our model and data are moved to GPU (or Apple MPS accelerator if you have a newer Mac)
        if available.

        ```python linenums="1" hl_lines="8 26 38 69 78" title="main.py"
        --8<-- "s1_development_environment/exercise_files/final_exercise/main_solution.py"
        ```

4. As documentation that your model is working when running the `train` command, the script needs to produce a single
    plot with the training curve (training step vs training loss). When the `evaluate` command is run, it should write
    the test set accuracy to the terminal.

It is part of the exercise to not implement in notebooks, as code development in real life happens in scripts.
As the model is simple to run (for now), you should be able to complete the exercise on your laptop,
even if you are only training on CPU. That said, you are allowed to upload your scripts to your own "Google Drive" and
then you can call your scripts from a Google Colab notebook, which is shown in the image below where all code is
placed in the `fashion_trainer.py` script and the Colab notebook is just used to execute it.

![colab](../figures/colab.PNG)

Be sure to have completed the final exercise before the next session, as we will be building on top of the model
you have created.
